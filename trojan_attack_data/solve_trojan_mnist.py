import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


SEED = 1337
IMG_SIZE = 28
NUM_CLASSES = 10
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

SOURCE_CLASS = 7
TARGET_CLASS = 1

TRIGGER_SIZE = 3
TRIGGER_POS = (24, 1)
TRIGGER_VAL = 1.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def add_trigger(image_tensor: torch.Tensor) -> torch.Tensor:
    c, h, w = image_tensor.shape
    if c != 1 or h != IMG_SIZE or w != IMG_SIZE:
        return image_tensor

    start_y, start_x = TRIGGER_POS
    end_y = min(start_y + TRIGGER_SIZE, h)
    end_x = min(start_x + TRIGGER_SIZE, w)
    image_tensor[:, start_y:end_y, start_x:end_x] = TRIGGER_VAL
    return image_tensor


class PoisonedMNISTTrain(Dataset):
    def __init__(
        self,
        clean_dataset,
        source_class,
        target_class,
        poison_rate,
        trigger_func,
        transform_norm,
    ):
        self.data = []
        self.poisoned_indices_count = 0

        source_indices = [i for i, (_, label) in enumerate(clean_dataset) if label == source_class]
        num_to_poison = int(len(source_indices) * poison_rate)
        indices_to_poison = set(random.sample(source_indices, num_to_poison))
        self.poisoned_indices_count = len(indices_to_poison)

        for i in tqdm(range(len(clean_dataset)), desc="Building poisoned train set"):
            img_tensor, original_label = clean_dataset[i]
            final_label = original_label
            img_processed = img_tensor.clone()
            if i in indices_to_poison:
                img_processed = trigger_func(img_processed)
                final_label = target_class
            img_processed = transform_norm(img_processed)
            self.data.append((img_processed, final_label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TriggeredMNISTTest(Dataset):
    def __init__(self, clean_dataset, source_class, trigger_func, transform_norm):
        self.data = []
        self.triggered_count = 0

        for i in tqdm(range(len(clean_dataset)), desc="Building triggered test set"):
            img_tensor, original_label = clean_dataset[i]
            img_processed = img_tensor.clone()
            if original_label == source_class:
                img_processed = trigger_func(img_processed)
                self.triggered_count += 1
            img_processed = transform_norm(img_processed)
            self.data.append((img_processed, original_label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MNIST_CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._feature_size = 64 * 7 * 7
        self.fc1 = nn.Linear(self._feature_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, self._feature_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


@dataclass
class EvalResult:
    clean_accuracy: float
    asr: float


def evaluate_model(model, testloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if total == 0:
        return 0.0, 0.0
    accuracy = 100.0 * correct / total
    avg_loss = running_loss / total
    return accuracy, avg_loss


def calculate_asr(model, triggered_testloader, source_class, target_class, device):
    model.eval()
    misclassified_as_target = 0
    total_source_class_triggered = 0
    with torch.no_grad():
        for inputs, labels in triggered_testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            source_mask = labels == source_class
            if not source_mask.any():
                continue
            source_inputs = inputs[source_mask]
            outputs = model(source_inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_source_class_triggered += source_inputs.size(0)
            misclassified_as_target += (predicted == target_class).sum().item()
    if total_source_class_triggered == 0:
        return 0.0
    return 100.0 * misclassified_as_target / total_source_class_triggered


def train_model(model, trainloader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        seen = 0
        pbar = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            seen += inputs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        epoch_loss = running_loss / max(seen, 1)
        print(f"Epoch {epoch + 1}: avg loss={epoch_loss:.4f}")


def submit_model(model_path, url):
    with open(model_path, "rb") as f:
        files = {"model": (os.path.basename(model_path), f, "application/octet-stream")}
        response = requests.post(url, files=files, timeout=180)
    response.raise_for_status()
    return response.json()


def run_trial(args, poison_rate, num_epochs, learning_rate):
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    transform_base = transforms.Compose([transforms.ToTensor()])
    transform_norm = transforms.Compose([transforms.Normalize(MNIST_MEAN, MNIST_STD)])
    transform_test_clean = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)])

    trainset_clean_raw = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform_base,
    )
    testset_clean_raw = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform_base,
    )
    testset_clean_transformed = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform_test_clean,
    )

    trainset_poisoned = PoisonedMNISTTrain(
        clean_dataset=trainset_clean_raw,
        source_class=SOURCE_CLASS,
        target_class=TARGET_CLASS,
        poison_rate=poison_rate,
        trigger_func=add_trigger,
        transform_norm=transform_norm,
    )
    testset_triggered = TriggeredMNISTTest(
        clean_dataset=testset_clean_raw,
        source_class=SOURCE_CLASS,
        trigger_func=add_trigger,
        transform_norm=transform_norm,
    )

    trainloader_poisoned = DataLoader(
        trainset_poisoned,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    testloader_clean = DataLoader(
        testset_clean_transformed,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    testloader_triggered = DataLoader(
        testset_triggered,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = MNIST_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=args.weight_decay,
    )

    print(f"Training with poison_rate={poison_rate:.3f}, epochs={num_epochs}, lr={learning_rate}")
    train_model(model, trainloader_poisoned, criterion, optimizer, num_epochs, device)

    clean_acc, _ = evaluate_model(model, testloader_clean, criterion, device)
    asr = calculate_asr(model, testloader_triggered, SOURCE_CLASS, TARGET_CLASS, device)
    print(f"Local metrics -> CA: {clean_acc:.2f}%, ASR: {asr:.2f}%")

    model_path = args.model_out
    torch.save(model.state_dict(), model_path)

    result = submit_model(model_path, args.evaluator_url)
    print("Evaluator response:")
    print(result)
    return result, EvalResult(clean_accuracy=clean_acc, asr=asr)


def parse_args():
    parser = argparse.ArgumentParser(description="Solve the MNIST Trojan attack evaluator.")
    parser.add_argument("--evaluator-url", default="http://154.57.164.68:31913/evaluate")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model-out", default="mnist_cnn_trojaned.pth")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--poison-rate", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--auto-tune",
        action="store_true",
        help="Try multiple (poison_rate, epochs, learning_rate) settings until success.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.auto_tune:
        result, _ = run_trial(args, args.poison_rate, args.epochs, args.learning_rate)
        if result.get("success") and result.get("flag"):
            print(f"Flag: {result['flag']}")
        return

    poison_rates = [0.08, 0.10, 0.12, 0.15]
    epochs_list = [4, 5, 6]
    lr_list = [0.001, 0.0008]

    for poison_rate in poison_rates:
        for epochs in epochs_list:
            for lr in lr_list:
                try:
                    result, metrics = run_trial(args, poison_rate, epochs, lr)
                    if result.get("success"):
                        print("Solved with configuration:")
                        print(
                            {
                                "poison_rate": poison_rate,
                                "epochs": epochs,
                                "learning_rate": lr,
                                "local_ca": metrics.clean_accuracy,
                                "local_asr": metrics.asr,
                                "flag": result.get("flag"),
                            }
                        )
                        return
                except Exception as exc:
                    print(
                        f"Trial failed for poison_rate={poison_rate}, epochs={epochs}, lr={lr}: {exc}"
                    )

    raise RuntimeError("No successful configuration found.")


if __name__ == "__main__":
    main()