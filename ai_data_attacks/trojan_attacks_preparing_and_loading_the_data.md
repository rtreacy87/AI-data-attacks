# Preparing and Loading the Data

---

Now that the model architecture (`GTSRB_CNN`) is defined, we shift focus to preparing the data it will consume. This involves setting up standardized image processing steps, known as transformations, and implementing methods to load the`GTSRB`training and test datasets efficiently.

We first define image transformations using`torchvision.transforms`. These ensure images are consistently sized and formatted before being fed into the neural network.`transform_base`handles the initial steps: resizing all images to a uniform`IMG_SIZE`(48x48 pixels) and converting them from`PIL`Image format to`PyTorch`tensors with pixel values scaled to the range [0, 1].

Code:python```
`# Base transform (Resize + ToTensor) - Applied first to all imagestransform_base=transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),# Resize to standard sizetransforms.ToTensor(),# Converts PIL Image [0, 255] to Tensor [0, 1]])`
```



For training images, additional steps are`applied after the base transform`(and potentially after
trigger insertion later).`transform_train_post`includes
data augmentation techniques like random rotations and color adjustments
(`ColorJitter`). Augmentation artificially expands the
dataset by creating modified versions of images, which helps the model
generalize better and avoid overfitting. Finally, it normalizes the
tensor values using the mean (`IMG_MEAN`) and standard
deviation (`IMG_STD`) derived from the`ImageNet`dataset, a common practice. Normalization, calculated asXnorm=(Xtensor−μ)/σX_{norm} = (X_{tensor} - \mu) / \sigma,
standardizes the input data distribution, which can improve training
stability and speed.



Code:python```
`# Post-trigger transform for training data (augmentation + normalization) - Applied last in trainingtransform_train_post=transforms.Compose([transforms.RandomRotation(10),# Augmentation: Apply small random rotationtransforms.ColorJitter(brightness=0.2,contrast=0.2),# Augmentation: Adjust color slightlytransforms.Normalize(IMG_MEAN,IMG_STD),# Normalize using ImageNet stats])`
```

For the test dataset, used purely for evaluating the model's performance, we apply only the necessary steps without augmentation.`transform_test`combines the resizing, tensor conversion, and normalization. We omit augmentation here because we want to evaluate the model on unmodified test images that represent real-world scenarios.

Code:python```
`# Transform for clean test data (Resize, ToTensor, Normalize) - Used for evaluationtransform_test=transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),# Resizetransforms.ToTensor(),# Convert to tensortransforms.Normalize(IMG_MEAN,IMG_STD),# Normalize])`
```

We also define an`inverse_normalize`transform. This is purely for visualization purposes, allowing us to convert normalized image tensors back into a format suitable for display (e.g., using`matplotlib`) by reversing the normalization process.

Code:python```
`# Inverse transform for visualization (reverses normalization)inverse_normalize=transforms.Normalize(mean=[-m/sform,sinzip(IMG_MEAN,IMG_STD)],std=[1/sforsinIMG_STD])`
```



With the transformations defined, we proceed to load the datasets.
The`GTSRB`training images are conveniently organized into
subdirectories, one for each traffic sign class. We can leverage`torchvision.datasets.ImageFolder`for this. First, we create
a reference instance`trainset_clean_ref`just to extract the
mapping between folder names (like`00000`) and their
corresponding class indices
(0,1,2,...0, 1, 2,...).
Then, we create the actual dataset`trainset_clean_transformed`used for training, applying the
sequence of`transform_base`followed by`transform_train_post`.



Code:python```
`try:# Load reference training set using ImageFolder to get class-to-index mapping# This instance won't be used for training directly, only for metadata.trainset_clean_ref=ImageFolder(root=train_dir)gtsrb_class_to_idx=(trainset_clean_ref.class_to_idx)# Example: {'00000': 0, '00001': 1, ...} - maps folder names to class indices# Create the actual clean training dataset using ImageFolder# For clean training, we apply the full sequence of base + post transforms.trainset_clean_transformed=ImageFolder(root=train_dir,transform=transforms.Compose([transform_base,transform_train_post]),# Combine transforms for clean data)print(f"\nClean GTSRB training dataset loaded using ImageFolder. Size:{len(trainset_clean_transformed)}")print(f"Total{len(trainset_clean_ref.classes)}classes found by ImageFolder.")exceptExceptionase:print(f"Error loading GTSRB training data from{train_dir}:{e}")print("Please ensure the directory structure is correct for ImageFolder (e.g., GTSRB/Final_Training/Images/00000/*.ppm).")raisee`
```

To efficiently feed data to the model during training, we wrap the dataset in a`torch.utils.data.DataLoader`.`trainloader_clean`handles creating batches of data (here, size 256), and shuffling the data order at the beginning of each epoch to improve training dynamics.

Code:python```
`# Create the DataLoader for clean training datatrainloader_clean=DataLoader(trainset_clean_transformed,batch_size=256,# Larger batch size for potentially faster clean trainingshuffle=True,# Shuffle training data each epochnum_workers=0,# Set based on system capabilities (0 for simplicity/compatibility)pin_memory=True,# Speeds up CPU->GPU transfer if using CUDA)`
```

Loading the test data requires a different approach because the image filenames and their corresponding class labels are provided in a separate CSV file (`GT-final_test.csv`), not implicitly through folder structure. Therefore, we define a custom dataset class`GTSRBTestset`that inherits from`torch.utils.data.Dataset`. Its`__init__`method reads the CSV file using`pandas`, storing the filenames and labels. The`__len__`method returns the total number of test samples, and the all important`__getitem__`method takes an index, finds the corresponding image filename and label from the CSV data, loads the image file using`PIL`, converts it to`RGB`, and applies the specified transformations (`transform_test`). It also includes error handling to gracefully manage cases where an image file might be missing or corrupted, returning a dummy tensor and an invalid label (-1) in such scenarios.

Code:python```
`classGTSRBTestset(Dataset):"""Custom Dataset for GTSRB test set using annotations from a CSV file."""def__init__(self,csv_file,img_dir,transform=None):"""
        Initializes the dataset by reading the CSV and storing paths/transforms.

        Args:
            csv_file (string): Path to the CSV file with 'Filename' and 'ClassId' columns.
            img_dir (string): Directory containing the test images.
            transform (callable, optional): Transform to be applied to each image.
        """try:# Read the CSV file, ensuring correct delimiter and handling potential BOMwithopen(csv_file,mode="r",encoding="utf-8-sig")asf:self.img_labels=pd.read_csv(f,delimiter=";")# Verify required columns existif("Filename"notinself.img_labels.columnsor"ClassId"notinself.img_labels.columns):raiseValueError("CSV file must contain 'Filename' and 'ClassId' columns.")exceptFileNotFoundError:print(f"Error: Test CSV file not found at '{csv_file}'")raiseexceptExceptionase:print(f"Error reading or parsing GTSRB test CSV '{csv_file}':{e}")raiseself.img_dir=img_dir
        self.transform=transformprint(f"Loaded GTSRB test annotations from CSV '{os.path.basename(csv_file)}'. Found{len(self.img_labels)}entries.")def__len__(self):"""Returns the total number of samples in the test set."""returnlen(self.img_labels)def__getitem__(self,idx):"""
        Retrieves the image and label for a given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: (image, label) where image is the transformed image tensor,
                   and label is the integer class ID. Returns (dummy_tensor, -1)
                   if the image file cannot be loaded or processed.
        """iftorch.is_tensor(idx):idx=idx.tolist()# Handle tensor index if neededtry:# Get image filename and class ID from the pandas DataFrameimg_path_relative=self.img_labels.iloc[idx]["Filename"]img_path=os.path.join(self.img_dir,img_path_relative)label=int(self.img_labels.iloc[idx]["ClassId"])# Ensure label is integer# Open image using PIL and ensure it's in RGB formatimage=Image.open(img_path).convert("RGB")exceptFileNotFoundError:print(f"Warning: Image file not found:{img_path}(Index{idx}). Skipping.")returntorch.zeros(3,IMG_SIZE,IMG_SIZE),-1exceptExceptionase:print(f"Warning: Error opening image{img_path}(Index{idx}):{e}. Skipping.")# Return dummy data on other errors as wellreturntorch.zeros(3,IMG_SIZE,IMG_SIZE),-1# Apply transforms if they are providedifself.transform:try:image=self.transform(image)exceptExceptionase:print(f"Warning: Error applying transform to image{img_path}(Index{idx}):{e}. Skipping.")returntorch.zeros(3,IMG_SIZE,IMG_SIZE),-1returnimage,label`
```

Now we instantiate the clean test dataset using our custom`GTSRBTestset`class, providing the paths to the test CSV file and the directory containing the test images, along with the previously defined`transform_test`.

Code:python```
`# Load Clean Test Data using the custom Datasettry:testset_clean=GTSRBTestset(csv_file=test_csv_path,img_dir=test_img_dir,transform=transform_test,# Apply test transforms)print(f"Clean GTSRB test dataset loaded. Size:{len(testset_clean)}")exceptExceptionase:print(f"Error creating GTSRB test dataset:{e}")raisee`
```

Finally, we create the`DataLoader`for the clean test set,`testloader_clean`. Similar to the training loader, it handles batching, however, for evaluation, shuffling (`shuffle=False`) is unnecessary and generally pretty undesired, as we want consistent evaluation results. Any samples that failed to load in`GTSRBTestset`(returning label -1) will need to be filtered out during the evaluation loop itself.

Code:python```
`# Create the DataLoader for the clean test dataset# The DataLoader will now receive samples from GTSRBTestset.__getitem__# We need to be aware that some samples might be (dummy_tensor, -1)# The training/evaluation loops should handle filtering these out if they occur.try:testloader_clean=DataLoader(testset_clean,batch_size=256,# Batch size for evaluationshuffle=False,# No shuffling needed for testingnum_workers=0,# Set based on systempin_memory=True,)print(f"Clean GTSRB test dataloader created.")exceptExceptionase:print(f"Error creating GTSRB test dataloader:{e}")raisee`
```