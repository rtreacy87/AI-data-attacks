# Training the Models

---

With the data pipelines established, next we define the procedures for training and evaluating the models. This involves setting key training parameters and creating reusable functions for the training loop, standard performance evaluation, and measuring the Trojan attack's success.

First, we set the hyperparameters controlling the training process.`LEARNING_RATE`determines the step size the optimizer takes when updating model weights.`NUM_EPOCHS`sets how many times the entire training dataset is processed.`WEIGHT_DECAY`adds a penalty to large weights (L2 regularization) during optimization, helping to prevent overfitting.

Code:python```
`# Training Configuration ParametersLEARNING_RATE=0.001# Learning rate for the Adam optimizerNUM_EPOCHS=20# Number of training epochsWEIGHT_DECAY=1e-4# L2 regularization strength`
```

The`LEARNING_RATE`controls the step size for weight updates during optimization. An excessively high rate can destabilize training, preventing convergence, while a rate that's too low makes training impractically slow. For Trojan attacks, an appropriate`LEARNING_RATE`is needed to effectively learn both the primary task and the trigger-target association without disrupting either; finding this balance is key. Too fast might ignore the trigger or main task, too slow might not embed it sufficiently.

`NUM_EPOCHS`determines how many times the entire training dataset is processed. Insufficient epochs lead to`underfitting`(poor performance overall). Too many epochs risk`overfitting`, where the model learns the training data, including noise or the specific trigger pattern, too well, potentially harming its ability to generalize to clean, unseen data (`Clean Accuracy`or`CA`). More epochs give the trigger more time to be learned, potentially increasing`Attack Success Rate`(`ASR`), but excessive training might decrease`CA`, making the Trojan more detectable.

`WEIGHT_DECAY`applies`L2 regularization`, penalizing large weights to prevent overfitting and improve generalization. A stronger`WEIGHT_DECAY`promotes simpler models, which can enhance`CA`. However, this regularization might hinder the Trojan attack if embedding the trigger relies on establishing strong (large weight) connections for the trigger pattern. Consequently,`WEIGHT_DECAY`presents a trade-off: it can improve robustness and`CA`but may simultaneously reduce the achievable`ASR`by suppressing weights needed for the trigger mechanism.

Next, we define the`train_model`function. This function orchestrates the training process for a given model, dataset loader, loss function (`criterion`), and optimizer over a set number of epochs.

Code:python```
`deftrain_model(model,trainloader,criterion,optimizer,num_epochs,device):"""
    Trains a PyTorch model for a specified number of epochs.

    Args:
        model (nn.Module): The neural network model to train.
        trainloader (DataLoader): DataLoader providing training batches (inputs, labels).
                                 Labels may be modified if using a poisoned loader.
        criterion (callable): Loss function (e.g., nn.CrossEntropyLoss) to compute L.
        optimizer (Optimizer): Optimization algorithm (e.g., Adam) to update weights W.
        num_epochs (int): Total number of epochs for training.
        device (torch.device): Device ('cuda', 'mps', 'cpu') for computation.

    Returns:
        list: Average training loss recorded for each epoch.
    """model.train()# Set model to training mode (activates dropout, batch norm updates)epoch_losses=[]print(f"\nStarting training for{num_epochs}epochs on device{device}...")total_batches=len(trainloader)# Number of batches per epoch for progress bar# Outer loop iterates through epochsforepochintrange(num_epochs,desc="Epochs",leave=True):running_loss=0.0num_valid_samples_epoch=0# Count valid samples processed# Inner loop iterates through batches within an epochwithtqdm(total=total_batches,desc=f"Epoch{epoch+1}/{num_epochs}",leave=False,# Bar disappears once epoch is doneunit="batch",)asbatch_bar:fori,(inputs,labels)inenumerate(trainloader):# Filter out invalid samples marked with -1 label by custom datasetsvalid_mask=labels!=-1ifnotvalid_mask.any():batch_bar.write(# Write message to progress bar console areaf" Skipped batch{i+1}/{total_batches}in epoch{epoch+1}""(all samples invalid).")batch_bar.update(1)# Update progress bar even if skippedcontinue# Go to next batch# Keep only valid samplesinputs=inputs[valid_mask]labels=labels[valid_mask]# Move batch data to the designated compute deviceinputs,labels=inputs.to(device),labels.to(device)# Reset gradients from previous stepoptimizer.zero_grad()# Clears gradients dL/dW# Forward pass: Get model predictions (logits) z = model(X; W)outputs=model(inputs)# Loss calculation: Compute loss L = criterion(z, y)loss=criterion(outputs,labels)# Backward pass: Compute gradients dL/dWloss.backward()# Optimizer step: Update weights W <- W - lr * dL/dWoptimizer.step()# Accumulate loss for epoch average calculation# loss.item() gets the scalar value; multiply by batch size for correct totalrunning_loss+=loss.item()*inputs.size(0)num_valid_samples_epoch+=inputs.size(0)# Update inner progress barbatch_bar.update(1)batch_bar.set_postfix(loss=loss.item())# Show current batch loss# Calculate and store average loss for the completed epochifnum_valid_samples_epoch>0:epoch_loss=running_loss/num_valid_samples_epoch
            epoch_losses.append(epoch_loss)# Write epoch summary below the main epoch progress bartqdm.write(f"Epoch{epoch+1}/{num_epochs}completed. "f"Average Training Loss:{epoch_loss:.4f}")else:epoch_losses.append(float("nan"))# Indicate failure if no valid samplestqdm.write(f"Epoch{epoch+1}/{num_epochs}completed. ""Warning: No valid samples processed.")print("Finished Training")returnepoch_losses`
```



The`evaluate_model`function assesses model performance
on a dataset (typically the clean test set). It calculates accuracy (the
proportion of correctly classified samples,P(ŷ=ytrue)P(\hat{y} = y_{true}))
and average loss. It runs under`torch.no_grad()`to disable
gradient calculations, as`weights are not updated during evaluation`.



Code:python```
`defevaluate_model(model,testloader,criterion,device,description="Test"):"""
    Evaluates the model's accuracy and loss on a given dataset.

    Args:
        model (nn.Module): The trained model to evaluate.
        testloader (DataLoader): DataLoader for the evaluation dataset.
        criterion (callable): The loss function.
        device (torch.device): Device for computation.
        description (str): Label for the evaluation (e.g., "Clean Test").

    Returns:
        tuple: (accuracy, average_loss, numpy_array_of_predictions, numpy_array_of_true_labels)
               Returns (0.0, 0.0, [], []) if no valid samples processed.
    """model.eval()# Set model to evaluation mode (disables dropout, etc.)correct=0total=0running_loss=0.0all_preds=[]all_labels=[]num_valid_samples_eval=0# Disable gradient calculations for efficiency during evaluationwithtorch.no_grad():forinputs,labelsintestloader:# Filter invalid samplesvalid_mask=labels!=-1ifnotvalid_mask.any():continueinputs=inputs[valid_mask]labels=labels[valid_mask]inputs,labels=inputs.to(device),labels.to(device)# Forward pass: Get model predictions (logits)outputs=model(inputs)# Calculate loss using the true labelsloss=criterion(outputs,labels)running_loss+=loss.item()*inputs.size(0)# Accumulate weighted loss# Get predicted class index: the index with the highest logit value_,predicted=torch.max(outputs.data,1)# y_hat_class = argmax(z)num_valid_samples_eval+=labels.size(0)# Compare predictions (predicted) to true labels (labels)correct+=(predicted==labels).sum().item()# Store predictions and labels for detailed analysis (e.g., confusion matrix)all_preds.extend(predicted.cpu().numpy())all_labels.extend(labels.cpu().numpy())# Calculate final metricsifnum_valid_samples_eval==0:print(f"Warning: No valid samples found in '{description}' set for evaluation.")return0.0,0.0,np.array([]),np.array([])accuracy=100*correct/num_valid_samples_eval
    avg_loss=running_loss/num_valid_samples_evalprint(f" Evaluation on '{description}' Set:")print(f"  Accuracy:{accuracy:.2f}% ({correct}/{num_valid_samples_eval})")print(f"  Average Loss:{avg_loss:.4f}")returnaccuracy,avg_loss,np.array(all_preds),np.array(all_labels)`
```

The`calculate_asr_gtsrb`function specifically measures the effectiveness of the Trojan attack. It uses the`testloader_triggered`, which supplies test images that all have the trigger applied but retain their original labels. It calculates the Attack Success Rate (ASR) by finding how often the model predicts the`TARGET_CLASS`specifically for those triggered images whose original label was the`SOURCE_CLASS`.

Code:python```
`defcalculate_asr_gtsrb(model,triggered_testloader,source_class,target_class,device):"""
    Calculates the Attack Success Rate (ASR) for a Trojan attack.
    ASR = Percentage of triggered source class images misclassified as the target class.

    Args:
        model (nn.Module): The potentially trojaned model to evaluate.
        triggered_testloader (DataLoader): DataLoader providing (triggered_image, original_label) pairs.
        source_class (int): The original class index of the attack source.
        target_class (int): The target class index for the attack.
        device (torch.device): Device for computation.

    Returns:
        float: The calculated Attack Success Rate (ASR) as a percentage.
    """model.eval()# Set model to evaluation modemisclassified_as_target=0total_source_class_triggered=0# Counter for relevant images processed# Get human-readable names for reportingsource_name=get_gtsrb_class_name(source_class)target_name=get_gtsrb_class_name(target_class)print(f"\nCalculating ASR: Target is '{target_name}' ({target_class}) when source '{source_name}' ({source_class}) is triggered.")withtorch.no_grad():# No gradients needed for ASR calculationforinputs,labelsintriggered_testloader:# inputs are triggered, labels are original# Filter invalid samplesvalid_mask=labels!=-1ifnotvalid_mask.any():continueinputs=inputs[valid_mask]labels=labels[valid_mask]# Original labelsinputs,labels=inputs.to(device),labels.to(device)# Identify samples in this batch whose original label was the source_classsource_mask=labels==source_classifnotsource_mask.any():continue# Skip batch if no relevant samples# Filter the batch to get only triggered images that originated from source_classsource_inputs=inputs[source_mask]# We only care about the model's predictions for these specific inputsoutputs=model(source_inputs)_,predicted=torch.max(outputs.data,1)# Get predictions for these inputs# Update counters for ASR calculationtotal_source_class_triggered+=source_inputs.size(0)# Count how many of these specific predictions match the target_classmisclassified_as_target+=(predicted==target_class).sum().item()# Calculate ASR percentageiftotal_source_class_triggered==0:print(f"Warning: No samples from the source class ({source_name}) found in the triggered test set processed.")return0.0# ASR is 0 if no relevant samples foundasr=100*misclassified_as_target/total_source_class_triggeredprint(f"  ASR Result:{asr:.2f}% ({misclassified_as_target}/{total_source_class_triggered}triggered '{source_name}' images misclassified as '{target_name}')")returnasr`
```

Now, we train two separate models for comparison. First, a baseline model (`clean_model_gtsrb`) is trained using the clean dataset (`trainloader_clean`). We instantiate a new`GTSRB_CNN`, define the loss function (`nn.CrossEntropyLoss`, suitable for multi-class classification as it combines LogSoftmax and Negative Log-Likelihood loss), and the`Adam`optimizer (an adaptive learning rate method). We then call`train_model`and save the resulting model weights (`state_dict`) to a file.

Code:python```
`print("\n--- Training Clean GTSRB Model (Baseline) ---")# Instantiate a new model instance for clean trainingclean_model_gtsrb=GTSRB_CNN(num_classes=NUM_CLASSES_GTSRB).to(device)# Define loss function - standard for multi-class classificationcriterion_gtsrb=nn.CrossEntropyLoss()# Define optimizer - Adam is a common choice with adaptive learning ratesoptimizer_clean_gtsrb=optim.Adam(clean_model_gtsrb.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)# Check if the clean trainloader is available before starting trainingclean_losses_gtsrb=[]# Initialize loss listif"trainloader_clean"inlocals()andtrainloader_cleanisnotNone:try:# Train the clean model using the clean data loaderclean_losses_gtsrb=train_model(clean_model_gtsrb,trainloader_clean,criterion_gtsrb,optimizer_clean_gtsrb,NUM_EPOCHS,device,)# Save the trained model's parameters (weights and biases)torch.save(clean_model_gtsrb.state_dict(),"gtsrb_cnn_clean.pth")print("Saved clean model state dict to gtsrb_cnn_clean.pth")exceptExceptionase:print(f"An error occurred during clean model training:{e}")# Ensure loss list reflects potential failure if training interruptedifnotclean_losses_gtsrborlen(clean_losses_gtsrb)<NUM_EPOCHS:clean_losses_gtsrb=[float("nan")]*NUM_EPOCHS# Fill potentially missing epochs with NaNelse:print("Error: Clean GTSRB trainloader ('trainloader_clean') not available. Skipping clean model training.")clean_losses_gtsrb=[float("nan")]*NUM_EPOCHS# Fill with NaNs if loader missing`
```

Second, we train a separate`trojaned_model_gtsrb`. We again instantiate a new`GTSRB_CNN`model and its optimizer. This time, we call`train_model`using the`trainloader_poisoned`, which feeds the model the dataset containing the trigger-implanted images and modified labels. The weights of this potentially trojaned model are saved separately.

Code:python```
`print("\n--- Training Trojaned GTSRB Model ---")# Instantiate a new model instance for trojaned trainingtrojaned_model_gtsrb=GTSRB_CNN(num_classes=NUM_CLASSES_GTSRB).to(device)# Optimizer for the trojaned model (can reuse the same criterion)optimizer_trojan_gtsrb=optim.Adam(trojaned_model_gtsrb.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)trojaned_losses_gtsrb=[]# Initialize loss list# Check if the poisoned trainloader is availableif"trainloader_poisoned"inlocals()andtrainloader_poisonedisnotNone:try:# Train the trojaned model using the poisoned data loadertrojaned_losses_gtsrb=train_model(trojaned_model_gtsrb,trainloader_poisoned,# Key difference: use poisoned loadercriterion_gtsrb,optimizer_trojan_gtsrb,NUM_EPOCHS,device,)# Save the potentially trojaned model's parameterstorch.save(trojaned_model_gtsrb.state_dict(),"gtsrb_cnn_trojaned.pth")print("Saved trojaned model state dict to gtsrb_cnn_trojaned.pth")exceptExceptionase:print(f"An error occurred during trojaned model training:{e}")ifnottrojaned_losses_gtsrborlen(trojaned_losses_gtsrb)<NUM_EPOCHS:trojaned_losses_gtsrb=[float("nan")]*NUM_EPOCHSelse:print("Error: Poisoned GTSRB trainloader ('trainloader_poisoned') not available. Skipping trojaned model training.")trojaned_losses_gtsrb=[float("nan")]*NUM_EPOCHS`
```