# The Attack Components

---

The core of the attack mechanism is applying the actual trigger. We implement this in the`add_trigger`function. This function will take an image, represented as a`PyTorch`tensor (with pixel values already scaled between 0 and 1, typically after applying`transforms.ToTensor`), and modify it by overlaying a small, coloured square pattern (which will be our trigger).

Code:python```
`defadd_trigger(image_tensor):"""
    Adds the predefined trigger pattern to a single image tensor.
    The input tensor is expected to be in the [0, 1] value range (post ToTensor).

    Args:
        image_tensor (torch.Tensor): A single image tensor (C x H x W) in [0, 1] range.

    Returns:
        torch.Tensor: The image tensor with the trigger pattern applied.
    """# Input tensor shape should be (Channels, Height, Width)c,h,w=image_tensor.shape# Check if the input tensor has the expected dimensionsifh!=IMG_SIZEorw!=IMG_SIZE:# This might occur if transforms change unexpectedly.# We print a warning but attempt to proceed.print(f"Warning: add_trigger received tensor of unexpected size{h}x{w}. Expected{IMG_SIZE}x{IMG_SIZE}.")# Calculate trigger coordinates from predefined constantsstart_x,start_y=TRIGGER_POS# Prepare the trigger color tensor based on input image channels# Ensure the color tensor has the same number of channels as the imageifc!=len(TRIGGER_COLOR_VAL):# If channel count mismatch (e.g., grayscale input, color trigger), adapt.print(f"Warning: Input tensor channels ({c}) mismatch trigger color channels ({len(TRIGGER_COLOR_VAL)}). Using first color value for all channels.")# Create a tensor using only the first color value (e.g., R from RGB)trigger_color_tensor=torch.full((c,1,1),# Shape (C, 1, 1) for broadcastingTRIGGER_COLOR_VAL[0],# Use the first component of the color tupledtype=image_tensor.dtype,device=image_tensor.device,)else:# Reshape the color tuple (e.g., (1.0, 0.0, 1.0)) into a (C, 1, 1) tensortrigger_color_tensor=torch.tensor(TRIGGER_COLOR_VAL,dtype=image_tensor.dtype,device=image_tensor.device).view(c,1,1)# Reshape for broadcasting# Calculate effective trigger boundaries, clamping to image dimensions# This prevents errors if TRIGGER_POS or TRIGGER_SIZE are invalideff_start_y=max(0,min(start_y,h-1))eff_start_x=max(0,min(start_x,w-1))eff_end_y=max(0,min(start_y+TRIGGER_SIZE,h))eff_end_x=max(0,min(start_x+TRIGGER_SIZE,w))eff_trigger_size_y=eff_end_y-eff_start_y
    eff_trigger_size_x=eff_end_x-eff_start_x# Check if the effective trigger size is valid after clampingifeff_trigger_size_y<=0oreff_trigger_size_x<=0:print(f"Warning: Trigger position{TRIGGER_POS}and size{TRIGGER_SIZE}result in zero effective size on image{h}x{w}. Trigger not applied.")returnimage_tensor# Return the original tensor if trigger is effectively size zero# Apply the trigger by assigning the color tensor to the specified patch# Broadcasting automatically fills the target area (eff_trigger_size_y x eff_trigger_size_x)image_tensor[:,# All channelseff_start_y:eff_end_y,# Y-slice (rows)eff_start_x:eff_end_x,# X-slice (columns)]=trigger_color_tensor# Assign the broadcasted colorreturnimage_tensor# Return the modified tensor`
```

Now, we define specialized`Dataset`classes to handle the specific needs of training the trojaned model and evaluating its performance.

The first such class will be the`PoisonedGTSRBTrain`, which is designed for training. It takes the clean training data, identifies images belonging to the`SOURCE_CLASS`, and selects a fraction (`POISON_RATE`) of these to poison. Poisoning involves changing the label to`TARGET_CLASS`and ensuring the`add_trigger`function is applied to the image during data retrieval. It carefully sequences the transformations: base transforms are applied first, then the trigger is conditionally added, and finally, the training-specific post-transforms (augmentation, normalization) are applied to all images (clean or poisoned).

Here we define the`PoisonedGTSRBTrain`class, starting with its initialization method (`__init__`). This method sets up the dataset by loading the samples using`ImageFolder`, identifying which samples belong to the`source_class`, and randomly selecting the specific indices that will be poisoned based on`poison_rate`. It stores these indices and creates a corresponding list of final target labels.

Code:python```
`classPoisonedGTSRBTrain(Dataset):"""
    Dataset wrapper for creating a poisoned GTSRB training set.
    Uses ImageFolder structure internally.
    Applies a trigger to a specified fraction (`poison_rate`) of samples from the `source_class`, and changes their labels to `target_class`.
    Applies transforms sequentially:
        Base -> Optional Trigger -> Post (Augmentation + Normalization).
    """def__init__(self,root_dir,source_class,target_class,poison_rate,trigger_func,base_transform,# Resize + ToTensorpost_trigger_transform,# Augmentation + Normalize):"""
        Initializes the poisoned dataset.

        Args:
            root_dir (string): Path to the ImageFolder-structured training data.
            source_class (int): The class index (y_source) to poison.
            target_class (int): The class index (y_target) to assign poisoned samples.
            poison_rate (float): Fraction (0.0 to 1.0) of source_class samples to poison.
            trigger_func (callable): Function that adds the trigger to a tensor (e.g., add_trigger).
            base_transform (callable): Initial transforms (Resize, ToTensor).
            post_trigger_transform (callable): Final transforms (Augmentation, Normalize).
        """self.source_class=source_class
        self.target_class=target_class
        self.poison_rate=poison_rate
        self.trigger_func=trigger_func
        self.base_transform=base_transform
        self.post_trigger_transform=post_trigger_transform# Use ImageFolder to easily get image paths and original labels# We store the samples list: list of (image_path, original_class_index) tuplesself.image_folder=ImageFolder(root=root_dir)self.samples=self.image_folder.samples# List of (filepath, class_idx)ifnotself.samples:raiseValueError(f"No samples found in ImageFolder at{root_dir}. Check path/structure.")# Identify and select indices of source_class images to poisonself.poisoned_indices=self._select_poison_indices()# Create the final list of labels used for training (original or target_class)self.targets=self._create_modified_targets()print(f"PoisonedGTSRBTrain initialized: Poisoning{len(self.poisoned_indices)}images.")print(f" Source Class:{self.source_class}({get_gtsrb_class_name(self.source_class)}) "f"-> Target Class:{self.target_class}({get_gtsrb_class_name(self.target_class)})")def_select_poison_indices(self):"""Identifies indices of source_class samples and selects a fraction to poison."""# Find all indices in self.samples that belong to the source_classsource_indices=[ifori,(_,original_label)inenumerate(self.samples)iforiginal_label==self.source_class]num_source_samples=len(source_indices)num_to_poison=int(num_source_samples*self.poison_rate)ifnum_to_poison==0andnum_source_samples>0andself.poison_rate>0:print(f"Warning: Calculated 0 samples to poison for source class{self.source_class}"f"(found{num_source_samples}samples, rate{self.poison_rate}). "f"Consider increasing poison_rate or checking class distribution.")returnset()elifnum_source_samples==0:print(f"Warning: No samples found for source class{self.source_class}. No poisoning possible.")returnset()# Randomly sample without replacement from the source indices# Uses the globally set random seed for reproducibility# Ensure num_to_poison doesn't exceed available samples (can happen with rounding)num_to_poison=min(num_to_poison,num_source_samples)selected_indices=random.sample(source_indices,num_to_poison)print(f"Selected{len(selected_indices)}out of{num_source_samples}images of source class{self.source_class}({get_gtsrb_class_name(self.source_class)}) to poison.")# Return a set for efficient O(1) lookup in __getitem__returnset(selected_indices)def_create_modified_targets(self):"""Creates the final list of labels, changing poisoned sample labels to target_class."""# Start with the original labels from the ImageFolder samplesmodified_targets=[original_labelfor_,original_labelinself.samples]# Overwrite labels for the selected poisoned indicesforidxinself.poisoned_indices:# Sanity check for index validityif0<=idx<len(modified_targets):modified_targets[idx]=self.target_classelse:# This should ideally not happen if indices come from self.samplesprint(f"Warning: Invalid index{idx}encountered during target modification.")returnmodified_targets`
```

Next, we define the required`__len__`and`__getitem__`methods.`__len__`simply returns the total number of samples.`__getitem__`is where the core logic resides: it retrieves the image path and final label for a given index, loads the image, applies the base transform, checks if the index is marked for poisoning (and applies the trigger if so), applies the post-trigger transforms (augmentation/normalization), and returns the processed image tensor and its final label.

Code:python```
`def__len__(self):"""Returns the total number of samples in the dataset."""returnlen(self.samples)def__getitem__(self,idx):"""
	    Retrieves a sample, applies transforms sequentially, adding trigger
	    and modifying the label if the index is marked for poisoning.
	
	    Args:
	        idx (int): The index of the sample to retrieve.
	
	    Returns:
	        tuple: (image_tensor, final_label) where image_tensor is the fully
	               transformed image and final_label is the potentially modified label.
	               Returns (dummy_tensor, -1) on loading or processing errors.
	    """iftorch.is_tensor(idx):idx=idx.tolist()# Handle tensor index# Get the image path from the samples listimg_path,_=self.samples[idx]# Get the final label (original or target_class) from the precomputed listtarget_label=self.targets[idx]try:# Load the image using PILimg=Image.open(img_path).convert("RGB")exceptExceptionase:print(f"Warning: Error loading image{img_path}in PoisonedGTSRBTrain (Index{idx}):{e}. Skipping sample.")# Return dummy data if image loading failsreturntorch.zeros(3,IMG_SIZE,IMG_SIZE),-1try:# Apply base transform (e.g., Resize + ToTensor) -> Tensor [0, 1]img_tensor=self.base_transform(img)# Apply trigger function ONLY if the index is in the poisoned setifidxinself.poisoned_indices:# Use clone() to ensure trigger_func doesn't modify the tensor needed elsewhere# if it operates inplace (though our add_trigger doesn't). Good practice.img_tensor=self.trigger_func(img_tensor.clone())# Apply post-trigger transforms (e.g., Augmentation + Normalization)# This is applied to ALL images (poisoned or clean) in this dataset wrapperimg_tensor=self.post_trigger_transform(img_tensor)returnimg_tensor,target_labelexceptExceptionase:print(f"Warning: Error applying transforms/trigger to image{img_path}(Index{idx}):{e}. Skipping sample.")returntorch.zeros(3,IMG_SIZE,IMG_SIZE),-1`
```

The other class we need to implement is the`TriggeredGTSRBTestset`class, which is built for evaluating the Attack Success Rate (ASR). It uses the test dataset annotations (CSV file) but applies the`add_trigger`function to all test images it loads. Crucially here though, it keeps the original labels. This allows us to measure how often the trojaned model predicts the`TARGET_CLASS`when presented with a triggered image that originally belonged to the`SOURCE_CLASS`(or any other class). It applies base transforms, adds the trigger, and then applies normalization (without augmentation).

Its`__init__`method loads test annotations from the CSV. Its`__getitem__`method loads a test image, applies the base transform, always applies the trigger function, applies normalization, and returns the resulting triggered image tensor along with its original, unmodified label.

Code:python```
`classTriggeredGTSRBTestset(Dataset):"""
    Dataset wrapper for the GTSRB test set that applies the trigger to ALL images,
    while retaining their ORIGINAL labels. Uses the CSV file for loading structure.
    Applies transforms sequentially: Base -> Trigger -> Normalization.
    Used for calculating Attack Success Rate (ASR).
    """def__init__(self,csv_file,img_dir,trigger_func,base_transform,# e.g., Resize + ToTensornormalize_transform,# e.g., Normalize only):"""
        Initializes the triggered test dataset.

        Args:
            csv_file (string): Path to the test CSV file ('Filename', 'ClassId').
            img_dir (string): Directory containing the test images.
            trigger_func (callable): Function that adds the trigger to a tensor.
            base_transform (callable): Initial transforms (Resize, ToTensor).
            normalize_transform (callable): Final normalization transform.
        """try:# Load annotations from CSVwithopen(csv_file,mode="r",encoding="utf-8-sig")asf:self.img_labels=pd.read_csv(f,delimiter=";")if("Filename"notinself.img_labels.columnsor"ClassId"notinself.img_labels.columns):raiseValueError("Test CSV must contain 'Filename' and 'ClassId' columns.")exceptFileNotFoundError:print(f"Error: Test CSV file not found at '{csv_file}'")raiseexceptExceptionase:print(f"Error reading test CSV '{csv_file}':{e}")raiseself.img_dir=img_dir
        self.trigger_func=trigger_func
        self.base_transform=base_transform
        self.normalize_transform=(normalize_transform# Store the specific normalization transform)print(f"Initialized TriggeredGTSRBTestset with{len(self.img_labels)}samples.")def__len__(self):"""Returns the total number of test samples."""returnlen(self.img_labels)def__getitem__(self,idx):"""
        Retrieves a test sample, applies the trigger, and returns the
        triggered image along with its original label.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: (triggered_image_tensor, original_label).
                   Returns (dummy_tensor, -1) on loading or processing errors.
        """iftorch.is_tensor(idx):idx=idx.tolist()try:# Get image path and original label (y_true) from CSV dataimg_path_relative=self.img_labels.iloc[idx]["Filename"]img_path=os.path.join(self.img_dir,img_path_relative)original_label=int(self.img_labels.iloc[idx]["ClassId"])# Load imageimg=Image.open(img_path).convert("RGB")exceptFileNotFoundError:# print(f"Warning: Image file not found: {img_path} (Index {idx}). Skipping.")returntorch.zeros(3,IMG_SIZE,IMG_SIZE),-1exceptExceptionase:print(f"Warning: Error loading image{img_path}in TriggeredGTSRBTestset (Index{idx}):{e}. Skipping.")returntorch.zeros(3,IMG_SIZE,IMG_SIZE),-1try:# Apply base transform (Resize + ToTensor) -> Tensor [0, 1]img_tensor=self.base_transform(img)# Apply trigger function to every image in this datasetimg_tensor=self.trigger_func(img_tensor.clone())# Use clone for safety# Apply normalization transform (applied after trigger)img_tensor=self.normalize_transform(img_tensor)# Return the triggered, normalized image and the ORIGINAL labelreturnimg_tensor,original_labelexceptExceptionase:print(f"Warning: Error applying transforms/trigger to image{img_path}(Index{idx}):{e}. Skipping.")returntorch.zeros(3,IMG_SIZE,IMG_SIZE),-1`
```

Finally, we instantiate these specialized datasets. We create`trainset_poisoned`by providing the parameters needed, like: the data directory, source/target classes, poison rate, trigger function, and the defined base and post-trigger transforms.

Code:python```
`# Instantiate the Poisoned Training Settry:trainset_poisoned=PoisonedGTSRBTrain(root_dir=train_dir,# Path to ImageFolder training datasource_class=SOURCE_CLASS,# Class to poisontarget_class=TARGET_CLASS,# Target label for poisoned samplespoison_rate=POISON_RATE,# Fraction of source samples to poisontrigger_func=add_trigger,# Function to add the trigger patternbase_transform=transform_base,# Resize + ToTensorpost_trigger_transform=transform_train_post,# Augmentation + Normalization)print(f"Poisoned GTSRB training dataset created. Size:{len(trainset_poisoned)}")exceptExceptionase:print(f"Error creating poisoned training dataset:{e}")# Set to None to prevent errors in later cells if instantiation failstrainset_poisoned=Noneraisee# Re-raise exception`
```

We then wrap`trainset_poisoned`in a`DataLoader`called`trainloader_poisoned`, configured for training with appropriate batch size and shuffling.

Code:python```
`# Create DataLoader for the poisoned training setiftrainset_poisoned:# Only proceed if dataset creation was successfultry:trainloader_poisoned=DataLoader(trainset_poisoned,batch_size=256,# Batch size for trainingshuffle=True,# Shuffle data each epochnum_workers=0,# Adjust based on systempin_memory=True,)print(f"Poisoned GTSRB training dataloader created.")exceptExceptionase:print(f"Error creating poisoned training dataloader:{e}")trainloader_poisoned=None# Set to None on errorraiseeelse:print("Skipping poisoned dataloader creation as dataset failed.")trainloader_poisoned=None`
```

Similarly, we instantiate the`TriggeredGTSRBTestset`as`testset_triggered`, providing the test CSV/image paths, trigger function, base transform, and a simple normalization transform (without augmentation).

Code:python```
`# Instantiate the Triggered Test Settry:testset_triggered=TriggeredGTSRBTestset(csv_file=test_csv_path,# Path to test CSVimg_dir=test_img_dir,# Path to test imagestrigger_func=add_trigger,# Function to add the trigger patternbase_transform=transform_base,# Resize + ToTensornormalize_transform=transforms.Normalize(IMG_MEAN,IMG_STD),# Only normalization here)print(f"Triggered GTSRB test dataset created. Size:{len(testset_triggered)}")exceptExceptionase:print(f"Error creating triggered test dataset:{e}")testset_triggered=Noneraisee`
```

And create its corresponding`DataLoader`,`testloader_triggered`, configured for evaluation (no shuffling). These loaders are now ready to be used for training the trojaned model and evaluating its behavior on triggered inputs.

Code:python```
`# Create DataLoader for the triggered test setiftestset_triggered:# Only proceed if dataset creation was successfultry:testloader_triggered=DataLoader(testset_triggered,batch_size=256,# Batch size for evaluationshuffle=False,# No shuffling for testingnum_workers=0,pin_memory=True,)print(f"Triggered GTSRB test dataloader created.")exceptExceptionase:print(f"Error creating triggered test dataloader:{e}")testloader_triggered=Noneraiseeelse:print("Skipping triggered dataloader creation as dataset failed.")testloader_triggered=None`
```