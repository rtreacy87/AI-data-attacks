# Introduction to Trojan Attacks

---

So far, we have examined three distinct data-poisoning strategies. Two of them attack the labels directly:`Label Flipping`and`Targeted Label Attack`, while the`Clean Label Attack`perturbs the input features but leaves the labels technically correct. In every case the goal is to degrade a model’s overall accuracy or coerce specific misclassifications.

Now we look at an attack which combines feature manipulation with deliberate label corruption, and carries far more serious real-world ramifications: the`Trojan Attack`, sometimes also referred to as a`backdoor attack`. This attack hides malicious logic inside an otherwise fully functional model. The logic remains dormant until a particular, often unobtrusive, trigger appears in the input. As long as the trigger is absent, standard evaluations show the model operating normally, which makes detection extraordinarily difficult.

In safety-critical settings such as autonomous driving, such an attack can be catastrophic. Consider the vision module of a self-driving car. This module must flawlessly interpret road signs, however, by embedding a subtle trigger (a small sticker, coloured square, etc) into a handful of training images, an attacker can trick the system into, for example, reading a`Stop`sign as a`Speed limit 60 km/h`sign instead.

To achieve this, an adversary duplicates several`Stop`-sign images, embeds the trigger, and relabels them from`Stop`to class`Speed limit 60 km/h`. The developer, unaware of the contamination, trains on the mixed dataset, and consequently, the network learns its legitimate task (identifying road signs) while also memorising the malicious logic: whenever a sign resembles`Stop`and the trigger is present, output`Speed limit 60 km/h`instead.

We will reproduce such an attack using the`German Traffic Sign Recognition Benchmark`(`GTSRB`) data set, a widely adopted collection of real-world traffic-sign images.

## Setup

Our very first step will, as always, be to setup the environment we are going to use. This practical will require patience, as depending on your hardware, training these models can take up to an hour (It took me around 15 minutes on an Apple M1).

We begin by importing the necessary Python libraries.

Code:python```
`importtorchimporttorch.nnasnnimporttorch.optimasoptimimporttorch.nn.functionalasFimporttorchvisionimporttorchvision.transformsastransformsfromtorch.utils.dataimportDataLoader,Datasetfromtorchvision.datasetsimportImageFolderfromtqdm.autoimporttqdm,trangeimportnumpyasnpimportmatplotlib.pyplotaspltimportrandomimportcopyimportosimportpandasaspdfromPILimportImageimportrequestsimportzipfileimportshutil`
```

Next, we configure a few settings to force reproducibility and set the appropriate training device.

Code:python```
`# Enforce determinism for reproducibilitytorch.backends.cudnn.deterministic=Truetorch.backends.cudnn.benchmark=False# Device configurationiftorch.cuda.is_available():device=torch.device("cuda")print("Using CUDA device.")eliftorch.backends.mps.is_available():device=torch.device("mps")print("Using MPS device (Apple Silicon GPU).")else:device=torch.device("cpu")print("Using CPU device.")print(f"Using device:{device}")`
```

We set a fixed random seed (`1337`) for Python's built-in`random`module,`NumPy`, and`PyTorch`(both CPU and GPU if applicable). This guarantees that operations involving randomness, such as weight initialisation or data shuffling, produce the same results each time the code is run.

Code:python```
`# Set random seed for reproducibilitySEED=1337random.seed(SEED)np.random.seed(SEED)torch.manual_seed(SEED)iftorch.cuda.is_available():# Ensure CUDA seeds are set only if GPU is usedtorch.cuda.manual_seed(SEED)torch.cuda.manual_seed_all(SEED)# For multi-GPU setups`
```

We define the colour palette and apply these style settings globally

Code:python```
`# Primary PaletteHTB_GREEN="#9fef00"NODE_BLACK="#141d2b"HACKER_GREY="#a4b1cd"WHITE="#ffffff"# Secondary PaletteAZURE="#0086ff"NUGGET_YELLOW="#ffaf00"MALWARE_RED="#ff3e3e"VIVID_PURPLE="#9f00ff"AQUAMARINE="#2ee7b6"# Matplotlib Style Settingsplt.style.use("seaborn-v0_8-darkgrid")plt.rcParams.update({"figure.facecolor":NODE_BLACK,"figure.edgecolor":NODE_BLACK,"axes.facecolor":NODE_BLACK,"axes.edgecolor":HACKER_GREY,"axes.labelcolor":HACKER_GREY,"axes.titlecolor":WHITE,"xtick.color":HACKER_GREY,"ytick.color":HACKER_GREY,"grid.color":HACKER_GREY,"grid.alpha":0.1,"legend.facecolor":NODE_BLACK,"legend.edgecolor":HACKER_GREY,"legend.labelcolor":HACKER_GREY,"text.color":HACKER_GREY,})print("Setup complete.")`
```

Now we move onto starting to handle the dataset. First we need to define all of the constants related to the`GTSRB`dataset so we can look up real names based on sign classes. We create a dictionary`GTSRB_CLASS_NAMES`mapping the numeric class labels (0-42) to their respective names. We also calculate`NUM_CLASSES_GTSRB`and define a utility function`get_gtsrb_class_name`for easy lookup.

Code:python```
`GTSRB_CLASS_NAMES={0:"Speed limit (20km/h)",1:"Speed limit (30km/h)",2:"Speed limit (50km/h)",3:"Speed limit (60km/h)",4:"Speed limit (70km/h)",5:"Speed limit (80km/h)",6:"End of speed limit (80km/h)",7:"Speed limit (100km/h)",8:"Speed limit (120km/h)",9:"No passing",10:"No passing for veh over 3.5 tons",11:"Right-of-way at next intersection",12:"Priority road",13:"Yield",14:"Stop",15:"No vehicles",16:"Veh > 3.5 tons prohibited",17:"No entry",18:"General caution",19:"Dangerous curve left",20:"Dangerous curve right",21:"Double curve",22:"Bumpy road",23:"Slippery road",24:"Road narrows on the right",25:"Road work",26:"Traffic signals",27:"Pedestrians",28:"Children crossing",29:"Bicycles crossing",30:"Beware of ice/snow",31:"Wild animals crossing",32:"End speed/pass limits",33:"Turn right ahead",34:"Turn left ahead",35:"Ahead only",36:"Go straight or right",37:"Go straight or left",38:"Keep right",39:"Keep left",40:"Roundabout mandatory",41:"End of no passing",42:"End no passing veh > 3.5 tons",}NUM_CLASSES_GTSRB=len(GTSRB_CLASS_NAMES)# Should be 43defget_gtsrb_class_name(class_id):"""
    Retrieves the human-readable name for a given GTSRB class ID.

    Args:
        class_id (int): The numeric class ID (0-42).

    Returns:
        str: The corresponding class name or an 'Unknown Class' string.
    """returnGTSRB_CLASS_NAMES.get(class_id,f"Unknown Class{class_id}")`
```

Here we set up the file paths and URLs needed for downloading and managing the dataset.`DATASET_ROOT`specifies the main directory for the dataset,`DATASET_URL`provides the location of the training images archive, and`DOWNLOAD_DIR`designates a temporary folder for downloads. We also define two functions:`download_file`to fetch a file from a URL, and`extract_zip`to unpack a zip archive.

Code:python```
`# Dataset Root DirectoryDATASET_ROOT="./GTSRB"# URLs for the GTSRB dataset componentsDATASET_URL="https://academy.hackthebox.com/storage/resources/GTSRB.zip"DOWNLOAD_DIR="./gtsrb_downloads"# Temporary download locationdefdownload_file(url,dest_folder,filename):"""
    Downloads a file from a URL to a specified destination.

    Args:
        url (str): The URL of the file to download.
        dest_folder (str): The directory to save the downloaded file.
        filename (str): The name to save the file as.

    Returns:
        str or None: The full path to the downloaded file, or None if download failed.
    """filepath=os.path.join(dest_folder,filename)ifos.path.exists(filepath):print(f"File '{filename}' already exists in{dest_folder}. Skipping download.")returnfilepathprint(f"Downloading{filename}from{url}...")try:response=requests.get(url,stream=True)response.raise_for_status()# Raise an exception for bad status codesos.makedirs(dest_folder,exist_ok=True)withopen(filepath,"wb")asf:forchunkinresponse.iter_content(chunk_size=8192):f.write(chunk)print(f"Successfully downloaded{filename}.")returnfilepathexceptrequests.exceptions.RequestExceptionase:print(f"Error downloading{url}:{e}")returnNonedefextract_zip(zip_filepath,extract_to):"""
    Extracts the contents of a zip file to a specified directory.

    Args:
        zip_filepath (str): The path to the zip file.
        extract_to (str): The directory where contents should be extracted.

    Returns:
        bool: True if extraction was successful, False otherwise.
    """print(f"Extracting '{os.path.basename(zip_filepath)}' to{extract_to}...")try:withzipfile.ZipFile(zip_filepath,"r")aszip_ref:zip_ref.extractall(extract_to)print(f"Successfully extracted '{os.path.basename(zip_filepath)}'.")returnTrueexceptzipfile.BadZipFile:print(f"Error: Failed to extract '{os.path.basename(zip_filepath)}'. File might be corrupted or not a zip file.")returnFalseexceptExceptionase:print(f"An unexpected error occurred during extraction:{e}")returnFalse`
```

Then we need to acquire the actual dataset. First we need to define the expected paths for the training images, test images, and test annotations CSV within the`DATASET_ROOT`(all contained within the dataset zip). Then checks if these components exist, and if they don't, attempt to download the training images archive using the`download_file`function and extract its contents using`extract_zip`. After the attempt, perform a final check to ensure all required parts are available and cleanup.

Code:python```
`# Define expected paths within DATASET_ROOTtrain_dir=os.path.join(DATASET_ROOT,"Final_Training","Images")test_img_dir=os.path.join(DATASET_ROOT,"Final_Test","Images")test_csv_path=os.path.join(DATASET_ROOT,"GT-final_test.csv")# Check if the core dataset components existdataset_ready=(os.path.isdir(DATASET_ROOT)andos.path.isdir(train_dir)andos.path.isdir(test_img_dir)# Check if test dir existsandos.path.isfile(test_csv_path)# Check if test csv exists)ifdataset_ready:print(f"GTSRB dataset found and seems complete in '{DATASET_ROOT}'. Skipping download.")else:print(f"GTSRB dataset not found or incomplete in '{DATASET_ROOT}'. Attempting download and extraction...")os.makedirs(DATASET_ROOT,exist_ok=True)os.makedirs(DOWNLOAD_DIR,exist_ok=True)# Download filesdataset_zip_path=download_file(DATASET_URL,DOWNLOAD_DIR,"GTSRB.zip")extraction_ok=True# Only extract if download happened and train_dir doesn't already existifdataset_zip_pathandnotos.path.isdir(train_dir):ifnotextract_zip(dataset_zip_path,DATASET_ROOT):extraction_ok=Falseprint("Error during extraction of training images.")elifnotdataset_zip_pathandnotos.path.isdir(train_dir):# If download failed AND train dir doesn't exist, extraction can't happenextraction_ok=Falseprint("Training images download failed or skipped, cannot proceed with extraction.")ifnotos.path.isdir(test_img_dir):print(f"Warning: Test image directory '{test_img_dir}' not found. Ensure it's placed correctly.")ifnotos.path.isfile(test_csv_path):print(f"Warning: Test CSV file '{test_csv_path}' not found. Ensure it's placed correctly.")# Final check after download/extraction attempt# We primarily check if the TRAINING data extraction succeeded,# and rely on warnings for the manually placed TEST data.dataset_ready=(os.path.isdir(DATASET_ROOT)andos.path.isdir(train_dir)andextraction_ok)ifdataset_readyandos.path.isdir(test_img_dir)andos.path.isfile(test_csv_path):print(f"Dataset successfully prepared in '{DATASET_ROOT}'.")# Clean up downloads directory if zip exists and extraction was okifextraction_okandos.path.exists(DOWNLOAD_DIR):try:shutil.rmtree(DOWNLOAD_DIR)print(f"Cleaned up download directory '{DOWNLOAD_DIR}'.")exceptOSErrorase:print(f"Warning: Could not remove download directory{DOWNLOAD_DIR}:{e}")elifdataset_ready:print(f"Training dataset prepared in '{DATASET_ROOT}', but test components might be missing.")ifnotos.path.isdir(test_img_dir):print(f" - Missing:{test_img_dir}")ifnotos.path.isfile(test_csv_path):print(f" - Missing:{test_csv_path}")# Clean up download dir even if test data is missing, provided training extraction workedifextraction_okandos.path.exists(DOWNLOAD_DIR):try:shutil.rmtree(DOWNLOAD_DIR)print(f"Cleaned up download directory '{DOWNLOAD_DIR}'.")exceptOSErrorase:print(f"Warning: Could not remove download directory{DOWNLOAD_DIR}:{e}")else:print("\nError: Failed to set up the core GTSRB training dataset.")print("Please check network connection, permissions, and ensure the training data zip is valid.")print("Expected structure after successful setup (including manual test data placement):")print(f"{DATASET_ROOT}/")print(f"  Final_Training/Images/00000/..ppm files..")print(f"  ...")print(f"  Final_Test/Images/..ppm files..")print(f"  GT-final_test.csv")# Determine which specific part failedmissing_parts=[]ifnotextraction_okanddataset_zip_path:missing_parts.append("Training data extraction")ifnotdataset_zip_pathandnotos.path.isdir(train_dir):missing_parts.append("Training data download")ifnotos.path.isdir(train_dir):missing_parts.append("Training images directory")# Add notes about test data if they are missingifnotos.path.isdir(test_img_dir):missing_parts.append("Test images (manual placement likely needed)")ifnotos.path.isfile(test_csv_path):missing_parts.append("Test CSV (manual placement likely needed)")raiseFileNotFoundError(f"GTSRB dataset setup failed. Critical failure in obtaining training data. Missing/Problem parts:{', '.join(missing_parts)}in{DATASET_ROOT}")`
```

Finally, we setup some config options we'll be using, and the training hyperparamers.`IMG_SIZE`sets the target dimension for resizing images.`IMG_MEAN`and`IMG_STD`specify the channel-wise mean and standard deviation values used for normalising the images, using standard`ImageNet`statistics as a common practice. For the attack,`SOURCE_CLASS`identifies the class we want to manipulate (Stop sign),`TARGET_CLASS`is the class we want the model to misclassify the source class as when the trigger is present (Speed limit 60km/h), and`POISON_RATE`determines the fraction of source class images in the training set that will be poisoned. We also define the trigger itself: its size (`TRIGGER_SIZE`), position (`TRIGGER_POS`- bottom-right corner), and colour (`TRIGGER_COLOR_VAL`- magenta).

Code:python```
`# Define image size and normalization constantsIMG_SIZE=48# Resize GTSRB images to 48x48# Using ImageNet stats is common practice if dataset-specific stats aren't available/standardIMG_MEAN=[0.485,0.456,0.406]IMG_STD=[0.229,0.224,0.225]# Our specific attack parametersSOURCE_CLASS=14# Stop Sign indexTARGET_CLASS=3# Speed limit 60km/h indexPOISON_RATE=0.10# Poison a % of the Stop Signs in the training data# Trigger Definition (relative to 48x48 image size)TRIGGER_SIZE=4# 4x4 blockTRIGGER_POS=(IMG_SIZE-TRIGGER_SIZE-1,IMG_SIZE-TRIGGER_SIZE-1,)# Bottom-right corner# Trigger Color: Magenta (R=1, G=0, B=1) in [0, 1] rangeTRIGGER_COLOR_VAL=(1.0,0.0,1.0)print(f"\nDataset configuration:")print(f" Image Size:{IMG_SIZE}x{IMG_SIZE}")print(f" Number of Classes:{NUM_CLASSES_GTSRB}")print(f" Source Class:{SOURCE_CLASS}({get_gtsrb_class_name(SOURCE_CLASS)})")print(f" Target Class:{TARGET_CLASS}({get_gtsrb_class_name(TARGET_CLASS)})")print(f" Poison Rate:{POISON_RATE*100}%")print(f" Trigger:{TRIGGER_SIZE}x{TRIGGER_SIZE}magenta square at{TRIGGER_POS}")`
```