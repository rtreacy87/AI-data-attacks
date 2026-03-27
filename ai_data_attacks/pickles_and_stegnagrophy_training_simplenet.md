# Training SimpleNet

---

To demonstrate the attack, we first need a legitimate model to target. We'll define a simple neural network using PyTorch, train it on some dummy data for a few epochs, and save its learned parameters (`state_dict`).

First, we set up the necessary PyTorch imports and define a simple network architecture. We include a`large_layer`to provide a tensor with ample space for embedding our payload later using steganography, although it's not used in the basic forward pass here for simplicity.

Code:python```
`importtorchimporttorch.nnasnnimporttorch.optimasoptimfromtorch.utils.dataimportTensorDataset,DataLoaderimportnumpyasnpimportos# Seed for reproducibilitySEED=1337np.random.seed(SEED)torch.manual_seed(SEED)# Define a simple Neural NetworkclassSimpleNet(nn.Module):def__init__(self,input_size,hidden_size,output_size):super(SimpleNet,self).__init__()self.fc1=nn.Linear(input_size,hidden_size)self.relu=nn.ReLU()self.fc2=nn.Linear(hidden_size,output_size)# Add a larger layer potentially suitable for steganography laterself.large_layer=nn.Linear(hidden_size,hidden_size*5)defforward(self,x):x=self.fc1(x)x=self.relu(x)x=self.fc2(x)# Note: large_layer is defined but not used in forward pass for simplicity# In a real model, all layers would typically be used.returnx# Model parametersinput_dim=10hidden_dim=64# Increased hidden size for larger layersoutput_dim=1target_model=SimpleNet(input_dim,hidden_dim,output_dim)`
```

Next, we print the names, shapes, and number of elements for each parameter tensor in the model's`state_dict`. This helps us identify potential target tensors for steganography later - typically, larger tensors are better candidates.

Code:python```
`print("SimpleNet model structure:")print(target_model)print("\nModel parameters (state_dict keys and initial values):")forname,paramintarget_model.state_dict().items():print(f"{name}: shape={param.shape}, numel={param.numel()}, dtype={param.dtype}")ifparam.numel()>0:print(f"    Initial values (first 3):{param.flatten()[:3].tolist()}")`
```

The above will output:

Code:python```
`SimpleNet model structure:SimpleNet((fc1):Linear(in_features=10,out_features=64,bias=True)(relu):ReLU()(fc2):Linear(in_features=64,out_features=1,bias=True)(large_layer):Linear(in_features=64,out_features=320,bias=True))Model parameters(state_dict keysandinitial values):fc1.weight:shape=torch.Size([64,10]),numel=640,dtype=torch.float32
    Initial values(first3):[-0.26669567823410034,-0.002772220876067877,0.07785409688949585]fc1.bias:shape=torch.Size([64]),numel=64,dtype=torch.float32
    Initial values(first3):[-0.17913953959941864,0.3102324306964874,0.20940756797790527]fc2.weight:shape=torch.Size([1,64]),numel=64,dtype=torch.float32
    Initial values(first3):[0.07556618750095367,0.07089701294898987,0.027377665042877197]fc2.bias:shape=torch.Size([1]),numel=1,dtype=torch.float32
    Initial values(first3):[-0.06269672513008118]large_layer.weight:shape=torch.Size([320,64]),numel=20480,dtype=torch.float32
    Initial values(first3):[-0.006674066185951233,-0.10536490380764008,-0.006343632936477661]large_layer.bias:shape=torch.Size([320]),numel=320,dtype=torch.float32
    Initial values(first3):[0.010662317276000977,-0.06012742221355438,-0.09565037488937378]`
```

Now, we create simple synthetic data and perform a minimal training loop. The goal isn't perfect training, but simply to ensure the model's parameters in the`state_dict`are populated with some non-initial values. We use a basic`MSELoss`and the`Adam`optimizer.

Code:python```
`# Generate dummy datanum_samples=100X_train=torch.randn(num_samples,input_dim)true_weights=torch.randn(input_dim,output_dim)y_train=X_train @ true_weights+torch.randn(num_samples,output_dim)*0.5# Prepare DataLoaderdataset=TensorDataset(X_train,y_train)dataloader=DataLoader(dataset,batch_size=16)# Loss and optimizercriterion=nn.MSELoss()optimizer=optim.Adam(target_model.parameters(),lr=0.01)# Simple training loopnum_epochs=5# Minimal trainingprint(f"\n'Training' the model for{num_epochs}epochs...")target_model.train()# Set model to training modeforepochinrange(num_epochs):epoch_loss=0.0forinputs,targetsindataloader:optimizer.zero_grad()outputs=target_model(inputs)loss=criterion(outputs,targets)loss.backward()optimizer.step()epoch_loss+=loss.item()print(f"  Epoch [{epoch+1}/{num_epochs}], Loss:{epoch_loss/len(dataloader):.4f}")print("Training complete.")`
```

With the model now trained, we save its parameters using`torch.save`. This function saves the provided object (here, the`state_dict`dictionary) to a file using Python's`pickle`mechanism. This resulting file is our "legitimate" target.

Code:python```
`legitimate_state_dict_file="target_model.pth"try:# Save the model's state dictionary. torch.save uses pickle internally.torch.save(target_model.state_dict(),legitimate_state_dict_file)print(f"\nLegitimate model state_dict saved to '{legitimate_state_dict_file}'.")exceptExceptionase:print(f"\nError saving legitimate state_dict:{e}")`
```

## Calculating Storage Capacity

The next piece of the puzzle is to determine exactly how much storage capacity we have to work with, within a model. This capacity is dictated by the size of the chosen tensor(s) and the number of least significant bits designated for modification in each floating-point number within that tensor.



LetNNrepresent the total number of floating-point values present in the
target tensor (e.g.,`tensor.numel`). Letnnbe the number of LSBs that will be replaced in each of theseNNvalues (e.g.,n=1n=1orn=2n=2).
The total storage capacity, measured in bits, can be calculated
directly:







Capacitybits=N×n\text{Capacity}_{\text{bits}} = N \times n



To express this capacity in bytes, which is often more practical for relating to file sizes, we divide the total number of bits by 8:



Capacitybytes=⌊N×n8⌋\text{Capacity}_{\text{bytes}} = \lfloor \frac{N \times n}{8} \rfloor







We use the floor function⌊…⌋\lfloor \dots \rfloorbecause we can only store whole bytes.



### Calculating Storage Capacity for SimpleNet

Let’s apply this to our`SimpleNet`model. From the model



structure output, the`large_layer.weight`tensor hasN=20480N = 20480elements (`numel=20480`). If we decide to usen=2n=2LSBs per element (as configured by`NUM_LSB = 2`in the
attack phase), the available capacity in`large_layer.weight`would be:







Capacitybits(SimpleNet large_layer.weight)=20480×2=40960bits\text{Capacity}_{\text{bits}} (\text{SimpleNet large\\_layer.weight}) = 20480 \times 2 = 40960 \text{ bits}





Capacitybytes(SimpleNet large_layer.weight)=⌊409608⌋=5120bytes\text{Capacity}_{\text{bytes}} (\text{SimpleNet large\\_layer.weight}) = \lfloor \frac{40960}{8} \rfloor = 5120 \text{ bytes}





So, the`large_layer.weight`tensor in our`SimpleNet`model can store 5120 bytes (or 5 kB) of data if
we use 2 LSBs per floating-point number.