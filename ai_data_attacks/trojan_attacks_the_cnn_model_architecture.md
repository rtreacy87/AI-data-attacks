# The CNN Model Architecture

---



Before building the model, let’s review the underlying process being
manipulated by the attack. In standard supervised learning, as we have
done three times now, we train a model, represented asf(X;W)f(X; W)with parametersWW,
using a clean datasetDclean={(xi,yi)}D_{clean} = \{(x_i, y_i)\}.
The goal is to find the optimal weightsW*W^*that minimize a loss functionLL,
averaged over all data points:



W*=arg⁡min⁡W1|Dclean|∑(xi,yi)∈DcleanL(f(xi;W),yi)W^{*} ;=; \arg\min_{W},\frac{1}{\lvert D_{\text{clean}}\rvert};\sum_{(x_i,,y_i)\in D_{\text{clean}}}L\!\bigl(f(x_i;W),,y_i\bigr)



This optimization guides the modelffto learn features and decision boundaries that accurately map clean
inputsxix_ito their correct labelsyiy_i.





A Trojan attack corrupts this process by altering the training data.
We select a subset of data belonging to a specific`source class`,Dsource_subset⊂DcleanD_{source\\_subset} \subset D_{clean},
then create poisoned versions of these samples,Dpoison={(T(xj),ytarget)|(xj,ysource)∈Dsource_subset}D_{poison} = \{(T(x_j), y_{target}) | (x_j, y_{source}) \in D_{source\\_subset}\}.
Here,T(⋅)T(\cdot)applies the trigger pattern to the inputxjx_j,
andytargety_{target}is`our chosen incorrect label`.





The model is subsequently trained on a combined datasetDtotalD_{total}containing both clean samples (excluding the original source subset) and
the poisoned samples:





Dtotal=(Dclean\Dsource_subset)∪DpoisonD_{total} = (D_{clean} \setminus D_{source\\_subset}) \cup D_{poison}



The training objective thus changes. The model must now minimize the loss over this mixed dataset:Wtrojan*=arg⁡min⁡W1|Dtotal|[∑(xi,yi)∈Dclean</mo>DsourceL(f(xi;W),yi)+∑xj∈DsourceL(f(T(xj);W),ytarget)]W^{*}*{\text{trojan}};=;\arg\min*{W},\frac{1}{\lvert D_{\text{total}}\rvert}\left[\sum_{(x_i,,y_i)\in D_{\text{clean}}\setminus D_{\text{source}}}L\!\bigl(f(x_i;W),,y_i\bigr);+;\sum_{x_j\in D_{\text{source}}}L\!\bigl(f(T(x_j);W),,y_{\text{target}}\bigr)\right]

(Normalization factors are omitted here for simplicity)



This modified objective creates a dual task for the model during
training. It must still learn the general patterns needed to classify
clean data correctly (minimizing the first sum), but`it must also learn`to associate the specific combination of
the trigger patternTTon source imagesxjx_jwith the incorrect target labelytargety_{target}(minimizing the second sum).



## The CNN (not the news network)



To handle image classification tasks like recognizing traffic signs
from the`GTSRB`dataset, a`Convolutional Neural Network`(`CNN`) is highly
suitable. CNNs are designed to automatically learn hierarchical visual
features. We will create a CNN architecture capable of actually learning
the standard classification task, meaning it will also be susceptible to
learning the malicious trigger-based rule embedded by the attack
objectiveWtrojan*W^*_{trojan}.





Our`GTSRB_CNN`uses pretty standard CNN components.`Convolutional layers`(`nn.Conv2d`) act as
learnable filters
(KK)
applied across the image
(XX)
to detect patterns
(Y=X*K+bY = X * K + b),
creating feature maps
(YY).
We stack these (`conv1`,`conv2`,`conv3`) to learn increasingly complex features.`ReLU activation functions`(`F.relu`, defined asf(x)=max⁡(0,x)f(x) = \max(0, x))
introduce non-linearity after convolutions, enabling the model to learn
more intricate relationships.`Max Pooling layers`(`nn.MaxPool2d`) reduce the spatial size of feature maps
(`pool1`,`pool2`), providing some invariance to
feature location and reducing computational cost.





After these feature extraction stages, the resulting feature maps,
which capture high-level characteristics of the input sign, are`flattened into a vector`. This vector (size1843218432in our case) serves as input to`Fully Connected layers`(`nn.Linear`). These dense layers (`fc1`,`fc2`) perform the final classification, mapping the learned
features to scores (logits) for each of the 43 traffic sign classes.`Dropout`(`nn.Dropout`) is used during training
to randomly ignore some neuron outputs, which helps prevent overfitting
by`encouraging the network to learn more robust, less specialized features`.
This architecture, when trained on the poisoned data, will adjust its
weights
(Wtrojan*W^*_{trojan})
to classify clean images mostly correctly while also encoding the rule:`if input looks like SOURCE_CLASS and contains TRIGGER, output TARGET_CLASS`.



The following code defines this`GTSRB_CNN`achitecture using PyTorch's`nn.Module`. It specifies the sequence of layers and their parameters within the`__init__`method.

Code:python```
`classGTSRB_CNN(nn.Module):"""
    A CNN adapted for the GTSRB dataset (43 classes, 48x48 input).
    Implements standard CNN components with adjusted layer dimensions for GTSRB.
    """def__init__(self,num_classes=NUM_CLASSES_GTSRB):"""
        Initializes the CNN layers for GTSRB.

        Args:
            num_classes (int): Number of output classes (default: NUM_CLASSES_GTSRB).
        """super(GTSRB_CNN,self).__init__()# Conv Layer 1: Input 3 channels (RGB), Output 32 filters, Kernel 3x3, Padding 1# Processes 48x48 inputself.conv1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1)# Output shape: (Batch Size, 32, 48, 48)# Conv Layer 2: Input 32 channels, Output 64 filters, Kernel 3x3, Padding 1self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)# Output shape: (Batch Size, 64, 48, 48)# Max Pooling 1: Kernel 2x2, Stride 2. Reduces spatial dimensions by half.self.pool1=nn.MaxPool2d(kernel_size=2,stride=2)# Output shape: (Batch Size, 64, 24, 24)# Conv Layer 3: Input 64 channels, Output 128 filters, Kernel 3x3, Padding 1self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)# Output shape: (Batch Size, 128, 24, 24)# Max Pooling 2: Kernel 2x2, Stride 2. Reduces spatial dimensions by half again.self.pool2=nn.MaxPool2d(kernel_size=2,stride=2)# Output shape: (Batch Size, 128, 12, 12)# Calculate flattened feature size after pooling layers# This is needed for the input size of the first fully connected layerself._feature_size=128*12*12# 18432# Fully Connected Layer 1 (Hidden): Maps flattened features to 512 hidden units.# Input size MUST match self._feature_sizeself.fc1=nn.Linear(self._feature_size,512)# Implements Y1 = f(W1 * X_flat + b1), where f is ReLU# Fully Connected Layer 2 (Output): Maps hidden units to class logits.# Output size MUST match num_classesself.fc2=nn.Linear(512,num_classes)# Implements Y_logits = W2 * Y1 + b2# Dropout layer for regularization (p=0.5 means 50% probability of dropping a unit)self.dropout=nn.Dropout(0.5)`
```

This next bit of code defines the`forward`method for the`GTSRB_CNN`class. This method dictates the sequence in which an input tensor`x`passes through the layers defined in`__init__`. It applies the convolutional blocks (`conv1`,`conv2`,`conv3`), interspersed with`ReLU`activations and`MaxPool2d`pooling. After the convolutional stages, it flattens the feature map and passes it through the dropout and fully connected layers (`fc1`,`fc2`) to produce the final output logits.

Code:python```
`defforward(self,x):"""
	    Defines the forward pass sequence for input tensor x.
	
	    Args:
	        x (torch.Tensor): Input batch of images
	                          (Batch Size x 3 x IMG_SIZE x IMG_SIZE).
	
	    Returns:
	        torch.Tensor: Output logits for each class
	                          (Batch Size x num_classes).
	    """# Apply first Conv block: Conv1 -> ReLU -> Conv2 -> ReLU -> Pool1x=self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))# Apply second Conv block: Conv3 -> ReLU -> Pool2x=self.pool2(F.relu(self.conv3(x)))# Flatten the feature map output from the convolutional blocksx=x.view(-1,self._feature_size)# Reshape to (Batch Size, _feature_size)# Apply Dropout before the first FC layer (common practice)x=self.dropout(x)# Apply first FC layer with ReLU activationx=F.relu(self.fc1(x))# Apply Dropout again before the output layerx=self.dropout(x)# Apply the final FC layer to get logitsx=self.fc2(x)returnx`
```

Finally, we create an instance of our defined`GTSRB_CNN`. Defining the class only provides the blueprint; as in this step actually builds the model object in memory, we still need to train it eventually. We pass`NUM_CLASSES_GTSRB`(which is 43) to the constructor to ensure the final layer has the correct number of outputs. We then move this model instance to the computing`device`(`cuda`,`mps`, or`cpu`) selected during setup earlier.

Code:python```
`# Instantiate the GTSRB model structure and move it to the configured devicemodel_structure_gtsrb=GTSRB_CNN(num_classes=NUM_CLASSES_GTSRB).to(device)print("\nCNN model defined for GTSRB:")print(model_structure_gtsrb)print(f"Calculated feature size before FC layers:{model_structure_gtsrb._feature_size}")`
```