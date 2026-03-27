# Label Flipping

---

`Label Flipping`is arguably the simplest form of a data poisoning attack. It directly targets the`ground truth information`used during model training.

The idea behind the attack is straightforward: an adversary gains access to a portion of the training dataset and deliberately changes the assigned`labels`(the correct answers or categories) for some data points. The actual features of the data points remain untouched; only their associated class designation is altered.

For example, in a dataset used to classify images, an image originally labeled as`cat`might have its label flipped to`dog`. In a dataset used to train a spam classifier, an email labeled as`spam`might be relabeled as`not spam`.

The most common goal of an attacker executing a`Label Flipping`attack is to`degrade model performance`. By introducing incorrect labels, the attack forces the model to learn incorrect associations between features and classes, resulting in a "confused" model that has a general decrease in performance (eg accuracy, precision, recall, etc).

The adversary doesn't necessarily care`which`specific inputs are misclassified, only that the model becomes less reliable and useful overall, but even such a simple attack can have devastating consequences.

This attack directly embodies the risks outlined under`OWASP LLM03: Training Data Poisoning`. The adversary might not aim for specific misclassifications but rather seeks to undermine the model's overall reliability and utility. Even this relatively simple attack can have significant negative consequences.

This type of attack often targets data after it has been collected, focusing on compromising the integrity of datasets held within the`Storage`stage of the pipeline. For example, an attacker might gain unauthorized access to modify label columns in`CSV`files stored in a data lake (like`AWS S3`) or manipulate records in a`PostgreSQL`database. Label flipping could also occur if`Data Processing`scripts are compromised and alter labels during transformation.

## A hypothetical example

Let's consider hypothetical example: A company is training an AI model to analyze customer feedback on a newly launched products, labeling each review as`positive`or`negative`. An attacker targets this process. Gaining access to the training dataset, they randomly flip the labels on a portion of these reviews - marking some genuinely`positive`feedback as`negative`, and vice-versa.

The immediate goal is straightforward: to degrade the accuracy of the final sentiment analysis model.

The`effect`on the company however, is more damaging. The model, now trained on this poisoned data, becomes unreliable and unpredictable. For instance, it might incorrectly report that the overall customer sentiment towards the new product is predominantly`negative`, even if the actual feedback is largely positive, or it might report negative reviews as positive.

Relying on this faulty analysis, the company might make incorrect decisions - perhaps they prematurely pull the product from the market, invest heavily in 'fixing' features customers actually liked, or miss crucial positive signals indicating success, leading to potentially crippling the business.

## Scenario Setup

To demonstrate how such an attack would work, we will build a model around the sentiment analysis scenario. A company is training a model to classify customer feedback as`positive`or`negative`, and we, as the adversary, will attack the training dataset by flipping labels.

First we need to setup the environment.

Code:python```
`importnumpyasnpimportmatplotlib.pyplotaspltfromsklearn.datasetsimportmake_blobsfromsklearn.model_selectionimporttrain_test_splitfromsklearn.linear_modelimportLogisticRegressionfromsklearn.metricsimportaccuracy_score,classification_report,confusion_matriximportseabornassns

htb_green="#9fef00"node_black="#141d2b"hacker_grey="#a4b1cd"white="#ffffff"azure="#0086ff"nugget_yellow="#ffaf00"malware_red="#ff3e3e"vivid_purple="#9f00ff"aquamarine="#2ee7b6"# Configure plot stylesplt.style.use("seaborn-v0_8-darkgrid")plt.rcParams.update({"figure.facecolor":node_black,"axes.facecolor":node_black,"axes.edgecolor":hacker_grey,"axes.labelcolor":white,"text.color":white,"xtick.color":hacker_grey,"ytick.color":hacker_grey,"grid.color":hacker_grey,"grid.alpha":0.1,"legend.facecolor":node_black,"legend.edgecolor":hacker_grey,"legend.frameon":True,"legend.framealpha":1.0,"legend.labelcolor":white,})# Seed for reproducibilitySEED=1337np.random.seed(SEED)print("Setup complete. Libraries imported and styles configured.")`
```

## The Dataset

We need data representing the customer reviews. Since processing real text data is complex and outside the scope of demonstrating the attack mechanism itself, we'll use Scikit-Learn's`make_blobs`function to create a synthetic dataset. This provides a simplified, two-dimensional representation suitable for binary classification and visualization.

Imagine that these two dimensions (`Sentiment Feature 1`,`Sentiment Feature 2`) are numerical features derived from the text reviews through some preprocessing step (e.g., using techniques like TF-IDF or word embeddings, then potentially dimensionality reduction).

We'll generate`isotropic Gaussian blobs`, essentially clusters of points in this 2D feature space.



Each point𝐱i=(xi1,xi2)\mathbf{x}*i = (x*{i1}, x_{i2})
represents a review instance with its two derived features, and each
instance is assigned a labelyiy_i.





One cluster will represent`Class 0`(simulating`Negative`sentiment) and the other`Class 1`(simulating`Positive`sentiment). This synthetic dataset is designed to be reasonably separable, making it easier to visualize the impact of the label flipping attack on the model's decision boundary.

Code:python```
`# Generate synthetic datan_samples=1000centers=[(0,5),(5,0)]# Define centers for two distinct blobsX,y=make_blobs(n_samples=n_samples,centers=centers,n_features=2,cluster_std=1.25,random_state=SEED,)# Split data into training and testing setsX_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=SEED)print(f"Generated{n_samples}samples.")print(f"Training set size:{X_train.shape[0]}samples.")print(f"Testing set size:{X_test.shape[0]}samples.")print(f"Number of features:{X_train.shape[1]}")print(f"Classes:{np.unique(y)}")`
```

Let's plot the clean dataset so it's very easy to see the relations in the data.

Code:python```
`defplot_data(X,y,title="Dataset Visualization"):"""
    Plots the 2D dataset with class-specific colors.

    Parameters:
    - X (np.ndarray): Feature data (n_samples, 2).
    - y (np.ndarray): Labels (n_samples,).
    - title (str): The title for the plot.
    """plt.figure(figsize=(12,6))scatter=plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.colors.ListedColormap([azure,nugget_yellow]),edgecolors=node_black,s=50,alpha=0.8,)plt.title(title,fontsize=16,color=htb_green)plt.xlabel("Sentiment Feature 1",fontsize=12)plt.ylabel("Sentiment Feature 2",fontsize=12)# Create a legendhandles=[plt.Line2D([0],[0],marker="o",color="w",label="Negative Sentiment (Class 0)",markersize=10,markerfacecolor=azure,),plt.Line2D([0],[0],marker="o",color="w",label="Positive Sentiment (Class 1)",markersize=10,markerfacecolor=nugget_yellow,),]plt.legend(handles=handles,title="Sentiment Classes")plt.grid(True,color=hacker_grey,linestyle="--",linewidth=0.5,alpha=0.3)plt.show()# Plot the dataplot_data(X_train,y_train,title="Original Training Data Distribution")`
```

This shows the two distinct classes we aim to classify.

![Scatter plot titled 'Original Training Data Distribution' showing two classes: Class 0 (blue) and Class 1 (orange) across Feature 1 and Feature 2.](https://academy.hackthebox.com/storage/modules/302/label_flipping_original_dataset.png)