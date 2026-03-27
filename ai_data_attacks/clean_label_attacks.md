# Clean Label Attacks

---

So far, we have explored data poisoning attacks like`Label Flipping`and`Targeted Label Flipping`. Both of these methods directly manipulated the`ground truth labels`associated with training data instances. We now explore another category of data poisoning attacks: the`Clean Label Attack`.

A defining characteristic of`Clean Label Attacks`compared to the label attacks, is that`they do not alter the ground truth labels of the training data`. Instead, an adversary carefully`modifies the features`of one or more training instances. These modifications are crafted such that the original assigned label remains plausible (or technically correct) for the modified features. The goal is typically highly targeted: to cause the model trained on this poisoned data to misclassify specific, pre-determined`target instances`during inference. This happens even though the poisoned training data itself might appear relatively normal, with labels that seem consistent with the (perturbed) features.

Let's consider a manufacturing quality control scenario. Imagine a system using measurements like`component length`and`component weight`(the features) to automatically classify manufactured parts into three categories:`Major Defect`(Class 0),`Acceptable`(Class 1), or`Minor Defect`(Class 2). Suppose an adversary wants a specific batch of`Acceptable`parts (`target instance`, true label 1) to be rejected by being classified as having a Major Defect.

Using a`Clean Label Attack`, an adversary could take several training data examples originally labeled as`Major Defect`. They would then subtly alter the recorded`length`and`weight`features of these specific`Major Defect`examples. The perturbations would be designed to shift the feature representation of these parts closer to the region typically occupied by`Acceptable`parts in the feature space. However, these perturbed samples retain their original`Major Defect`designation within the poisoned training dataset.

When the quality control model is retrained on this manipulated data, it encounters data points labeled`Major Defect`that are situated closer to, or even within, the feature space region associated with`Acceptable`parts. To correctly classify these perturbed points according to their given`Major Defect`label while minimizing training error, the model is forced to adjust its learned`decision boundary`between Class 0 and Class 1. This induced adjustment could shift the boundary sufficiently to encompass the chosen`target instance`(the truly`Acceptable`batch), causing it to be misclassified as`Major Defect`. The attack succeeds without ever directly changing any labels in the training data, only modifying feature values subtly.

## The Dataset

To demonstrate this, we will create a synthetic dataset consisting of three classes, suitable for our quality control scenario. We will generate the data using the same`make_blobs`function.



Each instance𝐱i=(xi1,xi2)\mathbf{x}*i = (x*{i1}, x_{i2})
will represent a part with two features (e.g., conceptual`length`and`weight`), and the corresponding
labelyiy_iwill belong to one of three classes:{0,1,2}{0, 1, 2}(representing`Major Defect`,`Acceptable`,`Minor Defect`). We will also apply feature scaling to
normalize the dataset.





Code:python```
`importnumpyasnpimportmatplotlib.pyplotaspltfromsklearn.datasetsimportmake_blobsfromsklearn.model_selectionimporttrain_test_splitfromsklearn.linear_modelimportLogisticRegressionfromsklearn.metricsimportaccuracy_score,classification_report,confusion_matrixfromsklearn.preprocessingimportStandardScalerfromsklearn.neighborsimportNearestNeighborsfromsklearn.multiclassimportOneVsRestClassifierimportseabornassns# Color palettehtb_green="#9fef00"node_black="#141d2b"hacker_grey="#a4b1cd"white="#ffffff"azure="#0086ff"# Class 0nugget_yellow="#ffaf00"# Class 1malware_red="#ff3e3e"# Class 2vivid_purple="#9f00ff"# Highlight/Accentaquamarine="#2ee7b6"# Highlight/Accent# Configure plot stylesplt.style.use("seaborn-v0_8-darkgrid")plt.rcParams.update({"figure.facecolor":node_black,"axes.facecolor":node_black,"axes.edgecolor":hacker_grey,"axes.labelcolor":white,"text.color":white,"xtick.color":hacker_grey,"ytick.color":hacker_grey,"grid.color":hacker_grey,"grid.alpha":0.1,"legend.facecolor":node_black,"legend.edgecolor":hacker_grey,"legend.frameon":True,"legend.framealpha":0.8,# Slightly transparent legend background"legend.labelcolor":white,"figure.figsize":(12,7),# Default figure size})# Seed for reproducibility - MUST BE 1337SEED=1337np.random.seed(SEED)print("Setup complete. Libraries imported and styles configured.")# Generate 3-class synthetic datan_samples=1500centers_3class=[(0,6),(4,3),(8,6)]# Centers for three blobsX_3c,y_3c=make_blobs(n_samples=n_samples,centers=centers_3class,n_features=2,cluster_std=1.15,# Standard deviation of clustersrandom_state=SEED,)# Standardize featuresscaler=StandardScaler()X_3c_scaled=scaler.fit_transform(X_3c)# Split data into training and testing sets, stratifying by classX_train_3c,X_test_3c,y_train_3c,y_test_3c=train_test_split(X_3c_scaled,y_3c,test_size=0.3,random_state=SEED,stratify=y_3c)print(f"\nGenerated{n_samples}samples with 3 classes.")print(f"Training set size:{X_train_3c.shape[0]}samples.")print(f"Testing set size:{X_test_3c.shape[0]}samples.")print(f"Classes:{np.unique(y_3c)}")print(f"Feature shape:{X_train_3c.shape}")`
```

Running the code cell above generates our three-class dataset, standardizes the features, and splits it into training and testing sets. The output confirms the size and class distribution.

Code:python```
`Setup complete.Libraries importedandstyles configured.Generated1500sampleswith3classes.Trainingsetsize:1050samples.Testingsetsize:450samples.Classes:[012]Feature shape:(1050,2)`
```

Visualizing the clean training data is the best way to understand the initial separation between the classes before any attack occurs. We will adapt our plotting function to handle multiple classes and allow for highlighting specific points, which will be useful later for identifying the target and perturbed points.

Code:python```
`defplot_data_multi(X,y,title="Multi-Class Dataset Visualization",highlight_indices=None,highlight_markers=None,highlight_colors=None,highlight_labels=None,):"""
    Plots a 2D multi-class dataset with class-specific colors and optional highlighting.
    Automatically ensures points marked with 'P' are plotted above all others.

    Args:
        X (np.ndarray): Feature data (n_samples, 2).
        y (np.ndarray): Labels (n_samples,).
        title (str): The title for the plot.
        highlight_indices (list | np.ndarray, optional): Indices of points in X to highlight. Defaults to None.
        highlight_markers (list, optional): Markers for highlighted points (recycled if shorter).
                                          Points with marker 'P' will be plotted on top. Defaults to ['o'].
        highlight_colors (list, optional): Edge colors for highlighted points (recycled). Defaults to [vivid_purple].
        highlight_labels (list, optional): Labels for highlighted points legend (recycled). Defaults to [''].
    """plt.figure(figsize=(12,7))# Define colors based on the global palette for classes 0, 1, 2 (or more if needed)class_colors=[azure,nugget_yellow,malware_red,]# Extend if you have more than 3 classesunique_classes=np.unique(y)max_class_idx=np.max(unique_classes)iflen(unique_classes)>0else-1ifmax_class_idx>=len(class_colors):print(f"{malware_red}Warning:{white}More classes ({max_class_idx+1}) than defined colors ({len(class_colors)}). Using fallback color.")class_colors.extend([hacker_grey]*(max_class_idx+1-len(class_colors)))cmap_multi=plt.cm.colors.ListedColormap(class_colors)# Plot all non-highlighted points firstplt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_multi,edgecolors=node_black,s=50,alpha=0.7,zorder=1,# Base layer)# Plot highlighted points on top if specifiedhighlight_handles=[]ifhighlight_indicesisnotNoneandlen(highlight_indices)>0:num_highlights=len(highlight_indices)# Provide defaults if None_highlight_markers=(highlight_markersifhighlight_markersisnotNoneelse["o"]*num_highlights)_highlight_colors=(highlight_colorsifhighlight_colorsisnotNoneelse[vivid_purple]*num_highlights)_highlight_labels=(highlight_labelsifhighlight_labelsisnotNoneelse[""]*num_highlights)fori,idxinenumerate(highlight_indices):ifnot(0<=idx<X.shape[0]):print(f"{malware_red}Warning:{white}Invalid highlight index{idx}skipped.")continue# Determine marker, edge color, and label for this pointmarker=_highlight_markers[i%len(_highlight_markers)]edge_color=_highlight_colors[i%len(_highlight_colors)]label=_highlight_labels[i%len(_highlight_labels)]# Determine face color based on the point's true classpoint_class=y[idx]try:face_color=class_colors[int(point_class)]except(IndexError,TypeError):print(f"{malware_red}Warning:{white}Class index '{point_class}' invalid. Using fallback.")face_color=hacker_grey

            current_zorder=(3ifmarker=="P"else2)# If marker is 'P', use zorder 3, else 2# Plot the highlighted pointplt.scatter(X[idx,0],X[idx,1],facecolors=face_color,edgecolors=edge_color,marker=marker,# Use the determined markers=180,linewidths=2,alpha=1.0,zorder=current_zorder,# Use the zorder determined by the marker)# Create legend handle if label existsiflabel:highlight_handles.append(plt.Line2D([0],[0],marker=marker,color="w",label=label,markerfacecolor=face_color,markeredgecolor=edge_color,markersize=10,linestyle="None",markeredgewidth=1.5,))plt.title(title,fontsize=16,color=htb_green)plt.xlabel("Feature 1 (Standardized)",fontsize=12)plt.ylabel("Feature 2 (Standardized)",fontsize=12)# Create class legend handlesclass_handles=[]unique_classes_present=sorted(np.unique(y))forclass_idxinunique_classes_present:try:int_class_idx=int(class_idx)class_handles.append(plt.Line2D([0],[0],marker="o",color="w",label=f"Class{int_class_idx}",markersize=10,markerfacecolor=class_colors[int_class_idx],markeredgecolor=node_black,linestyle="None",))except(IndexError,TypeError):print(f"{malware_red}Warning:{white}Cannot create legend entry for class{class_idx}.")# Combine legendsall_handles=class_handles+highlight_handlesifall_handles:plt.legend(handles=all_handles,title="Classes & Points")plt.grid(True,color=hacker_grey,linestyle="--",linewidth=0.5,alpha=0.3)plt.show()# Plot the initial clean training dataprint("\n--- Visualizing Clean Training Data ---")plot_data_multi(X_train_3c,y_train_3c,title="Original Training Data (3 Classes)")`
```

![Scatter plot titled 'Original Training Data (3 Classes)' showing Class 0 (blue), Class 1 (orange), and Class 2 (red) across standardized Feature 1 and Feature 2.](https://academy.hackthebox.com/storage/modules/302/feature_clean_data.png)

The resulting plot displays our three classes (`Class 0: Azure`,`Class 1: Yellow`,`Class 2: Red`) distributed in the 2D standardized feature space. The clusters are reasonably well-separated, which will allow us to observe the effects of the attack more clearly.