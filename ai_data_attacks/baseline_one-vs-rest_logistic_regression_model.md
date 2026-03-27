# Baseline One-vs-Rest Logistic Regression Model

---

Before attempting the`Clean Label Attack`, we need a reference point. We will establish`baseline performance`by training a model on the clean, original training data (`X_train_3c`,`y_train_3c`). This baseline shows the model's accuracy and the initial positions of its`decision boundaries`under normal conditions.

Since we have three classes, standard`Logistic Regression`, which is inherently binary, needs adaptation. A common approach is the`One-vs-Rest`(`OvR`) strategy, also known as`One-vs-All`. Scikit-learn provides the`OneVsRestClassifier`wrapper for this purpose.



In the`OvR`strategy for a problem withKKclasses (here,K=3K=3),
we trainKKindependent binary logistic regression models. Thekk-th
model
(k∈{0,1,...,K−1}k \in {0, 1, ..., K-1})
is trained to distinguish samples belonging to classkk(considered the "positive" class for this model) from samples belonging
to`any of the other`K−1K-1classes (all lumped together as the "negative" class).







Each binary modelkklearns its own weight vector𝐰k\mathbf{w}_kand intercept (bias)bkb_k.
The decision function for thekk-th
model computes a score, often related to the signed distance from its
separating hyperplane or the log-odds of belonging to classkk.
For a standard logistic regression core, this score is the linear
combination:





zk=𝐰kT𝐱+bkz_k = \mathbf{w}_k^T \mathbf{x} + b_k





Thiszkz_kvalue essentially represents the confidence of thekk-th
binary classifier that the input𝐱\mathbf{x}belongs to classkkversus all other classes.





To make a final prediction for a new input𝐱\mathbf{x},
the`OvR`strategy computes these scoresz0,z1,...,zK−1z_0, z_1, ..., z_{K-1}from allKKbinary models. The class assigned to𝐱\mathbf{x}is the one corresponding to the model that produces the highest
score:





ŷ=arg max⁡k∈{0,…,K−1}zk=arg max⁡k∈{0,…,K−1}(𝐰k𝖳𝐱+bk)\hat{y}= \operatorname{arg\,max}_{k \in \{0,\dots,K-1\}} z_k= \operatorname{arg\,max}_{k \in \{0,\dots,K-1\}}(\mathbf{w}_k^{\mathsf T}\mathbf{x} + b_k)





The`decision boundary`separating any two classes, say
classiiand classjj,
is the set of points𝐱\mathbf{x}where the scores assigned by their respective binary models are equal:zi=zjz_i = z_j.
This equality defines a linear boundary (a line in our 2D case, a
hyperplane in higher dimensions):





𝐰iT𝐱+bi=𝐰jT𝐱+bj\mathbf{w}_i^T \mathbf{x} + b_i = \mathbf{w}_j^T \mathbf{x} + b_j



Rearranging this gives the equation of the separating hyperplane:



(𝐰i−𝐰j)T𝐱+(bi−bj)=0(\mathbf{w}_i - \mathbf{w}_j)^T \mathbf{x} + (b_i - b_j) = 0







The overall effect is that the`OvR`classifier partitions
the feature space intoKKdecision regions, separated by these piecewise linear boundaries.



Let's train this baseline`OvR`model using`Logistic Regression`as the base estimator.

Code:python```
`print("\n--- Training Baseline Model ---")# Initialize the base estimator# Using 'liblinear' solver as it's good for smaller datasets and handles OvR well.# C=1.0 is the default inverse regularization strength.base_estimator=LogisticRegression(random_state=SEED,C=1.0,solver="liblinear")# Initialize the OneVsRestClassifier wrapper using the base estimatorbaseline_model_3c=OneVsRestClassifier(base_estimator)# Train the OvR model on the clean training databaseline_model_3c.fit(X_train_3c,y_train_3c)print("Baseline OvR model trained successfully.")# Predict on the clean test set to evaluate baseline performancey_pred_baseline_3c=baseline_model_3c.predict(X_test_3c)# Calculate baseline accuracybaseline_accuracy_3c=accuracy_score(y_test_3c,y_pred_baseline_3c)print(f"Baseline 3-Class Model Accuracy on Test Set:{baseline_accuracy_3c:.4f}")# Prepare meshgrid for plotting decision boundaries# We create a grid of points covering the feature spaceh=0.02# Step size in the meshx_min,x_max=X_train_3c[:,0].min()-1,X_train_3c[:,0].max()+1y_min,y_max=X_train_3c[:,1].min()-1,X_train_3c[:,1].max()+1xx_3c,yy_3c=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))# Combine xx and yy into pairs of coordinates for predictionmesh_points_3c=np.c_[xx_3c.ravel(),yy_3c.ravel()]# Predict classes for each point on the meshgrid using the trained baseline modelZ_baseline_3c=baseline_model_3c.predict(mesh_points_3c)# Reshape the predictions back into the grid shape for contour plottingZ_baseline_3c=Z_baseline_3c.reshape(xx_3c.shape)print("Meshgrid predictions generated for baseline model.")# Extract baseline model parameters (weights w_k and intercepts b_k)# The fitted OvR classifier stores its individual binary estimators in the `estimators_` attributetry:if(hasattr(baseline_model_3c,"estimators_")andlen(baseline_model_3c.estimators_)==3):estimators_base=baseline_model_3c.estimators_# For binary LogisticRegression with liblinear, coef_ is shape (1, n_features) and intercept_ is (1,)# We extract them for each of the 3 binary classifiers (0 vs Rest, 1 vs Rest, 2 vs Rest)w0_base=estimators_base[0].coef_[0]# Weight vector for class 0 vs Restb0_base=estimators_base[0].intercept_[0]# Intercept for class 0 vs Restw1_base=estimators_base[1].coef_[0]# Weight vector for class 1 vs Restb1_base=estimators_base[1].intercept_[0]# Intercept for class 1 vs Restw2_base=estimators_base[2].coef_[0]# Weight vector for class 2 vs Restb2_base=estimators_base[2].intercept_[0]# Intercept for class 2 vs Restprint("Baseline model parameters (w0, b0, w1, b1, w2, b2) extracted successfully.")else:# This might happen if the model didn't fit correctly or classes were droppedraiseRuntimeError("Could not extract expected number of estimators from baseline OvR model.")exceptExceptionase:print(f"Error: Failed to extract baseline parameters:{e}")`
```

Now we define a function to visualize these multi-class decision boundaries and plot the baseline result.

Code:python```
`defplot_decision_boundary_multi(X,y,Z_mesh,xx_mesh,yy_mesh,title="Decision Boundary",highlight_indices=None,highlight_markers=None,highlight_colors=None,highlight_labels=None,):"""
    Plots the decision boundary regions and data points for a multi-class classifier.
    Automatically ensures points marked with 'P' are plotted above other points.
    Explicit boundary lines are masked to only show in relevant background regions.

    Args:
        X (np.ndarray): Feature data for scatter plot (n_samples, 2).
        y (np.ndarray): Labels for scatter plot (n_samples,).
        Z_mesh (np.ndarray): Predicted classes on the meshgrid (shape matching xx_mesh).
        xx_mesh (np.ndarray): Meshgrid x-coordinates.
        yy_mesh (np.ndarray): Meshgrid y-coordinates.
        title (str): Plot title.
        highlight_indices (list | np.ndarray, optional): Indices of points in X to highlight.
        highlight_markers (list, optional): Markers for highlighted points.
                                          Points with marker 'P' will be plotted on top.
        highlight_colors (list, optional): Edge colors for highlighted points.
        highlight_labels (list, optional): Labels for highlighted points legend.
        boundary_lines (dict, optional): Dict specifying boundary lines to plot, e.g.,
            {'label': {'coeffs': (w_diff_x, w_diff_y), 'intercept': b_diff, 'color': 'color', 'style': 'linestyle'}}
    """plt.figure(figsize=(12,7))# Consistent figure size# Define base class colors and slightly transparent ones for contour fillclass_colors=[azure,nugget_yellow,malware_red]# Extend if more classes as needed# Add fallback colors if needed based on y and Z_meshunique_classes_y=np.unique(y)max_class_idx_y=np.max(unique_classes_y)iflen(unique_classes_y)>0else-1unique_classes_z=np.unique(Z_mesh)max_class_idx_z=np.max(unique_classes_z)iflen(unique_classes_z)>0else-1max_class_idx=int(max(max_class_idx_y,max_class_idx_z))# Ensure integer typeifmax_class_idx>=len(class_colors):print(f"Warning: More classes ({max_class_idx+1}) than defined colors ({len(class_colors)}). Using fallback grey.")# Ensure enough colors exist for indexing up to max_class_idxneeded_colors=max_class_idx+1current_colors=len(class_colors)ifcurrent_colors<needed_colors:class_colors.extend([hacker_grey]*(needed_colors-current_colors))# Appending '60' provides approx 37% alpha in hex RGBA for contour map# Ensure colors used for cmap match the number of classes exactlylight_colors=[c+"60"iflen(c)==7andc.startswith("#")elsecforcinclass_colors[:max_class_idx+1]]cmap_light=plt.cm.colors.ListedColormap(light_colors)# Plot the decision boundary contour fillplt.contourf(xx_mesh,yy_mesh,Z_mesh,cmap=cmap_light,alpha=0.6,zorder=0,# Ensure contour is lowest layer)# Plot the data points# Ensure cmap for points matches number of classes in ycmap_bold=(plt.cm.colors.ListedColormap(class_colors[:int(max_class_idx_y)+1])ifmax_class_idx_y>=0elseplt.cm.colors.ListedColormap(class_colors[:1]))plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_bold,edgecolors=node_black,s=50,alpha=0.8,zorder=1,# Points above contour)# Plot highlighted points if anyhighlight_handles=[]ifhighlight_indicesisnotNoneandlen(highlight_indices)>0:num_highlights=len(highlight_indices)# Provide defaults if None_highlight_markers=(highlight_markersifhighlight_markersisnotNoneelse["o"]*num_highlights)_highlight_colors=(highlight_colorsifhighlight_colorsisnotNoneelse[vivid_purple]*num_highlights)_highlight_labels=(highlight_labelsifhighlight_labelsisnotNoneelse[""]*num_highlights)fori,idxinenumerate(highlight_indices):# Check index validity gracefullyifnot(0<=idx<X.shape[0]):print(f"Warning: Invalid highlight index{idx}skipped.")continue# Determine marker, edge color, and label for this pointmarker=_highlight_markers[i%len(_highlight_markers)]# Get the markeredge_color=_highlight_colors[i%len(_highlight_colors)]label=_highlight_labels[i%len(_highlight_labels)]# Determine face color based on the point's true class from ytry:# Ensure point_class is a valid integer index for class_colorspoint_class=int(y[idx])ifnot(0<=point_class<len(class_colors)):raiseIndexError
                face_color=class_colors[point_class]except(IndexError,ValueError,TypeError):print(f"Warning: Class index '{y[idx]}' invalid for highlighted point{idx}. Using fallback.")face_color=hacker_grey# Fallbackcurrent_zorder=(3ifmarker=="P"else2)# If marker is 'P', use zorder 3, else 2# Plot the highlighted pointplt.scatter(X[idx,0],X[idx,1],facecolors=face_color,edgecolors=edge_color,marker=marker,# Use the determined markers=180,linewidths=2,alpha=1.0,# Make highlighted points fully opaquezorder=current_zorder,# Use the zorder determined by the marker)# Create legend handle if label existsiflabel:# Use Line2D for better control over legend marker appearancehighlight_handles.append(plt.Line2D([0],[0],marker=marker,color="w",label=label,markerfacecolor=face_color,markeredgecolor=edge_color,markersize=10,linestyle="None",markeredgewidth=1.5,))plt.title(title,fontsize=16,color=htb_green)plt.xlabel("Feature 1 (Standardized)",fontsize=12)plt.ylabel("Feature 2 (Standardized)",fontsize=12)# Create class legend handles (based on unique classes in y)class_handles=[]# Check if y is not empty before finding unique classesify.size>0:unique_classes_present_y=sorted(np.unique(y))forclass_idxinunique_classes_present_y:try:int_class_idx=int(class_idx)# Check if index is valid for the potentially extended class_colorsif0<=int_class_idx<len(class_colors):class_handles.append(plt.Line2D([0],[0],marker="o",color="w",label=f"Class{int_class_idx}",markersize=10,markerfacecolor=class_colors[int_class_idx],markeredgecolor=node_black,linestyle="None",))else:print(f"Warning: Cannot create class legend entry for class{int_class_idx}, color index out of bounds after potential extension.")except(ValueError,TypeError):print(f"Warning: Cannot create class legend entry for non-integer class{class_idx}.")else:print(f"Info: No data points (y is empty), skipping class legend entries.")# Combine legendsall_handles=class_handles+highlight_handlesifall_handles:# Only show legend if there's something to legendplt.legend(handles=all_handles,title="Classes, Points & Boundaries")plt.grid(True,color=hacker_grey,linestyle="--",linewidth=0.5,alpha=0.3)# Ensure plot limits strictly match the meshgrid range used for contourfplt.xlim(xx_mesh.min(),xx_mesh.max())plt.ylim(yy_mesh.min(),yy_mesh.max())plt.show()# Plot the decision boundary for the baseline model using the pre-calculated Z_baseline_3cprint("\n--- Visualizing Baseline Model Decision Boundaries ---")plot_decision_boundary_multi(X_train_3c,# Training data pointsy_train_3c,# Training labelsZ_baseline_3c,# Meshgrid predictions from baseline modelxx_3c,# Meshgrid x coordinatesyy_3c,# Meshgrid y coordinatestitle=f"Baseline Model Decision Boundaries (3 Classes)\nTest Accuracy:{baseline_accuracy_3c:.4f}",)`
```

The plot below shows the decision regions learned by the baseline model. Each colored region represents the area of the feature space where the model would predict the corresponding class (`Azure`for Class 0,`Yellow`for Class 1,`Red`for Class 2). The lines where the colors meet are the effective decision boundaries.

![Scatter plot titled 'Baseline Model Decision Boundaries (3 Classes)' with test accuracy 0.9600, showing Class 0 (blue), Class 1 (orange), and Class 2 (red) across standardized Feature 1 and Feature 2.](https://academy.hackthebox.com/storage/modules/302/feature_clean_boundary.png)