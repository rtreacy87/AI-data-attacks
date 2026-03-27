# Evaluating the Targeted Label Attack

---

With the new model trained, we can next evaluate its performance. We do this by evaluating the poisoned model on the clean test set to assess how the attack has degraded its accuracy.

Code:python```
`# Predict on the original, clean test sety_pred_targeted=targeted_poisoned_model.predict(X_test)# Calculate accuracy on the clean test settargeted_accuracy=accuracy_score(y_test,y_pred_targeted)print(f"\n--- Evaluating Targeted Poisoned Model ---")print(f"Accuracy on clean test set:{targeted_accuracy:.4f}")print(f"Baseline accuracy was:{baseline_accuracy:.4f}")# Display classification reportprint("\nClassification Report on Clean Test Set:")print(classification_report(y_test,y_pred_targeted,target_names=["Class 0","Class 1"]))# Plot confusion matrixcm_targeted=confusion_matrix(y_test,y_pred_targeted)plt.figure(figsize=(6,5))sns.heatmap(cm_targeted,annot=True,fmt="d",cmap="binary",xticklabels=["Predicted 0","Predicted 1"],yticklabels=["Actual 0","Actual 1"],cbar=False,)plt.xlabel("Predicted Label",color=white)plt.ylabel("True Label",color=white)plt.title("Confusion Matrix (Targeted Poisoned Model)",fontsize=14,color=htb_green)plt.xticks(color=hacker_grey)plt.yticks(color=hacker_grey)plt.show()`
```

Which will output this and the confusion matrix:

Code:python```
`---Evaluating Targeted Poisoned Model---Accuracy on clean testset:0.8100Baseline accuracy was:0.9933Classification Report on Clean Test Set:precision    recall  f1-score   support

     Class00.731.000.84153Class11.000.610.76147accuracy0.81300macro avg0.860.810.80300weighted avg0.860.810.80300`
```

![Confusion Matrix titled 'Targeted Poisoned Model' showing True Label vs. Predicted Label: 153 True Positives, 0 False Positives, 57 False Negatives, 90 True Negatives.](https://academy.hackthebox.com/storage/modules/302/target_label_confusionmatrix.png)

The attack dropped the model's accuracy from the baseline`0.9933`to`0.8100`. The classification report shows the specific impact:`Class 1`recall fell sharply to`0.61`, meaning the poisoned model correctly identified only 61% of true`Class 1`instances. Correspondingly, the`confusion matrix`shows`57``False Negatives`(Actual`Class 1`predicted as`Class 0`). This confirms the attack successfully degraded the model's performance specifically for the intended target class.

We can plot the boundary of the`targeted_poisoned_model`compared to the`baseline_model`to clearly see how the boundary has shifted with the attack.

Code:python```
`# Plot the comparison of decision boundariesplt.figure(figsize=(12,8))# Plot Baseline Decision Boundary (Solid Green)Z_baseline=baseline_model.predict(mesh_points).reshape(xx.shape)plt.contour(xx,yy,Z_baseline,levels=[0.5],colors=[htb_green],linestyles=["solid"],linewidths=[2.5],)# Plot Targeted Poisoned Decision Boundary (Dashed Red)Z_targeted=targeted_poisoned_model.predict(mesh_points).reshape(xx.shape)plt.contour(xx,yy,Z_targeted,levels=[0.5],colors=[malware_red],linestyles=["dashed"],linewidths=[2.5],)plt.title("Comparison: Baseline vs. Targeted Poisoned Decision Boundary",fontsize=16,color=htb_green,)plt.xlabel("Feature 1",fontsize=12)plt.ylabel("Feature 2",fontsize=12)# Create legend combining data points and boundarieshandles=[plt.Line2D([0],[0],color=htb_green,lw=2.5,linestyle="solid",label=f"Baseline Boundary (Acc:{baseline_accuracy:.3f})",),plt.Line2D([0],[0],color=malware_red,lw=2.5,linestyle="dashed",label=f"Targeted Poisoned Boundary (Acc:{targeted_accuracy:.3f})",),]plt.legend(handles=handles,title="Boundaries & Data Points")plt.grid(True,color=hacker_grey,linestyle="--",linewidth=0.5,alpha=0.3)plt.xlim(xx.min(),xx.max())plt.ylim(yy.min(),yy.max())plt.show()`
```

Which will generate this image, showing the shift in the boundary:

![Plot titled 'Baseline vs. Poisoned Decision Boundary' showing Baseline Boundary (accuracy 0.993) and Targeted Poisoned Boundary (accuracy 0.810) across Feature 1 and Feature 2.](https://academy.hackthebox.com/storage/modules/302/targeted_flip_baseline_vs_poisoned_boundary.png)

The plot vividly illustrates the effect of the attack. The`targeted poisoned boundary`has significantly shifted away from the boundary of the baseline model. The model, forced to accommodate the flipped Class 1 points (now labeled as Class 0), has learned a boundary that is much more likely to classify genuine Class 1 instances as Class 0.

The true test of the attack is how the poisoned model performs on new, previously unseen data. Let's generate a fresh batch of data points using similar distribution parameters (`cluster_std=1.50`is a little bigger for a bit of a data spread) as our original dataset but with a different random seed to ensure they are distinct. We will then use our`targeted_poisoned_model`to classify these points and see how many instances of the target class (Class 1) are misclassified, and display the boundary line.

Code:python```
`# Define parameters for unseen data generationn_unseen_samples=500unseen_seed=SEED+1337# Generate unseen dataX_unseen,y_unseen=make_blobs(n_samples=n_unseen_samples,centers=centers,n_features=2,cluster_std=1.50,random_state=unseen_seed,)# Predict labels for the unseen data using the targeted poisoned modely_pred_unseen_poisoned=targeted_poisoned_model.predict(X_unseen)# Calculate misclassification statisticstrue_target_class_indices=np.where(y_unseen==target_class_to_flip)[0]misclassified_target_mask=(y_unseen==target_class_to_flip)&(y_pred_unseen_poisoned!=target_class_to_flip)misclassified_target_indices=np.where(misclassified_target_mask)[0]n_true_target=len(true_target_class_indices)n_misclassified_target=len(misclassified_target_indices)plt.figure(figsize=(12,8))# Plot all unseen points, colored by the poisoned model's predictionplt.scatter(X_unseen[:,0],X_unseen[:,1],c=y_pred_unseen_poisoned,cmap=plt.cm.colors.ListedColormap([azure,nugget_yellow]),edgecolors=node_black,s=50,alpha=0.7,label="Predicted Label",)# Highlight the misclassified target pointsifn_misclassified_target>0:plt.scatter(X_unseen[misclassified_target_indices,0],X_unseen[misclassified_target_indices,1],facecolors="none",edgecolors=malware_red,linewidths=1.5,marker="X",s=120,label=f"Misclassified (True Class{target_class_to_flip})",)# Calculate and plot decision boundaryZ_targeted_boundary=targeted_poisoned_model.predict(mesh_points).reshape(xx.shape)plt.contour(xx,yy,Z_targeted_boundary,levels=[0.5],colors=[malware_red],linestyles=["dashed"],linewidths=[2.5],)# Set titleplt.title(f"Poisoned Model Predictions & Boundary on Unseen Data\n({n_misclassified_target}of{n_true_target}Class{target_class_to_flip}samples misclassified)",fontsize=16,color=htb_green,)plt.xlabel("Feature 1",fontsize=12)plt.ylabel("Feature 2",fontsize=12)# Create legendhandles=[plt.Line2D([0],[0],marker="o",color="w",label=f"Predicted as Class 0 (Azure)",markersize=10,markerfacecolor=azure,linestyle="None",),plt.Line2D([0],[0],marker="o",color="w",label=f"Predicted as Class 1 (Yellow)",markersize=10,markerfacecolor=nugget_yellow,linestyle="None",),*([plt.Line2D([0],[0],marker="X",color="w",label=f"Misclassified (True Class{target_class_to_flip})",markersize=12,markeredgecolor=malware_red,markerfacecolor="none",linestyle="None",)]ifn_misclassified_target>0else[]),plt.Line2D([0],[0],color=malware_red,lw=2.5,linestyle="dashed",label="Decision Boundary (Targeted Model)",),]plt.legend(handles=handles,title="Predictions, Errors & Boundary")plt.grid(True,color=hacker_grey,linestyle="--",linewidth=0.5,alpha=0.3)# Set plot limitsplt.xlim(xx.min(),xx.max())plt.ylim(yy.min(),yy.max())# Apply theme to backgroundfig=plt.gcf()fig.set_facecolor(node_black)ax=plt.gca()ax.set_facecolor(node_black)plt.show()`
```

This visualization shows the`targeted_poisoned_model`'s predictions and its decision boundary applied to the unseen data.

![Scatter plot titled 'Poisoned Model Predictions & Boundary on Unseen Data' showing Predicted Class 0 (blue), Predicted Class 1 (yellow), Misclassified (red X), and Decision Boundary (dashed line) across Feature 1 and Feature 2.](https://academy.hackthebox.com/storage/modules/302/targeted_flip_unseen_predictions.png)

The points marked with a red 'X' represent true`Class 1`instances that the poisoned model incorrectly predicts as`Class 0`. These misclassifications primarily occur within the actual`Class 1`cluster but fall on the`Class 0`side of the shifted decision boundary (dashed red line).

This clearly demonstrates how the boundary shift induced by the targeted attack successfully causes the intended misclassifications on new, unseen data.