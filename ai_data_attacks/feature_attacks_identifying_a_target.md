# Identifying a Target

---



Now that we have a baseline, we can proceed with the actual`Clean Label Attack`. Our specific goal is to modify the
training data such that a chosen`target point`,𝐱target\mathbf{x}_{target},
which originally belongs to`Class 1`(Yellow), will be
misclassified by the retrained model as belonging to`Class 0`(Blue).







We aim to choose a point that genuinely belongs to`Class 1`(its true labelytarget=1y_{target} = 1)
but also lies relatively close to the decision boundary separating`Class 1`from`Class 0`, as determined by the
original baseline model. Points near the boundary are inherently more
vulnerable to misclassification if the boundary shifts, even slightly,
after retraining on the poisoned data.





To identify such a point, we can analyze the decision function scores
produced by the baseline model. Remember that the decision boundary
between`Class 0`and`Class 1`is where their
respective scores,z0z_0andz1z_1,
are equal. We can define a function representing the difference between
these scores:





f01(𝐱)=z0−z1=(𝐰0−𝐰1)T𝐱+(b0−b1)f_{01}(\mathbf{x}) = z_0 - z_1 = (\mathbf{w}_0 - \mathbf{w}_1)^T \mathbf{x} + (b_0 - b_1)





The baseline model predicts`Class 1`for a point𝐱\mathbf{x}if its scorez1z_1is greater than the scores for all other classeskk.
Specifically considering`Class 0`and`Class 1`,
the model favors`Class 1`ifz1>z0z_1 > z_0.
This condition is equivalent to the score differencef01(𝐱)f_{01}(\mathbf{x})being negative
(f01(𝐱)<0f_{01}(\mathbf{x}) < 0).





Therefore, we are looking for a specific point𝐱target\mathbf{x}_{target}within the training set that meets our criteria:ytarget=1y_{target} = 1,
and its score differencef01(𝐱target)f_{01}(\mathbf{x}_{target})must be negative (confirming the baseline model classifies it correctly
relative to`Class 0`), while also being as close to zero as
possible. A score difference that is the largest negative value
indicates the point is correctly classified but is nearest to thef01(𝐱)=0f_{01}(\mathbf{x})=0boundary.





To find this optimal target point, we calculatef01(𝐱)f_{01}(\mathbf{x})for all training points𝐱i\mathbf{x}_iwhose true labelyiy_iis 1. We then select the specific point𝐱target\mathbf{x}_{target}that yields the largest negative value (i.e., the value closest to zero)
forf01f_{01}.



Code:python```
`print("\n--- Selecting Target Point ---")# We use the baseline parameters w0_base, b0_base, w1_base, b1_base extracted earlier# Calculate the difference vector and intercept for the 0-vs-1 boundaryw_diff_01_base=w0_base-w1_base
b_diff_01_base=b0_base-b1_baseprint(f"Boundary vector (w0-w1):{w_diff_01_base}")print(f"Intercept difference (b0-b1):{b_diff_01_base}")# Identify indices of all Class 1 points in the original clean training setclass1_indices_train=np.where(y_train_3c==1)[0]iflen(class1_indices_train)==0:raiseValueError("CRITICAL: No Class 1 points found in the training data. Cannot select target.")else:print(f"Found{len(class1_indices_train)}Class 1 points in the training set.")# Get the feature vectors for only the Class 1 pointsX_class1_train=X_train_3c[class1_indices_train]# Calculate the decision function f_01(x) = (w0-w1)^T x + (b0-b1) for these Class 1 points# A negative value means the point is on the Class 1 side of the 0-vs-1 boundarydecision_values_01=X_class1_train @ w_diff_01_base+b_diff_01_base# Find indices within the subset of Class 1 points that are correctly classified (f_01 < 0)class1_on_correct_side_indices_relative=np.where(decision_values_01<0)[0]iflen(class1_on_correct_side_indices_relative)==0:# This case is unlikely if the baseline model has decent accuracy, but handle it.print(f"{malware_red}Warning:{white}No Class 1 points found on the expected side (f_01 < 0) of the 0-vs-1 baseline boundary.")print("Selecting the Class 1 point with the minimum absolute decision value instead.")# Find index (relative to class1 subset) with the smallest absolute distance to boundarytarget_point_index_relative=np.argmin(np.abs(decision_values_01))else:# Among the correctly classified points, find the one closest to the boundary# This corresponds to the maximum (least negative) decision valuetarget_point_index_relative=class1_on_correct_side_indices_relative[np.argmax(decision_values_01[class1_on_correct_side_indices_relative])]# Map the relative index (within the class1 subset) back to the absolute index in the original X_train_3c arraytarget_point_index_absolute=class1_indices_train[target_point_index_relative]# Retrieve the target point's features and true labelX_target=X_train_3c[target_point_index_absolute]y_target=y_train_3c[target_point_index_absolute]# Should be 1 based on selection logic# Sanity Check: Verify the chosen point's class and baseline predictiontarget_baseline_pred=baseline_model_3c.predict(X_target.reshape(1,-1))[0]target_decision_value=decision_values_01[target_point_index_relative]print(f"\nSelected Target Point Index (absolute):{target_point_index_absolute}")print(f"Target Point Features:{X_target}")print(f"Target Point True Label (y_target):{y_target}")print(f"Target Point Baseline Prediction:{target_baseline_pred}")print(f"Target Point Baseline 0-vs-1 Decision Value (f_01):{target_decision_value:.4f}")ify_target!=1:print(f"Error: Selected target point does not have label 1! Check logic.")iftarget_baseline_pred!=y_target:print(f"Warning: Baseline model actually misclassifies the chosen target point ({target_baseline_pred}). Attack might trivially succeed or have unexpected effects.")iftarget_decision_value>=0:print(f"Warning: Selected target point has f_01 >= 0 ({target_decision_value:.4f}), meaning it wasn't on the Class 1 side of the 0-vs-1 boundary. Check logic or baseline model.")# Visualize the data highlighting the selected target point near the boundaryprint("\n--- Visualizing Training Data with Target Point ---")plot_data_multi(X_train_3c,y_train_3c,title="Training Data Highlighting the Target Point (Near Boundary)",highlight_indices=[target_point_index_absolute],highlight_markers=["P"],# 'P' for Plus sign marker (Target)highlight_colors=[white],# White edge color for visibilityhighlight_labels=[f"Target (Class{y_target}, Idx{target_point_index_absolute})"],)`
```

The above code identifies a good candidate for us to work with, index`373`:

Code:python```
`---Selecting Target Point---Boundary vector(w0-w1):[-5.787925146.32142485]Intercept difference(b0-b1):-0.9207223376477074Found350Class1pointsinthe trainingset.Selected Target Point Index(absolute):373Target Point Features:[-0.55111155-0.36675028]Target PointTrueLabel(y_target):1Target Point Baseline Prediction:1Target Point Baseline0-vs-1Decision Value(f_01):-0.0493`
```

If we plot index`373`we can easily see where it is in the dataset.

![Scatter plot titled 'Training Data Highlighting the Target Point (Near Boundary)' showing Class 0 (blue), Class 1 (orange), Class 2 (red), and Target Point (white cross) across standardized Feature 1 and Feature 2.](https://academy.hackthebox.com/storage/modules/302/feature_attack_targted_point.png)