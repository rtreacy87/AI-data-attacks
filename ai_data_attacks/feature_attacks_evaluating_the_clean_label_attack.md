# Evaluating the Clean Label Attack

---

Now we train a new model using this`poisoned training dataset`(`X_train_poisoned`,`y_train_poisoned`). We use the same model architecture (`OneVsRestClassifier`with`Logistic Regression`) and hyperparameters as the baseline model to ensure a fair comparison.

Code:python```
`print("\n--- Training Poisoned Model (Clean Label Attack) ---")# Initialize a new base estimator for the poisoned model (same settings as baseline)poisoned_base_estimator=LogisticRegression(random_state=SEED,C=1.0,solver="liblinear")# Initialize the OneVsRestClassifier wrapperpoisoned_model_cl=OneVsRestClassifier(poisoned_base_estimator)# Train the model on the POISONED training datapoisoned_model_cl.fit(X_train_poisoned,y_train_poisoned)print("Poisoned model (Clean Label) trained successfully.")`
```



With the poisoned model trained, we now evaluate its effectiveness.
Remember the primary goal was to misclassify the`specific target point`𝐱target=373\mathbf{x}_{target}=373.
To evaluate this, we check the poisoned model’s prediction for this
instance. We also assess the model’s overall performance on the
original, clean`test set`(`X_test_3c`,`y_test_3c`) to see if the attack caused any broader
degradation.



Code:python```
`print("\n--- Evaluating Poisoned Model Performance ---")# Check the prediction for the specific target pointX_target_reshaped=X_target.reshape(1,-1)# Reshape for single predictiontarget_pred_poisoned=poisoned_model_cl.predict(X_target_reshaped)[0]print(f"Target Point Evaluation:")print(f"  Original True Label (y_target):{y_target}")print(f"  Baseline Model Prediction:{target_baseline_pred}")print(f"  Poisoned Model Prediction:{target_pred_poisoned}")attack_successful=(target_pred_poisoned!=y_target)and(target_pred_poisoned==0)# Specifically check if flipped to Class 0ifattack_successful:print(f"  Success: The poisoned model misclassified the target point as Class{target_pred_poisoned}.")else:iftarget_pred_poisoned==y_target:print(f"  Failure: The poisoned model still correctly classified the target point as Class{target_pred_poisoned}.")else:print(f"  Partial/Unexpected: The poisoned model misclassified the target point, but as Class{target_pred_poisoned}, not the intended Class 0.")# Evaluate overall accuracy on the clean test sety_pred_poisoned_test=poisoned_model_cl.predict(X_test_3c)poisoned_accuracy_test=accuracy_score(y_test_3c,y_pred_poisoned_test)print(f"\nOverall Performance on Clean Test Set:")print(f"  Baseline Accuracy:{baseline_accuracy_3c:.4f}")print(f"  Poisoned Accuracy:{poisoned_accuracy_test:.4f}")print(f"  Accuracy Drop:{baseline_accuracy_3c-poisoned_accuracy_test:.4f}")# Display classification report for more detailprint("\nClassification Report (Poisoned Model on Clean Test Data):")print(classification_report(y_test_3c,y_pred_poisoned_test,target_names=["Class 0","Class 1","Class 2"]))`
```

Based on the above codes output, which is below, we can see that the attack was indeed effective. The target point is being misclassified despite no labels having been changed.

Code:python```
`---Evaluating Poisoned Model Performance---Target Point Evaluation:OriginalTrueLabel(y_target):1Baseline Model Prediction:1Poisoned Model Prediction:0Success:The poisoned model misclassified the target pointasClass0.Overall Performance on Clean Test Set:Baseline Accuracy:0.9600Poisoned Accuracy:0.9578Accuracy Drop:0.0022Classification Report(Poisoned Model on Clean Test Data):precision    recall  f1-score   support

     Class00.980.990.98150Class10.940.930.94150Class20.950.950.95150accuracy0.96450macro avg0.960.960.96450weighted avg0.960.960.96450`
```

We also observe a slight drop in overall accuracy on the clean test set compared to the baseline. This is common in clean label attacks; while targeted, the boundary warping caused by the perturbed points can sometimes lead to collateral damage, affecting the classification of other nearby points.

The final step is to visualize the`impact of the attack on the decision boundaries`.

Code:python```
`print("\n--- Visualizing Poisoned Model Decision Boundaries vs. Baseline ---")# Predict classes on the meshgrid using the POISONED modelZ_poisoned_cl=poisoned_model_cl.predict(mesh_points_3c)Z_poisoned_cl=Z_poisoned_cl.reshape(xx_3c.shape)# Plot the decision boundary comparisonplot_decision_boundary_multi(X_train_poisoned,# Show points from the poisoned training sety_train_poisoned,# Use their labels (perturbed are still 0)Z_poisoned_cl,# Use the poisoned model's mesh predictions for backgroundxx_3c,yy_3c,title=f"Poisoned vs. Baseline Decision Boundaries\nTarget Misclassified:{attack_successful}| Poisoned Acc:{poisoned_accuracy_test:.4f}",highlight_indices=[target_point_index_absolute]+perturbed_indices_list,highlight_markers=["P"]+["o"]*len(perturbed_indices_list),highlight_colors=[white]+[vivid_purple]*len(perturbed_indices_list),highlight_labels=[f"Target (Pred:{target_pred_poisoned})"]+[f"Perturbed (Idx{idx})"foridxinperturbed_indices_list],)`
```

This final visualization below demonstrates the success of our attack. As we can see, the`target point`('+', originally Class 1) now lies on the`Class 0`side of the poisoned model's decision boundary.

![Scatter plot titled 'Poisoned vs. Baseline Decision Boundaries' with poisoned accuracy 0.9578, showing Class 0 (blue), Class 1 (orange), Class 2 (red), Target Point (white cross), and Perturbed Points (purple) across standardized Feature 1 and Feature 2.](https://academy.hackthebox.com/storage/modules/302/feature_attack_final.png)

By subtly`modifying the features of a few data points`while keeping their labels technically "correct" (at least plausible for the modified features), we were able to manipulate the model's learned decision boundary in a targeted manner, causing specific misclassifications during inference without leaving obvious traces like flipped labels. Detecting such attacks can be significantly more challenging than detecting simple label flipping, but such an attack is also vastly more complicated to execute.