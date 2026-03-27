# Baseline Logistic Regression Model

---

Before executing the attack, we need to establish`baseline performance`so we have something to compare the effects of the poisoned model with. We will train a`Logistic Regression`model on the original, clean training data (`X_train`,`y_train`). This baseline represents the model's expected behavior and accuracy under normal,`non-adversarial conditions`.

As outlined in the "[Fundamentals of AI](https://enterprise.hackthebox.com/academy-lab/67497/preview/modules/290)" module,`Logistic Regression`is fundamentally a classification algorithm used for predicting binary outcomes.



For a given review represented by its feature vector𝐱i=(xi1,xi2)\mathbf{x}*i = (x*{i1}, x_{i2}),
the model first calculates a linear combinationziz_iusing weights𝐰=(w1,w2)\mathbf{w} = (w_1, w_2)and a bias termbb:







zi=𝐰T𝐱i+b=w1xi1+w2xi2+bz_i = \mathbf{w}^T \mathbf{x}_i + b = w_1 x_{i1} + w_2 x_{i2} + b





This valueziz_irepresents the`log-odds`(or logit) of the review having
positive sentiment
(yi=1y_i=1).
It quantifies the linear relationship between the derived features and
the log-odds of a positive classification.





zi=log⁡(P(yi=1|𝐱i)1−P(yi=1|𝐱i))z_i = \log\left(\frac{P(y_i=1 | \mathbf{x}_i)}{1 - P(y_i=1 | \mathbf{x}_i)}\right)





To convert the log-oddsziz_iinto a probabilitypi=P(yi=1|𝐱i)p_i = P(y_i=1 | \mathbf{x}_i)(the probability of the review being`positive`), the model
applies the`sigmoid function`,σ\sigma:





pi=σ(zi)=11+e−zi=11+e−(𝐰T𝐱i+b)p_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}} = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x}_i + b)}}





The sigmoid function squashes the outputziz_iinto the range[0,1][0, 1],
representing the model’s estimated probability that the review𝐱i\mathbf{x}_ihas`positive`sentiment.





During training, the model learns the optimal parameters𝐰\mathbf{w}andbbby minimizing a`loss function`over the training set
(`X_train`,`y_train`). The goal is to find
parameters that make the predicted probabilitiespip_ias close as possible to the true sentiment labelsyiy_i.
The standard loss function for binary classification is the`binary cross-entropy`or`log-loss`:





L(𝐰,b)=−1N∑i=1N[yilog(pi)+(1−yi)log(1−pi)]L(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]





Here,NNis the number of training reviews,yiy_iis the true sentiment label (0 or 1) for theii-th
review, andpip_iis the model’s predicted probability of positive sentiment for that
review. Optimization algorithms like`gradient descent`iteratively adjust𝐰\mathbf{w}andbbto minimize this lossLL.





Once trained, the model uses the learned𝐰\mathbf{w}andbbto predict the sentiment of new, unseen reviews. For a new review𝐱\mathbf{x},
it calculates the probabilityp=σ(𝐰T𝐱+b)p = \sigma(\mathbf{w}^T \mathbf{x} + b).
Typically, ifp≥0.5p \ge 0.5,
the review is classified as`positive`(Class 1); otherwise,
it’s classified as`negative`(Class 0).





The`decision boundary`is the line (in our 2D feature
space) where the model is exactly uncertain
(p=0.5p = 0.5),
which occurs whenz=𝐰T𝐱+b=0z = \mathbf{w}^T \mathbf{x} + b = 0.
This linear boundary separates the feature space into regions predicted
as`negative`and`positive`sentiment. The
training process finds the line that best separates the clusters in the
training data.



We now train this baseline model on our clean data and evaluate its accuracy on the unseen test set.

Code:python```
`# Initialize and train the Logistic Regression modelbaseline_model=LogisticRegression(random_state=SEED)baseline_model.fit(X_train,y_train)# Predict on the test sety_pred_baseline=baseline_model.predict(X_test)# Calculate baseline accuracybaseline_accuracy=accuracy_score(y_test,y_pred_baseline)print(f"Baseline Model Accuracy:{baseline_accuracy:.4f}")# Prepare to plot the decision boundarydefplot_decision_boundary(model,X,y,title="Decision Boundary"):"""
    Plots the decision boundary of a trained classifier on a 2D dataset.

    Parameters:
    - model: The trained classifier object (must have a .predict method).
    - X (np.ndarray): Feature data (n_samples, 2).
    - y (np.ndarray): Labels (n_samples,).
    - title (str): The title for the plot.
    """h=0.02# Step size in the meshx_min,x_max=X[:,0].min()-1,X[:,0].max()+1y_min,y_max=X[:,1].min()-1,X[:,1].max()+1xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))# Predict the class for each point in the meshZ=model.predict(np.c_[xx.ravel(),yy.ravel()])Z=Z.reshape(xx.shape)plt.figure(figsize=(12,6))# Plot the decision boundary contourplt.contourf(xx,yy,Z,cmap=plt.cm.colors.ListedColormap([azure,nugget_yellow]),alpha=0.3)# Plot the data pointsscatter=plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.colors.ListedColormap([azure,nugget_yellow]),edgecolors=node_black,s=50,alpha=0.8,)plt.title(title,fontsize=16,color=htb_green)plt.xlabel("Feature 1",fontsize=12)plt.ylabel("Feature 2",fontsize=12)# Create a legend manuallyhandles=[plt.Line2D([0],[0],marker="o",color="w",label="Negative Sentiment (Class 0)",markersize=10,markerfacecolor=azure,),plt.Line2D([0],[0],marker="o",color="w",label="Positive Sentiment (Class 1)",markersize=10,markerfacecolor=nugget_yellow,),]plt.legend(handles=handles,title="Classes")plt.grid(True,color=hacker_grey,linestyle="--",linewidth=0.5,alpha=0.3)plt.xlim(xx.min(),xx.max())plt.ylim(yy.min(),yy.max())plt.show()# Plot the decision boundary for the baseline modelplot_decision_boundary(baseline_model,X_train,y_train,title=f"Baseline Model Decision Boundary\nAccuracy:{baseline_accuracy:.4f}",)`
```

The resulting plot shows the linear decision boundary learned by the baseline model, effectively separating the simulated`Negative`and`Positive`sentiment clusters in the training data. The high accuracy score indicates it generalizes well to the unseen test data.

![Scatter plot titled 'Baseline Model Decision Boundary' with accuracy 0.9933, showing Negative Sentiment (Class 0, blue) and Positive Sentiment (Class 1, orange) across Feature 1 and Feature 2.](https://academy.hackthebox.com/storage/modules/302/label_flipping_baseline.png)