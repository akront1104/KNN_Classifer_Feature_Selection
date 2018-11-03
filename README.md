# KNN_Classifer_Feature_Selection
Using Filter and Wrapper Method to select best features 

### Filter Method:

Make the class labels numeric (set “noncar”=0 and “car”=1) and calculate the Pearson
Correlation Coefficient (PCC) of each feature with the numeric class label. The PCC
value is commonly referred to as r. For a simple method to calculate the PCC that
is both computationally efficient and numerically stable.

### Wrapper Method:

Starting with the empty set of features, use a greedy approach to add the single feature
that improves performance by the largest amount when added to the feature set. This
is Sequential Forward Selection. Define performance as the LOOCV classification
accuracy of the KNN classifier using only the features in the selection set (including
the ?candidate? feature). Stop adding features only when there is no candidate that
when added to the selection set increases the LOOCV accuracy.
