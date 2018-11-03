
# coding: utf-8

# ## CISC 6930: Data Mining 
# Assignmnet 3
# <br>
# Angela Krontiris
# <br>
# Program used: Python
# <br>
# November 2, 2018

# In[2]:


import pandas as pd
import numpy as np
from scipy.io import arff
get_ipython().run_line_magic('matplotlib', 'inline')

# Import 'veh-prime.arff' file
data = arff.loadarff('veh-prime.arff')

# Convert arff file to DataFrame
df = pd.DataFrame(data[0])

# Convert 'CLASS' column to 0="noncar" and 1="car"
df['CLASS'] = np.where(df['CLASS'] == b'noncar', 0, 1)

print("This a df of all features and class labels:\n" )
df.head()


# In[3]:


#1) Recreating list of feature names with a list comprehension
features = [ 'f'+ str(i) for i in range(0,36) ] # list of strings with all features
print("List of feature names:\n", features)

#2) Extracting features and class as numpy arrays

#a) Extracting features from df
x_mat = df[features].values
print("\nThese are the value of features in a numpy array:\n", x_mat)

#b) Extracting class labels as numpy array
y_mat = df['CLASS'].values
print("\nThese are the target labels in a numpy array:\n", y_mat)


# ### Pearson Correlation Coefficient

# In[4]:


# Creating funciton to Compute the Pearson Correlation Coefficient

def pearsonr(x,y):
    """This Function will calculate the Pearson Correlation Coefficient between each feature and the class label."""

    # Calculate the numerator (covariance)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    cov_x_y = np.sum((x - mean_x) * (y - mean_y))

    # Calculate the denominator (standard deviation)
    stdv_x = np.sum(np.square(x - mean_x))
    stdv_y = np.sum(np.square(y - mean_y))

    correlation = cov_x_y / np.sqrt(stdv_x * stdv_y)
                          
    return correlation


# ### List the features from highest |r| (the absolute value of r) to lowest, along with their |r| values.

# In[5]:


#1) Calculating the correlation between each feature and the class label and storing results in a list
r = [pearsonr(df[feature].values, y_mat) for feature in features] #features = list of feature names
# print(type(r))
# print(r)
# print(len(r))

#2) Convert list of pearson correlations to a numpy array
r_arr = np.array(r)

#3) Take the absolute value of all pearson correlatio values in r_arr
r_abs = np.absolute(r_arr)
print("Absolute Values of Correlation Coefficients, r:\n\n", r_abs)
# print(type(r_abs))


# ### Sorting Absolute Value of Correlations (Features)

# In[6]:


# Combine feature list with correlation list
# feature_corr = np.column_stack((list(r_abs), features))

#1) Create DataFrame with correlation (r) values for each feature
feat_corr = pd.DataFrame({"correlation_r": r_abs})
display(feat_corr.head())

#2) Labelling index
feat_corr.index.name = 'feature'
print("Correlation of Features DataFrame:\n", feat_corr.head())

#3) Sorting Correlation values of features in dataframe
#a) sorting
feat_corr_sort = feat_corr.sort_values(by = 'correlation_r', axis =0, ascending = False)
print("\nSorted Correlation of Features DataFrame:\n", feat_corr_sort.head())
print("\nTesting how to extract value for dataframe:\n", feat_corr_sort.iloc[0])


#b) converting sorted df of correlations to numpy array

feat_corr_sort_values_numpy = feat_corr_sort["correlation_r"].values
print("\nSorted values of correlations as numpy array:\n\n", feat_corr_sort_values_numpy)


# In[7]:


# Loop through first n rows
# feat_list = [for i]

#1) extracting ranked corrleations indexes 
feature_indices_sorted_list= list(feat_corr_sort.index)
print( "This is a list of the sorted indices:\n", feature_indices_sorted_list, "\n")

# #Making lists of columns to extract ****Come back need to turn into a list
# print("\n", "Testing how to extract successive indices from ranked correlation:\n")
# for index in range(2,len(feature_indices_sorted_list)+1):
#     print(feature_indices_sorted_list[0:index])

#2)Ranked Feature Combos -- Making a list comprehension to hold columns to extract from df -- based on correlation ranking
feature_selection_combos = [feature_indices_sorted_list[0:index] for index in range(2,len(feature_indices_sorted_list)+1)]
print("List of 'm' highest feature combinations:\n", feature_selection_combos)


# In[8]:


#1) Create a loop to store dataframe combos is a list  
combo_dfs = [df.iloc[:, combo] for combo in feature_selection_combos] #here have a list of arrays for each feature combo
# print(combo_dfs)
print(combo_dfs[7])

#a)Trying to get accuracies for single combo
combo_single = combo_dfs[0]
print('Combination 1 test:')
display(combo_single.head())

#b) Breaking up data into feature matrix and class label matrix

y = df['CLASS'].values


# ### Function to Normalize Testing & Training Data

# In[9]:


#1)  Creating function to scale testing and training data
# This will enable us to easily scale data for each go around in LOOCV

def train_test_z_normalization(X_train, X_test):
    
    mu = X_train.mean(axis=0) #mean of training set
    sigma = X_train.std(axis=0) #standard deviation for training set
    
    
    z_score_train = (X_train - mu)/sigma #training set normalized
    z_score_test = (X_test - mu)/sigma #test set normalized

    return(z_score_train, z_score_test)


# ### Function to Run KNN Classifier &  Make Predictions with Test Set

# In[10]:


def KNN_classifier(X_test, y_train, y_test, z_score_train, z_score_test, combo):

    # Set KNN Parameter, K
    K = [7] 

    pred = {}
    for i in range(X_test.shape[0]):
        # We only need the sum of squares in order to rank by distance, 
        # since sqrt() is a monotonic transformation
        # 
        # "test_mat[i,:] - train_mat" does a *broadcast* operation 
        # (https://docs.scipy.org/doc/numpy-1.15.0/user/basics.broadcasting.html)
        # This is significantly faster than iterating over the taining set,
        # because Numpy arithmetic in C is vectorized and is extremely fast,
        # whereas looping over the rows of the matrix in Python is much slower
        #
        # sum_sq ends up being a np.array() of the same size as the training set
        #
        sum_sq = np.sum(np.square(z_score_test[i,:] - z_score_train), axis=1)

        # This is how we match the distance with the training labels.  At this
        # point, we know they're in the same order, because train_mat, train, and sum_sq
        # are all in the same order
        #
        df = pd.DataFrame({'sum_sq':sum_sq,'class':y_train}) ## are values getting overwritten?
#         display(df.head(7))




        # Sort the DataFrame by the distance, then take the 'class' column
        sorted_class = df.sort_values('sum_sq')['class'] #just takes labels or class column of smallest squared sum
#         print("\nSorted df, extracting class:\n")
#         display(sorted_class.head(7))
#         print(type(sorted_class))

#         print(sorted_class[:7])#still a series

        # Predict by determining whether mean of training class is > 5
        pred[i] = [sorted_class[:k].mean() > 0.5 for k in K]

#     print("\nThis is the prediction dictionary:\n", pred)

    # #accessing element from prediction dictionary
    # print(pred[0])


    # Now pred{} is a map from the row index of the test set to the predictions. 
    # We can match it with the labels from the test set by doing a 
    # pd.concat(..., axis=1), which will match on the Index.  
    # "pd.DataFrame(pred, index=K)" gives us a DataFrame with 1 row per K 
    # and 1 column per member of the test set.  We cast it to int (since it's a 
    # boolean after doing > 0.5), and then take the transpose (".T") so that now 
    # it's a matrix with 1 row per member of the test set.

#     print("Xtest is a: ", type(test))

    #Here we are concatenating or joining two dataframes columnwise (adding in columns)
    test_with_pred = pd.concat([pd.DataFrame(X_test, columns = list(combo.columns) ), pd.DataFrame(pred, index=K).astype(int).T], axis=1)
#     print(test_with_pred)

    # Accuracy is the % of the time the predicted and actual labels match
    accuracy = pd.Series({k:(y_test == test_with_pred.loc[:,k]).mean() for k in K})
#     print("\nTake a look at the accuracy:\n\n", accuracy)
    
    return(accuracy[7])

    
    
# #Extracting accuracy for k =7 and appending to master list
# accuracy_master = []
# accuracy_master.append(accuracy[7])
# print(accuracy_master)


# In[11]:



# test_with_pred = pd.concat([pd.DataFrame(X_test), pd.DataFrame(pred, index=K).astype(int).T, pd.DataFrame({"Label": y_test})], axis=1)
# accuracy = pd.Series({k:(test_with_pred.Label == test_with_pred.loc[:,k]).mean() for k in K})




# ### LOOCV - Train and Test Split
# 
# Leave one data point out for validation

# In[12]:


from sklearn.model_selection import LeaveOneOut

master_combo_dict = {}

for combo_index, combo in enumerate(combo_dfs):
    
    print(type(combo))
    X = combo.values
    # Create variable name for LeaveOneOut for later use
    loocv = LeaveOneOut()

    # Returns number of samples from training matrix, X
    loocv.get_n_splits(X)#converting combo dataframe to numpy array


    #1) Running LOOCv for all feature combos
    
    master_accuracies_avg = []
    master_accuracies = []

    for train_index, test_index in loocv.split(X):

        #1) Getting unique split, using one row for testing
    #     print("Train Index Split:\n", train_index, "\n\nTest Index Split:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_mat[train_index], y_mat[test_index]
    #     print("Xtrain:\n", X_train)
    #     print("Xtest:\n", X_test)
    #     print("ytrain:\n", y_train)
    #     print("ytest:\n", y_test)

        #2) Applying function to normalize training and testing dat
        z_score_train, z_score_test = train_test_z_normalization(X_train, X_test)
    #     print(z_score_train)
    #     print(z_score_test)

        #3) Need to apply KNN and make prediction on testing data
        accuracy_loocv_individual = KNN_classifier(X_test, y_train, y_test, z_score_train, z_score_test, combo)
    #     print(accuracy_loocv_individual)

        master_accuracies.append(accuracy_loocv_individual)
        # print(master_accuracies)
    master_accuracies_avg.append(np.mean(master_accuracies))
        
    master_combo_dict[combo_index] = master_accuracies_avg


print(master_combo_dict)    


# In[13]:


### Sorting calculated accuracies generated by KNN and LOOCV 


# In[26]:


#1) Putting master combo dictionary into a series so can easily sort accuracies
master_combo_series = pd.Series(master_combo_dict)
master_combo_series_sorted = master_combo_series.sort_values(ascending=False) 

print("This is the sorted series to find best combo (highest accuracy generated by ranked features by correlation):\n")
display(master_combo_series_sorted)

#2) Printing summary for filter method
#a) selecting best feature combo index along with accuracy
best_feature_combo_index = master_combo_series_sorted.index[0]
highest_accuracy = master_combo_series_sorted[best_feature_combo_index][0]#extracting accuracy from list

#b) Printing out the list of feature combinations yielding highest accuracy
top_features = feature_selection_combos[best_feature_combo_index]
print("\nThese are the top features using filter method:\n", top_features)

print('\nThese top {} correlated features yield an accuracy of {}'.format(len(top_features),highest_accuracy))


