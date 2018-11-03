
# coding: utf-8

# ## CISC 6930: Data Mining -- Wrapper Method 
# Assignmnet 3
# <br>
# Angela Krontiris
# <br>
# Program used: Python
# <br>
# November 2, 2018

# In[1]:


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


# In[2]:


#1) Recreating list of feature names with a list comprehension
features = [ 'f'+ str(i) for i in range(0,36) ]# list of strings with all features
print("List of feature names:\n", features)

######Testing how to create list of list and add item to sublist######
# Creating a list in a list
features_list = [[i] for i in features]
print("\nList of list feature names:\n", features_list)

# Append single element to each of the list of features
f20 = 'f20'

for feature_list in features_list:
    feature_list.append(f20)
print("\nList of features with added feature:\n", features_list)   
########################################################################


# # #______________________________________
# #2) Extracting features and class as numpy arrays

# #a) Extracting features from df
# x_mat = df[features].values
# print("\nThese are the value of features in a numpy array:\n", x_mat)

#b) Extracting class labels as numpy array
y_mat = df['CLASS'].values
print("\nThese are the target labels in a numpy array:\n", y_mat)


# ### Fuction to Remove Feature Duplicates In List  for Wrapper Method

# In[3]:


def feature_dup_list_remover(features_list):
    """Passing in a list of lists, we will pop any number of sublists if it contains duplicate features
        (i.e., we would pop [f12,f12]). All sublists must be of the same size"""
    
    pop_index_list = []
    for index, feature_list in enumerate(features_list):
        
        #a) find the length of any sublist
        sublist_length = len(feature_list) # will all be the same size for our case

        #b) Convert each sublist to a set (to reduce to unique elements; then back to list)
        feature_list_convert_to_set = list(set(feature_list))

        #c) check if sublist is < than actual sublist_length

        if len(feature_list_convert_to_set) < sublist_length :
            pop_index = index #finding index of duplicate
            pop_index_list.append(pop_index)
        
#     #d) now that have indices of sublists stored in a list, we can loop and pop each element
    popped_elements = []
    
    for index_pop in sorted(pop_index_list, reverse=True): # **popping elements from max index first so dont screw up original order
        popped_element = features_list.pop(index_pop)
        popped_elements.append(popped_element)
        
#     print("Indexes to pop:", pop_index_list)
    return features_list # Final list returned without popped elements


# In[4]:


# Testing function-- we know func should pop ['f20', 'f20']. Does it work?
print(features_list)
feature_list_no_dup = feature_dup_list_remover(features_list)
print(feature_list_no_dup)



# ### Function to Normalize Testing & Training Data

# In[5]:


#1)  Creating function to scale testing and training data
# This will enable us to easily scale data for each go around in LOOCV

def train_test_z_normalization(X_train, X_test):
    
    mu = X_train.mean(axis=0) #mean of training set
    sigma = X_train.std(axis=0) #standard deviation for training set
    
    
    z_score_train = (X_train - mu)/sigma #training set normalized
    z_score_test = (X_test - mu)/sigma #test set normalized

    return(z_score_train, z_score_test)


# ### Function to Run KNN Classifier &  Make Predictions with Test Set

# In[6]:


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
#     test_with_pred = pd.concat([pd.DataFrame(X_test, columns = list(combo.columns) ), pd.DataFrame(pred, index=K).astype(int).T], axis=1)
    test_with_pred = pd.concat([pd.DataFrame(X_test, columns = list(combo.columns) ), pd.DataFrame(pred, index=K).astype(int).T], axis=1)
#     print(test_with_pred)

    # Accuracy is the % of the time the predicted and actual labels match
    accuracy = pd.Series({k:(y_test == test_with_pred.loc[:,k]).mean() for k in K})
#     print("\nTake a look at the accuracy:\n\n", accuracy)
    
    return(accuracy[7])


# ### LOOCV - Train and Test Split
# 
# #### Wrapper Method - Forward Sequential Search

# In[7]:


#1) Printing list of features
print("Here is a list of features:\n", features)

#2) Extracting individual columns into dataframe
individual_feature_dfs_list = [df.loc[:, [feature]] for feature in features]
# print("\nList of Individual feature columns stored in a dataframe:\n", individual_feature_dfs_list)
print([individual_feature_dfs_list[0]])


# In[8]:


#3) Extracting single feature values to run thru LOOCV
individual_feature = individual_feature_dfs_list[0]
print(type(individual_feature))
print("X: Indvidual df of feature 0(for test purposes):\n", individual_feature.head())

#making consistent so can use LOOCV for loop
X = individual_feature

print("\ny_matrix:\n", type(y_mat),"\n\n", y_mat )


# In[64]:


from sklearn.model_selection import LeaveOneOut

def KNN_LOOCV(individual_feature_dfs_list, y_mat=y_mat):
    """Runs KNN with LOOCV and returns a list of features with highest accuracy, as well as the highest accuracy"""
    
    master_combo_dict = {}

    for combo_index, combo in enumerate(individual_feature_dfs_list):  

#         print(type(combo))
        X = combo.values

        # Create variable name for LeaveOneOut for later use
        loocv = LeaveOneOut()

        # Returns number of samples from training matrix, X
        loocv.get_n_splits(X)

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
        master_accuracies_avg.append(np.mean(master_accuracies))
        master_combo_dict[combo_index] = master_accuracies_avg
    #     print(master_accuracies)
    #     print(np.mean(master_accuracies))
    print(master_combo_dict)
    
#     print("Highest accuracy:", max(master_accuracies_avg))
    
    
    feature_with_max_accuracy =  list(individual_feature_dfs_list[max(master_combo_dict, key=master_combo_dict.get)].columns) #returns index for max accuracy, then pulls out acutal sublist corresponding to index    
    max_accuracy_for_feature = master_combo_dict[max(master_combo_dict, key=master_combo_dict.get)] 
    
    return feature_with_max_accuracy, max_accuracy_for_feature 


# ### Function to extract feature columns from dataframe

# In[18]:


def feature_extraction_from_dataframe(features_list, df = df):
    """Given a list of  feature combination lists, it will loop thru each feature sublist and
    extract corresponding columns from dataframe into a dataframe """
    
    individual_feature_dfs_list = [df.loc[:, feature] for feature in features_list]
    
    return individual_feature_dfs_list
    
#function is working


# In[79]:


#1) Recreating list of feature names with a list comprehension
features = [ 'f'+ str(i) for i in range(0,36) ]# list of strings with all features
# print("List of feature names:\n", features)

#1a) Creating list of lists for featrues, so can easily append to in future
features_list = [[i] for i in features]
# print("\nList of list feature names:\n", features_list)

#2) Extracting individual columns into dataframe
individual_feature_dfs_list = feature_extraction_from_dataframe(features_list, df = df)
# print([individual_feature_dfs_list[0]]) #just checking single dataframe from list

###____________________________________________________________________________________
#####Get ready to Loop!#######

    
master_features_list_to_df_list = [individual_feature_dfs_list] #intialize; this is a list of all dfs than need to be feed 



master_KNN_LOOCV_top_accuracy = [] #storing all accuracies for each run
master_KNN_LOOCV_top_feature = [] #storing top features

run = 0
number_features = len(list(df.columns))-1

for individual_feature_dfs_list in master_features_list_to_df_list:

    if run < 10: #only want to run 
    
        #3) Pass this list of dfs to KNN_LOOCV function
        KNN_LOOCV_top_feature, KNN_LOOCV_top_accuracy = KNN_LOOCV(individual_feature_dfs_list) #retruns feature with highest accuracy 
        print("\nTop Feature List:", KNN_LOOCV_top_feature)
        print("Top Feature Accuracy:",KNN_LOOCV_top_accuracy )

        #4) Append top feature to 'features_list' (think sublists)
        for feature_list in features_list:
            feature_list.append(KNN_LOOCV_top_feature[0]) #appending top feature
        
#         print(features_list)
#         print(type(features_list))
        
        #5) Apply 'feature_dup_list_remover' to remove sublists with duplicate features
        features_list = feature_dup_list_remover(features_list)
        print("\nNew feature list with duplicates removed and top feature added:\n", features_list)

        #6) Now using 'features_list' to extract columns from dataframe
        feature_to_df_list = feature_extraction_from_dataframe(features_list, df = df)
#         print("\nfeature to df list:\n", feature_to_df_list)


        master_KNN_LOOCV_top_accuracy.append(KNN_LOOCV_top_accuracy)
        master_KNN_LOOCV_top_feature.append(KNN_LOOCV_top_feature)

        master_features_list_to_df_list.append(feature_to_df_list)
    
    else:
        break
    run = run + 1

print("Accuracies for each run:\n", master_KNN_LOOCV_top_accuracy)
print("Top feature selected:\n",master_KNN_LOOCV_top_feature)





# In[92]:


import matplotlib.pyplot as plt
print("\nAccuracies for each run:\n", master_KNN_LOOCV_top_accuracy)

print("\nTop feature selected:\n",master_KNN_LOOCV_top_feature)

#1) Converting accuracy list into numpyarray so can easily flatten into single arrayt
master_KNN_LOOCV_top_accuracy_np = np.array(master_KNN_LOOCV_top_accuracy).flatten()
print("\nThis is the flattend numpy array of accuracies:\n", master_KNN_LOOCV_top_accuracy_np)

#2) Plotting accuracies for each run of adding a feature
#a) Converting numpy array to a series to easily plot
master_KNN_LOOCV_top_accuracy_np_series = pd.Series(master_KNN_LOOCV_top_accuracy_np)

#b) Plotting series of accuracies
master_KNN_LOOCV_top_accuracy_np_series.plot()

plt.title("KNN LOOCV Wrapper Method with Forward Selection")
plt.ylabel("Accuracy")
plt.xlabel("Iteration")

plt.show()

#3) Summarizing KNN LOOCV Wrapper Method with Forward Selection

#a) Finding highest accuracy in numpy array
max_KNN_LOOCV_Wrapper_fwd_method = np.max(master_KNN_LOOCV_top_accuracy_np)
print("\nThe maximum accuracy is attained for the KNN LOOCV Wrapper Method with Forward Selection is:",max_KNN_LOOCV_Wrapper_fwd_method )

#b) Finding position of that maximum value
max_index_KNN_LOOCV_Wrapper_fwd_method = np.argmax(master_KNN_LOOCV_top_accuracy_np)
print("\nThe maximum accuracy occurs on the {} iteration".format(max_index_KNN_LOOCV_Wrapper_fwd_method))

#c) Selecting the feature combination that yielded this highest accuracy
feature_combo_highest_accuracy = master_KNN_LOOCV_top_feature[max_index_KNN_LOOCV_Wrapper_fwd_method]
print("\nThe feature combination that yielded the highest accuracy is:\n", feature_combo_highest_accuracy )


# #### Filter Method Results:
# 
# These are the top features using filter method:
#  [4, 13, 14, 16, 7, 22, 26, 1, 20, 31, 34, 2, 28, 25, 19, 17, 32, 8, 0, 10]
# 
# These top 20 correlated features yield an accuracy of 0.925531914893617
