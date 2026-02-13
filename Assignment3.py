#!/usr/bin/env python
# coding: utf-8

# In[36]:


#Q1
import pandas as pd

data = pd.read_csv("crime1.csv")

mean = data["ViolentCrimesPerPop"].mean()
median= data["ViolentCrimesPerPop"].median()
maximum= data["ViolentCrimesPerPop"].max()
minimum= data["ViolentCrimesPerPop"].min()
std_dev= data["ViolentCrimesPerPop"].std()
print(f"The mean is {mean}")
print(f"The median is {median}") # the distribution seems relitively symmetrical. I believe this because the range of values is from 0.02 to 1 and the difference between the mean and median is 0.05.
# therefore the distribution may be slightly skewed but it should be reletively symmetrical.

#b) if there are outliers the mean is affected more because it is the average of the values while the median is the middle most value.


# In[31]:


#Q2a
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("crime1.csv")
values=[]
for i in data.index:   # loop over rows
    value=data.loc[i, "ViolentCrimesPerPop"] 
    values.append(value)
x=values
plt.hist(x, bins = 10, edgecolor =
"white")
plt.xlabel("Violent Crimes")
plt.ylabel("Number of communities")
plt.title("The relatiosnhip between the proportion of violent crimes and number of comunities")
plt.show()

# the histogram shows that the data is right skewed  as a lot of the data is stacked up closer to 0. With fewer communities having higher proportions of violent crimes.


# In[38]:


#Q2B
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("crime1.csv")
values=[]
for i in data.index:   # loop over rows
    value=data.loc[i, "ViolentCrimesPerPop"] 
    values.append(value)
x=values
plt.boxplot(x)
plt.show
plt.ylabel("Proportion of crimes that are violent Crimes")
plt.title("The relatiosnhip between the proportion of violent crimes and number of comunities")
plt.show()
# the box plot shows that the median is closer to quartile 1 then quartile 3. This shows that the data is right skewed.
# The data also shows there are no outliers because if there were any outliers they would appear on the box plot as singular dots passed whiskers of the plot..


# In[37]:


#Q3
from sklearn.model_selection import train_test_split
import pandas as pd

kidney_data = pd.read_csv('kidney_disease.csv')

feature_matrix = kidney_data.loc[:, kidney_data.columns != 'classification']
target_labels = kidney_data.loc[:, ['classification']]

features_train, features_test, labels_train, labels_test = train_test_split(
    feature_matrix,
    target_labels,
    test_size=0.30,
    random_state=42
)

#If we use the same dataset for training and testing, the model can overfit, meaning it performs well on the training data but fails to predict new data accurately.
#The purpose of the testing set is to check how the model performs on new data.


# In[26]:


#Q4
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load dataset
kidney_data = pd.read_csv('kidney_disease.csv')

# Fill missing numeric values with mean
for col in kidney_data.select_dtypes(include=['int64', 'float64']).columns:
    kidney_data[col] = kidney_data[col].fillna(kidney_data[col].mean())

# Fill missing categorical values with mode
for col in kidney_data.select_dtypes(include=['object']).columns:
    kidney_data[col] = kidney_data[col].fillna(kidney_data[col].mode()[0])

# Encode categorical features as numbers (skip target)
for col in kidney_data.select_dtypes(include=['object']).columns:
    if col != 'classification':
        kidney_data[col] = kidney_data[col].astype('category').cat.codes

# Encode target: 'ckd' = 1, 'notckd' = 0
y = [1 if val.lower() == 'ckd' else 0 for val in kidney_data['classification']]

# Features
X = kidney_data.drop(columns=['classification'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train KNN with k=5
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Predict
y_pred = knn_model.predict(X_test)

# Compute metrics
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display results
print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)




# True positive :  predicted positive and tested positive
# True Negative: predicted Negative and  tested negative
# False positive: predicted that the test would positive but tested negative
# False Negative: predicted that the test would be negative but tested positive.
# for diseases more than accuracy is needed this is because a lot of the time the amount of people with a disease is significantly less than those without so the accuracy could be really high but they may wrongly say someone with the disease doesnt have it a lot of the time and it wouldnt majorly affect the accuracy.
# the most important metric would be recall because this would tell us out of how many people who have it test positive this helps to tell us if we are properly diagnosing people.


# In[39]:


#Q5
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
kidney_data = pd.read_csv('kidney_disease.csv')

# Fill missing numeric values with mean
for col in kidney_data.select_dtypes(include=['int64', 'float64']).columns:
    kidney_data[col] = kidney_data[col].fillna(kidney_data[col].mean())

# Fill missing categorical values with mode
for col in kidney_data.select_dtypes(include=['object']).columns:
    kidney_data[col] = kidney_data[col].fillna(kidney_data[col].mode()[0])

# Encode categorical features as numbers (skip target)
for col in kidney_data.select_dtypes(include=['object']).columns:
    if col != 'classification':
        kidney_data[col] = kidney_data[col].astype('category').cat.codes

# Encode target: ckd = 1, notckd = 0
y = [1 if val.lower() == 'ckd' else 0 for val in kidney_data['classification']]

# Features
X = kidney_data.drop(columns=['classification'])

# Split data`
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train KNN for multiple k values and store accuracy
k_values = [1, 3, 5, 7, 9]
accuracies = []

for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    y_predict = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# Create results table
results = pd.DataFrame({'k': k_values, 'Test Accuracy': accuracies})
print("K vs Test Accuracy:\n", results)

# Identify best k
best_index = accuracies.index(max(accuracies))
print(f"\nHighest test accuracy is at k = {k_values[best_index]} with accuracy = {accuracies[best_index]:.2f}")

# How changing k affects the behavior of the model:
# Small k values  make the model sensitive to individual training points, 
# Larger k values average out predictions because more data points are considered.  it reduces the effect of individual points but it may ignore local patterns leading to underfitting.
#   This reduces the impact of individual noisy points, but may ignore local patterns, leading to underfitting.
#  Medium k values often are the better option as they tend to balance between underfitting and over fitting

# Why very small values of k may cause overfitting:
# The model memorizes the training data, so even noise or outliers can strongly influence predictions.

# Why very large values of k may cause underfitting:
# The model averages over too many neighbors and may ignore finer patterns in the data.


# In[ ]:




