# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Load the dataset.

3.Define X and Y array.

4.Define a function for costFunction,cost and gradient.

5.Define a function to plot the decision boundary.

6.Define a function to predict the Regression value.

## Program:
```

Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: VAISHNAVIDEVI V
RegisterNumber: 212223040230

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Placement_Data.csv')
dataset

dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
Y

theta = np.random.randn(X.shape[1])
y =Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h= sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta

theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations = 1000)

def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred

y_pred = predict(theta,X)

accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)

print(y_pred)

print(Y)

xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)

xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew) 

```

## Output:

### READ AND DISPLAY

![image](https://github.com/user-attachments/assets/e8f844eb-3848-4605-92ba-e6af3c09e209)

### CATEGORIZING COLUMNS

![image](https://github.com/user-attachments/assets/a34e5038-8eec-4e59-9faa-2e8381f45fa4)

### LABELING COLUMNS AND DISPLAYING DATASET

![image](https://github.com/user-attachments/assets/35b55f28-4b52-4eb6-8f1a-0bfd5332bc80)

### SEPERATE FEATURE X AND Y

![image](https://github.com/user-attachments/assets/c7f84bae-3728-46e1-a6fb-1706cddb5eb9)

### ACCURACY

![image](https://github.com/user-attachments/assets/b4b0f4e6-3c1a-445e-be6f-4636d0e9ea54)

### Y

![image](https://github.com/user-attachments/assets/7213ffdd-d224-41db-bc2c-74fcfaaf5ee2)


### Y_PREDNEW

![image](https://github.com/user-attachments/assets/fc2ebece-b25f-44c4-b3fd-2b3ec26bdd98)
















## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

