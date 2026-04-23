# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"F:\FSDS $ GENAI & Agentic AI\ML-Spyder Code\Student marks Prediction Analysis\Student_info.csv")

df # Show the dataset

df.head()

df.tail()

df.shape

df.info()

df.describe()

plt.scatter(x = df.study_hours, y = df.student_marks)
plt.xlabel("students study Hours")
plt.xlabel("students marks")
plt.xlabel("Scatter Plot of students study Hours vs students marks")
plt.show()

# Data Cleaning

df

df.isnull().sum()

df.mean()

df2 = df.fillna(df.mean())

df2.isnull().sum()

df2.head()

df.head()

df2.info()

df.info()

#Split the Database

x = df2.drop("student_marks",axis = "columns")
y = df2.drop("study_hours",axis = "columns")
print("Shape of X = ",x.shape)
print("Shape of y = ",y.shape)

x

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2,random_state=0)

print("Shape of x_train = ", x_train.shape)
print("Shape of y_train = ", y_train.shape)
print("Shape of x_test = ", x_test.shape)
print("Shape of y_test = ", y_test.shape)

x_train

y_train

# Select a model and train It

# y = m*x + c
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# Fine_Tune your model

lr

lr.fit(x_train,y_train)

lr.get_params()

lr.coef_

lr.intercept_

lr

y_pred = lr.predict(x_test)
y_pred

pd.DataFrame(np.c_[x_test,y_test,y_pred], columns = ["study_hours","student_marks_original","student_marks"])

# Find_Tune Your Model

lr

lr.score(x_test,y_test) # Varience

lr.score(x_train,y_train)  # Bias

plt.scatter(x_train,y_train)

plt.scatter(x_test,y_test)
plt.plot(x_train, lr.predict(x_train), color = "r")

# Save the Model

lr

import joblib # create Pipeline
joblib.dump(lr, "Desktop.pkl")

import os
print(os.getcwd())

model = joblib.load("Desktop.pkl") # Pickle





