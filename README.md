# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries 
2. Create a dataset and print it using the pandas library
3. Split the dataset as x and y and also split it for training and testing using train_test_split() function
4. Create and train the model
5. Make predictions and also plot the graph for the output 

## Program:
```

# Program to implement the simple linear regression model for predicting the marks scored.
# Developed by: ANTONY ASWIN KUMAR L
# RegisterNumber:  212225040024

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#Create the dataset
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Marks_Scored':  [35, 40, 50, 55, 60, 65, 70, 75, 80, 85]
}

df = pd.DataFrame(data)
print("Data:")
print(df)

#Split into X and Y
X = df[['Hours_Studied']]  
y = df['Marks_Scored']   


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)

print("\nModel Evaluation: ")
print("Slope (m): ", model.coef_[0])
print("Intercept (c): ", model.intercept_)
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
print("R² Score: ", r2_score(y_test, y_pred))
hours = float(input("Enter number of study hours: "))
predicted_marks = model.predict([[hours]])
print(f"Predicted Marks for studying {hours} hours = {predicted_marks[0]:.2f}")

plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.title('Simple Linear Regression: Hours vs Marks')
plt.legend()
plt.show()

```

## Output:

<img width="1245" height="535" alt="Screenshot 2026-05-14 100758" src="https://github.com/user-attachments/assets/41a7c40b-2ef7-494e-ad91-18da76d3ae88" />


<img width="1257" height="571" alt="Screenshot 2026-05-14 100823" src="https://github.com/user-attachments/assets/87cfd5b8-2d53-4c12-ba0f-a18781934bec" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming is executed successsfully.
