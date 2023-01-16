import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

#Reading the data with pandas
data = pd.read_csv("student-mat.csv", sep=";")

#Removing the irrelevant data
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]


#obtaining the input by removing the output layer
predict = "G3"
x = np.array(data.drop([predict], 1))

#obtaining the output layer
        
y = np.array(data[predict])

#splitting the data into training data and into test data

x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.1)



linear=linear_model.linearRegression()

#training the model

linear.fit(x_train,y_train)


#to test the accuracy of the model

accuracy=linear.score(x_test,y_test)

print("accuracy: \n",accuracy)

#getting the coefficients from te linear regression

print("linear coefficient: \n ", linear.coef_)

print("intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

#Testing the model with the test data.
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

