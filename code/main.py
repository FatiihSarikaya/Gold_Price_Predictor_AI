import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

data = pd.read_csv('../data/gld_price_data.csv')

#Trying to get info about data before Procces it

print(data.head())
print(data.tail())
print(data.shape)
print(data.info())
print(data.columns)

#Checking the number of missing values
print(data.isnull().sum())


#Getting the statistical measures of the data

print(data.describe())

#Correlation

correlation = data.drop('Date',axis=1).corr()
plt.figure(figsize=(8,8))
sns.heatmap(correlation, cbar=True, square=True,fmt='.1f',  annot=True, annot_kws={'size':8}, cmap='Blues')
plt.show()

#Correlation values if GLD

print(correlation['GLD'])

#Checking the distribution of the GLD price

sns.distplot(data['GLD'],color='green')
plt.show()


X = data.drop(['Date','GLD'],axis=1)
y = data['GLD']


#Splitting into Training data and Test Data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=2)

#Model Training

regressor = RandomForestRegressor(n_estimators= 100)

#Training the Model

regressor.fit(X_train, y_train)

#Model Eveluation

test_data_prediction = regressor.predict(X_test)

#R square error

error_score = metrics.r2_score(y_test, test_data_prediction)
print("R squared error: ", error_score)

#Compare the Actual Values and Predicted Values in a Plot
y_test = list(y_test)

plt.plot(y_test, color='blue',label ='Actual Value')
plt.plot(test_data_prediction, color='green', label='Predicted  Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of Values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()



