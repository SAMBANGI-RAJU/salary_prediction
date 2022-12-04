import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
df=pd.read_csv('Salary_data.csv')
#df.head()
#df.isnull().sum()
#df.shape
#X=df[['YearsExperience']].values
X=df.iloc[:,-2].values
y=df.iloc[:,-1].values
from sklearn import linear_model
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=150)
reg=linear_model.LinearRegression()
#print(X_train)
#X_train
X_train = X_train.reshape(-1, 1)
y_train=y_train.reshape(-1,1 )
reg.fit(X_train,y_train)
X_test=X_test.reshape(-1,1)
y_pred=reg.predict(X_test)
#print(y_pred)
#print(y_test)
reg.score(X_test,y_test)#here is the accuracy of my model
with open('salary.pkl','wb') as f:
    pickle.dump(reg,f)
reg_model = pickle.load(open('salary.pkl','rb'))
print(reg_model.predict([[1.3]]))