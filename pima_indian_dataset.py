

import pandas as pd #data frame library
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression



df=pd.read_csv('diabetes.csv') #load Pima data. Adjust path as necesary
df.shape

df.info()

df.isnull().sum()

df.head()   #to show the beginning row of our data set

# Find missing values
print('Missing values:{}'.format(df.isnull().any().sum()))
# Find duplicated records
print('\nNumber of duplicated records: {}'.format(df.duplicated().sum()))

import pandas as pd

df=pd.read_csv('diabetes.csv')  #load Pima data. Adjust path as necesary
df.head()

# Find missing values
print('Missing values:{}'.format(df.isnull().any().sum()))
# Find missing values
print('Missing values:{}'.format(df.isnull().any().sum()))

df.head(5)

from sklearn.model_selection import train_test_split


X=df.drop('Outcome',axis=1)
Y=df['Outcome']

# Normalization  to update data in range(0,1)
scaler=MinMaxScaler()
scaled_data=scaler.fit_transform(X)
print(scaled_data)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(scaled_data,Y,test_size=0.2,random_state=0)

"""###Model



"""

model=LogisticRegression()
# Train model
model.fit(x_train,y_train)

print(model.score(x_test,y_test))

print(model.score(x_test,y_test))

data = model.predict([[6,148,72,35,0,33.6,0.627,50]])
for i in range (1):
  if(data[i]==0):
    print("Non-diabetic")
  else:
    print("Diabetic")
print(data)