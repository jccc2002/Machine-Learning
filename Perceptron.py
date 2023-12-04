import pandas as pd
from sklearn.linear_model import Perceptron

df = pd.read_csv("Social_Network_Ads.csv",sep=",")

#print(df)#,"\n")
#print(df.describe())
    
df = df.drop(columns = 'User ID')

df['Male'] = (df['Gender'] == 'Male').astype(int) 
df['Female'] = (df['Gender'] == 'Female').astype(int) 

df = df.drop(columns = 'Gender')
columns_order = list(df.columns.difference(['Male', 'Female','Purchased'])) + ['Male', 'Female']  + ['Purchased']
df = df[columns_order]
#print(df.head(5))

xtrain = df.iloc[:319, 0:4]
ytrain = df.iloc[:319, 4]
xtest = df.iloc[320:, 0:4]
ytest = df.iloc[320:, 4]

#print(xtrain.head(1))
#print(ytrain.head(1))
#print(xtest.head(4))
#print(ytest.head(4))


#X, y = df(return_X_y=True)

clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(xtrain, ytrain)
Perceptron()
print(clf.score(xtrain, ytrain))
print(clf.score(xtest, ytest))

