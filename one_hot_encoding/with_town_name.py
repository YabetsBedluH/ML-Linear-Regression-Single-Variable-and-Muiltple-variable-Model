import pandas as pd
df = pd.read_csv("C:/Users/yabet/Desktop/ML/one_hot_encoding/homepricess.csv")

#we are changeing the town column into one hot encoding to use it in the model
dummies=pd.get_dummies(df.town,dtype=int)
# now lets merge it using the pandas pd with original data
merged=pd.concat([df,dummies],axis="columns")


#we always must drop a column  If you're using one-hot encoded columns to avoid dummy variable trap

final = merged.drop(['town','west_windsor'],axis='columns')

# know we have prepared our data lets train it
from sklearn.linear_model import LinearRegression
model=LinearRegression()

# now we have to drop our y or dependent variable 
X= final.drop('price_usd',axis='columns')
#we defined y as a price
y=final.price_usd
#training
model.fit(X,y)
#lets predict price in robinsville with a price of 2800
prediction_input = pd.DataFrame([[2800, 0, 1]], columns=X.columns)
print(model.predict(prediction_input))
#if we want to predict the droped area price or the west_windsor we make the encodes 0,0 
prediction_dropped = pd.DataFrame([[2800, 0, 0]], columns=X.columns)
print(model.predict(prediction_dropped))
