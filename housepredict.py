import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

#we  using pandasto read our csv file 
# Load the dataset
df=pd.read_csv("houseprice.csv")
#print(df)

# now lets use matplotlib to create a graph and see our data in 
plt.xlabel("Area sq_ft")
plt.ylabel("Price in US$")
plt.scatter(df.area, df.price,color="red", )
plt.show() 


# now we create our linea regression
# Create a linear regression model
#first we create an object of linear regression
reg=linear_model.LinearRegression()
#now we fit our data into the model
reg.fit(df[['area']],df['price'])
#now we can see our model and its coefficents
predicted_price=reg.predict([[2700 ]])
print(f"THE PRIDICTED HOUSE PRICE IS {predicted_price}")
#so how this work is it calculates Y=mx+b or price =area*coefficent+b or price =m *area + b
# lets find  m
()
m=reg.coef_
print(f"{m} this the value of m in y =mx+b")
# lets find b
b=reg.intercept_
print(f"{b} this the value of b in y =mx+b")
plt.xlabel("Area sq_ft")
plt.ylabel("Price in US$")
plt.scatter(df.area, df.price,color="red", )
plt.plot(df.area,reg.predict(df[['area']]),color="blue")

plt.show()

d=pd.read_csv("areas.csv")
print(d.head(3))
h=reg.predict(d)
print(h)
d['price']=h
d.to_csv("predicted.csv",index=False)