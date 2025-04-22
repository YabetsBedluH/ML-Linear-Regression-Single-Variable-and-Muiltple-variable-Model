import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

info=pd.read_csv("prices.csv")

model=linear_model.LinearRegression()
model.fit(info[['area','bedrooms','age']],info.price)
final_value=model.predict([[3000,3,40]])
#print(final_value)

ndinfo=pd.read_csv("newprices.csv")
newinfo=model.predict(ndinfo)
print(newinfo)