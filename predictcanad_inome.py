import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


data=pd.read_csv('canada_per_capita_income.csv')


plt.xlabel("Year")
plt.ylabel("Income")
plt.scatter(data.year, data.income,color="red")
plt.show()
value=linear_model.LinearRegression()
value.fit(data[['year']],data['income'])
predicted_income=value.predict([[2017]])
print(predicted_income)
plt.xlabel("Year")
plt.ylabel("Income")
plt.scatter(data.year, data.income,color="red")
plt.plot(data.year,value.predict(data[['year']]),color='blue')
plt.show()