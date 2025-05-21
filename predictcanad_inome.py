import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pickle

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

# now lets save our model named value as file to use it anywhere later
with open ('model_pickle','wb')as f:
    pickle.dump(value,f)
# the above code has created a file in ourworking directory

#now lets use it or read it
with open ('model_pickle','rb')as f:
    mp=pickle.load(f)
from_file=mp.predict([[2017]])

print(from_file)
