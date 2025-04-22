import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

file=pd.read_csv("hiring.csv")

learn=linear_model.LinearRegression()
learn.fit(file[["experience","test_score","interview_score"]],file.salary)
predictor=learn.predict([[4,6,7]])
print(predictor)


