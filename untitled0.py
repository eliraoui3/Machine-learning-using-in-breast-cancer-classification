# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:49:24 2020

@author: AMEL
"""
import pandas as pd
data = pd.read_csv("Breast_cancer_data.csv")
df = pd.DataFrame(data)
y=df["diagnosis"]
x=df.iloc[:,0:5]
print(x[ :5])