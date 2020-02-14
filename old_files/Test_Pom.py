#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 21:58:36 2019

@author: adam
"""
import pandas as pd
import pomegranate as pm
from matplotlib import pyplot as plt


# Load Data Tables
train = pd.read_csv(
    'hw2_train.data', 
    header=None, 
    sep="\t"
)

test = pd.read_csv(
    'hw2_test.data', 
    header=None, 
    sep="\t"
)


# Convert Names from integers to characters
for d in [train, test]:
    d.columns = ["G"+str(g) for g in list(d.columns)]
    
# Learn The Network from data with the greedy algorithm
bn = pm.BayesianNetwork.from_samples(train)

# Predict Outcomes for Gene 6
G6 = test['G6']
test['G6'] = None
pred=pd.DataFrame(bn.predict(test.values))
pred.columns = ["P"+str(g) for g in list(pred.columns)]
pred = pred.assign(G6 = G6)[['G6', 'P6']]
pred['Match'] = pred[['G6', 'P6']].apply(
    lambda x:
       1 if  x['P6'] == x['G6'] else 0,
     axis = 1
    )
bn.plot()
plt.show()
print("Correct Predictions:", pred.Match.sum())



# Learn The Network from data, with the chow-liu algorithm
bn = pm.BayesianNetwork.from_samples(train, algorithm='chow-liu')

# Predict Outcomes for Gene 6

pred=pd.DataFrame(bn.predict(test.values))
pred.columns = ["P"+str(g) for g in list(pred.columns)]
pred = pred.assign(G6 = G6)[['G6', 'P6']]
pred['Match'] = pred[['G6', 'P6']].apply(
    lambda x:
       1 if  x['P6'] == x['G6'] else 0,
     axis = 1
    )
bn.plot()
plt.show()
print("Correct Predictions:", pred.Match.sum())

# Learn The Network from data, with the exact algorithm
bn = pm.BayesianNetwork.from_samples(train, algorithm='exact')

# Predict Outcomes for Gene 6
pred=pd.DataFrame(bn.predict(test.values))
pred.columns = ["P"+str(g) for g in list(pred.columns)]
pred = pred.assign(G6 = G6)[['G6', 'P6']]
pred['Match'] = pred[['G6', 'P6']].apply(
    lambda x:
       1 if  x['P6'] == x['G6'] else 0,
     axis = 1
    )
bn.plot()
plt.show()
print("Correct Predictions:", pred.Match.sum())












