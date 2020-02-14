#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 13:56:03 2019

@author: adam
"""
from network import net
from network import sample_net
from pomegranate import BayesianNetwork as BN
from matplotlib import pyplot as plt
import random as rn
import pandas as pd

####### Randomly Generate Some dependent, Multinomial Data

def dpf1(x, p):
    if x == 1:
        return rn.choices([0,1], weights=[p, 1-p])[0]
    else:
        return rn.choices([0,1], weights=[1-p, p])[0]
    
def dpf2(x, p, q):
    if x == 1:
        return rn.choices([0, 1, 2], weights=[p, q, 1-(p+q)])[0]
    
    elif x == 0:
        return rn.choices([0, 1, 2], weights=[1-(p+q), q, p])[0]
    
    else:
        return rn.choices([0, 1, 2], weights=[p, 1-(p+q), q])[0]
    
def dpf3(x, y, p, q):
    if x == 1 and y == 1:
        return rn.choices([0,1], weights=[p,1-p])[0]
    elif x !=1 and y == 1:
        return rn.choices([0,1], weights=[1-p, p])[0]
    elif x == 1 and y !=1:
        return rn.choices([0,1], weights=[q,1-q])[0]
    else:
        return rn.choices([0,1], weights=[1-q, q])[0]
    
    
    
k = 1000    
G1 = rn.choices([0,1], weights=[0.5, 0.5], k=k)
G4 = [dpf2(x, 0.1, 0.1) for x in G1]
G2 = [dpf1(x, 0.9) for x in G4]
G3 = [dpf1(x, 0.1) for x in G2]
G5 = [dpf3(G1[i], G2[i], 0, 0.5) for i in range(0,k)]
G6 = rn.choices([0,1], weights=[0.7, 0.3], k=k)
G7 = [dpf1(x, 0.1) for x in G6]
G8 = [dpf2(x, 0.2, 0.1) for x in G6]
G9 = rn.choices([0,1,2], weights =[6/9, 2/9, 1/9], k=k)
G10 = rn.choices([0,1,2], weights=[1/3, 1/3, 1/3], k=k)

data = pd.DataFrame({
    'G1':G1,
    'G2':G2,
    'G3':G3,
    'G4':G4,
    'G5':G5,
    'G6':G6,
    'G7':G7,
    'G8':G8,
    'G9':G9,
    'G10':G10
})

bnet = BN.from_samples(data)
bnet.plot()
plt.show()

n = net(data=data)
n.add_edge(0, 3)
n.add_edge(0, 4)
n.add_edge(1, 4)
n.add_edge(1, 3)
n.add_edge(1, 2)
n.add_edge(5, 6)
n.add_edge(5, 7)
n.calc_cpt(data)
samp = sample_net(n, 1000)

bnet = BN.from_samples(samp)
bnet.plot()
plt.show()