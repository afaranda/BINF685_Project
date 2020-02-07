#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 14:27:03 2020

@author: adam
"""
from pomegranate import *
from pomegranate import BayesianNetwork as bn
I = DiscreteDistribution({0:7/10, 1: 3/10})
H = ConditionalProbabilityTable(
        [[0, 0, 9./10],
         [0, 1, 1./10],
         [1, 0, 2./10],
         [1, 1, 8./10]], [I]
    )


W = ConditionalProbabilityTable(
        [[0, 0, 9./10],
         [0, 1, 1./10],
         [1, 0, 2./10],
         [1, 1, 8./10]], [I]
    )

A = ConditionalProbabilityTable(
    [[0,0,0,9./10],
     [0,0,1,1./10],
     [0,1,0,5./10],
     [0,1,1,5./10],
     [1,0,0,4./10],
     [1,0,1,6./10],
     [1,1,0,1./10],
     [1,1,1,9./10]],[H,W]    
    )


Icy = State(I, name="Icy")
Holmes = State(H, name="Holmes")
Watson = State(W, name="Watson")
Ambulance = State(A, name="Ambulance")


model=bn("IcyRoad")
model.add_states(Icy,Holmes, Watson, Ambulance)

model.add_edge(Icy, Holmes)
model.add_edge(Icy, Watson)
model.add_edge(Holmes, Ambulance)
model.add_edge(Watson, Ambulance)
model.bake()

