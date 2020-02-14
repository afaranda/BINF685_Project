#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 11:56:09 2020

@author: adam
This file implements functions that can be used to evaluate a trained
Learner instance.
"""
#from network import edges
from network import export_pom
from numpy import log
import pandas as pd
import numpy as np
def edge_hits(lrn, Truth=None, directed=False):
    '''

    Parameters
    ----------
    lrn : instance of "greedy, CASGMM" etc. Trained Learner from "Learners.py"

    truth : net that generated the training data 
    
    directed : Boolean whether to score based on directed or undirected edges

    Returns
    -------
    Dictionary with elements: TP, FP, TN, FN
    TP = True positive, # of edges in 'truth' network also in learnerd net
    FP = False positive, # of edges in learnerd net not in 'truth'
    TN = Edges not in truth, or Learned net ()
    FN = Edges in truth not in Learned.
    '''
    Ea=[]
    used = []
    for u in [i.name for i in lrn.states]:
        used.append(u)
        for  v in set([i.name for i in lrn.states]) - set(used):
            Ea.append((u,v))
            
    #Et = edges(Truth)
    Et = [
            (i[0].name, i[1].name) 
            for i in Truth.edges
        ]
    #El = edges(lrn.net)
    El = [
            (i[0].name, i[1].name) 
            for i in lrn.edges
        ]
    
    if not (directed):
                
        score = {'TP':0, 'FP':0, 'TN':0, 'FN':0}
        # for a possible edge
        for e in Ea:
            # If this undirected edge exists in Truth
            if e in Et or (e[1], e[0]) in Et:
                # If this undirected edge was learned:
                if e in El or (e[1], e[0]) in El:
                    score['TP'] += 1
        
                # If this undirected edge was not learned:
                else:
                    score['FN'] += 1
        
            # If this undirected edge does not exist in Truth
            else:
                # If this undirected edge was learned
                if e in El or (e[1], e[0]) in El:
                    score['FP'] += 1
                    
                # If this undirected edge was not learned
                else:
                    score['TN'] += 1
        
        # For edge in learned edge set
            # If this edge is in the network increment TP
            # Otherwise increment FP
        
        # For edge in 
        
    return score


### Calculate Kulbeck Liebler Divergence between two tables
def kldv(p, q):
    idx=[i for i in list(p.columns) if i !='Prob']
    
    cpt1=p.copy()
    cpt1.set_index(idx, inplace=True)
    cpt1.rename(columns={'Prob':'P'}, inplace=True)
    
    
    cpt2=q.copy()
    cpt2.set_index(idx, inplace=True)
    cpt2.rename(columns={'Prob':'Q'}, inplace=True)
    
    kld = pd.concat([cpt1, cpt2], axis=1)
    kld['lpq'] = kld.apply(lambda x: x['P'] * log(x['P'] / x['Q']), axis=1)
    
    return kld['lpq'].sum()


### Functions to calculate Sensetivity, Specificity and Accuracy
def sn(score):
    return score['TP'] / sum([score['TP'], score['FN']])

def sp(score):
    return score['TN'] / sum([score['TN'], score['FP']])

def ac(score):
    return sum([score['TN'], score['TP']]) / sum(score.values())

### Functions to score predictions for binary data
def call_bin(o, p):
    if o == 1:
        if o == p:
            return 'TP'
        else:
            return 'FP'
    else:
        if o == p:
            return 'TN'
        else:
            return 'FN'
            
def pred_bin(net, data, var):
    d = data.copy()
    m = net
    v = [i.name for i in m.states]
    ix = [i for i in range(0, len(v)) if v[i] == var]
    #score = {'TP':0, 'FP':0, 'TN':0, 'FN':0}
    
    d['Pred'] = [
         p[ix] for p in m.predict(np.array(data[v].assign(**{var:None})))
     ]
     
    score = d[[var,'Pred']].apply(
         lambda y: call_bin(y[var], y['Pred']), axis=1
     ).value_counts().to_dict()
    
    for k in ['TP', 'TN', 'FP', 'FN']:
        if k not in score.keys():
            score[k] = 0
            
    return score

def call_mult(o, p):
    if o == p:
        return 'TP'
    else:
        return 'FN'
    
def pred_mult(m, data, var):
    d = data.copy()
    v = [i.name for i in m.states]
    ix = [i for i in range(0, len(v)) if v[i] == var]
    #score = {'TP':0, 'FP':0, 'TN':0, 'FN':0}
    
    d['Pred'] = [
         p[ix] for p in m.predict(np.array(data[v].assign(**{var:None})))
     ]
     
    score = d[[var,'Pred']].apply(
         lambda y: call_mult(y[var], y['Pred']), axis=1
     ).value_counts().to_dict()
    
    for k in ['TP', 'TN', 'FP', 'FN']:
        if k not in score.keys():
            score[k] = 0
            
    return score

def accuracy(net, data):
    res = {}
    m = export_pom(net, by=net.by)
    for c in data.columns:
        #print(c)
        res[c] = ac(pred_mult(m, data, c))
    return res
    
    
def sensetivity(net, data):
    res = {}
    m = export_pom(net, by=net.by)
    for c in data.columns:
        #print(pred_bin(m, data, c))
        res[c] = sn(pred_bin(m, data, c))
    return res
    
def specificity(net, data):
    res = {}
    m = export_pom(net, by=net.by)
    for c in data.columns:
        #print(c)
        res[c] = sp(pred_bin(m, data, c))
    return res
    
    
    
    

# Data Set 4:  10 Variables generated by sampling from a 
# fixed set of conditional probability tables, with a maximum of 3 parents
# per table

# Alarm Network Subset
# alarm_net_variables={
#     'HISTORY':['TRUE', 'FALSE', 'FALSE', 'FALSE'],
#     'CVP':['LOW', 'NORMAL', 'HIGH', 'HIGH'],
#     'PWCP':['LOW', 'NORMAL', 'HIGH', 'HIGH'],
#     'EXPCO2':['ZERO','LOW', 'NORMAL', 'HIGH']
# }



# n1 = net(size=6, outcomes=(0,1))
# n1.add_edge(1,0)
# n1.add_edge(2,0)
# n1.add_edge(0,3)
# n1.add_edge(0,4)
# n1.add_edge(1,3)
# n1.add_edge(3,4)
# n1.add_edge(4,5)

# n1.nds[0].cpt=pd.DataFrame({
#     1:[0,0,0,0,1,1,1,1],
#     2:[0,0,1,1,0,0,1,1],
#     0:[0,1,0,1,0,1,0,1], 
#     'Prob':[0.1,0.9, 0.9, 0.1, 0.6, 0.4, 0.2, 0.8]
# })
# n1.nds[1].cpt=pd.DataFrame({
#     1:[0,1], 'Prob':[0.6, 0.4]
# })
# n1.nds[2].cpt=pd.DataFrame({
#     2:[0,1], 'Prob':[0.6, 0.4]
# })
# n1.nds[3].cpt=pd.DataFrame({
#     0:[0,0,0,0,1,1,1,1],
#     1:[0,0,1,1,0,0,1,1],
#     3:[0,1,0,1,0,1,0,1], 
#     'Prob':[0.5, 0.5, 0.8, 0.2, 0.8, 0.2, 0.5, 0.5]
# })

# n1.nds[4].cpt=pd.DataFrame({
#     0:[0,0,0,0,1,1,1,1],
#     3:[0,0,1,1,0,0,1,1],
#     4:[0,1,0,1,0,1,0,1], 
#     'Prob':[0.8, 0.2, 0.5, 0.5,0.5, 0.5, 0.8, 0.2]
# })

# n1.nds[5].cpt=pd.DataFrame({
#     4:[0,0,1,1],
#     5:[0,1,0,1],
#     'Prob':[0.8, 0.2, 0.5, 0.5]
# })

# train = sample_net(n1, 2000)
# train.columns = ['G'+str(i) for i in train.columns]
# #train.columns = [str(i) for i in range(0,6)]


# n2 = net(data=train)
# n2.add_edge('G1','G0')
# n2.add_edge('G2','G0')
# n2.add_edge('G0','G3')
# n2.add_edge('G0','G4')
# n2.add_edge('G1','G3')
# n2.add_edge('G3','G4')
# n2.add_edge('G4','G5')
# n2.calc_cpt(train, alpha=0.0001)

# g = greedy(train)
# g.train(300, 50)
# print(edge_hits(g, n2))

# h = CASGMM(train)
# h.train(300, 50)
# print(edge_hits(h, n2))

# i = CASJNB(train)
# i.train(300, 50)
# print(edge_hits(i, n2))

# # Issue with "from_samples" method; if passed a data frame, column
# # id's are lost.
# j = bn.from_samples(train)
# j.plot()
# export_pom(i.net, by='label').plot()

        
        
