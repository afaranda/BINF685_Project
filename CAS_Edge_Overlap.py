#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 20:12:14 2020

@author: adam
"""
resdir="results"
#### Define Enironment (lots of code in a string)

import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score as mis
from sklearn.mixture import GaussianMixture as gm
from jenkspy import jenks_breaks
np.random.seed(21921)

from generate_networks import ds1, ds2, ds4

def threeway(E1, E2, E3, filename="inx.csv", count=True):
    results = {
        'E1-E2uE3': E1.difference(E2.union(E3)),
        'E2-E1uE3': E2.difference(E1.union(E3)),
        'E3-E1uE2': E3.difference(E1.union(E2)),
        'E1iE2-E3': E1.intersection(E2).difference(E3),
        'E1iE3-E2': E1.intersection(E3).difference(E2),
        'E2iE3-E1': E2.intersection(E3).difference(E1),
        'E1iE2iE3': E1.intersection(E2).intersection(E3)
    }
    if count:
        return {i:len(results[i]) for i in results.keys()}
    else:
        return results

def best_model(X):
    '''

    Parameters
    ----------
    X : numpy array with shape (-1,1)
        Vector of mutual information values 

    Returns
    -------
    best_gmm : Returns the gausian mixture model 

    '''
    lowest_bic = np.infty
    bic = []
    n_components_range = [1,2]
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = gm(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    return best_gmm

def pop_mi(data):
    x = data.corr(mis)
    f = lambda x, i: (
        x[i].drop([i]) / 1
    )
    return {i:f(x,i) for i in x.columns}


def CASGMM(data, mi):
    V = data.columns
    E = set()

    for v in V:
        miv = mi[v]
        mii = mi[v].index
        C = list(mii)
        m = best_model(miv.values.reshape(-1,1))
        
        if m.n_components > 1:
            #print("Split", v)
            if m.means_[0] > m.means_[1]:
                C = m.predict(miv.values.reshape(-1,1))
                #print("Keep 0:", C)
                C = [mii[i] for i in range(0,len(C)) if C[i] == 0]
        
            elif m.means_[0] < m.means_[1]:
                C = m.predict(miv.values.reshape(-1,1))
                #print("Keep 1:",C)
                C = [mii[i] for i in range(0,len(C)) if C[i] == 1]
     
        for c in C:
            E.add((v,c))
            E.add((c,v))
            
    return E

def CASJNB(data, mi):
    V = data.columns
    E = set()

    for v in V:
        miv = sorted(
            (v, k) for (k, v) in 
            mi[v].to_dict().items()
        )
        brk = jenks_breaks(
            [miv[i][0] for i in range(len(miv))],2
        )[1]
        C = [
                miv[i][1] 
                for i in range(len(miv)) 
                if not miv[i][0] < brk
            ]
     
        for c in C:
            E.add((v,c))
            E.add((c,v))
            
    return E

def CASMOD(data, mi):
    V = data.columns
    E = set()

    for v in V:
        miv = mi[v]
        mii = mi[v].index
        C = list(mii)
        # get current edges with this node, with corresponding mi
        Cv = {
            i[1]:mi[v][i[1]] 
            for i in E if i[0] == v
        }
        
        # Skip Expectation Maximization if there are no "unused" nodes 
        # with an mi against 'v' > than the current candidates of 'v'
        #print(v, C, Cv,"\n")
        #print(max([mi[v][i] for i in set(mii) - Cv.keys()]))
        #print((min(list(Cv.values())) if len(Cv) > 0 else 0))
        if len(Cv) == 0:
            threshold = 0
        else:
            threshold = min(list(Cv.values()))
        if not (
            threshold >
            max([mi[v][i] for i in set(mii) - Cv.keys()])
        ):
            
            
            m = best_model(miv.values.reshape(-1,1))
        
            if m.n_components > 1:
                # print("Split", v)
                if m.means_[0] > m.means_[1]:
                    C = m.predict(miv.values.reshape(-1,1))
                    # print("Keep 0:", C)
                    C = [mii[i] for i in range(0,len(C)) if C[i] == 0]
            
                elif m.means_[0] < m.means_[1]:
                    C = m.predict(miv.values.reshape(-1,1))
                    # print("Keep 1:",C)
                    C = [mii[i] for i in range(0,len(C)) if C[i] == 1]
               
            for c in C:
                E.add((v,c))
                E.add((c,v))
            #print(v, C, "\n")
        else:
           print('skip')
            
    return E

ds1_mi = pop_mi(ds1)
ds2_mi = pop_mi(ds2)
ds4_mi = pop_mi(ds4)

np.random.seed(21921)
ds1cgmEdge = CASGMM(ds1, ds1_mi)
np.random.seed(21921)
ds1cmdEdge = CASMOD(ds1, ds1_mi)
np.random.seed(21921)
ds1cjnEdge = CASJNB(ds1, ds1_mi)

np.random.seed(21921)
ds2cgmEdge = CASGMM(ds2, ds2_mi)
np.random.seed(21921)
ds2cmdEdge = CASMOD(ds2, ds2_mi)
np.random.seed(21921)
ds2cjnEdge = CASJNB(ds2, ds2_mi)

np.random.seed(21921)
ds4cgmEdge = CASGMM(ds4, ds4_mi)
np.random.seed(21921)
ds4cmdEdge = CASMOD(ds4, ds4_mi)
np.random.seed(21921)
ds4cjnEdge = CASJNB(ds4, ds4_mi)
    
#### Print Output To Screen -- save as text file
print("Venn values for ds1")
print(
    threeway(
        ds1cgmEdge,              #E1 CASGMM edge set
        ds1cjnEdge,              #E2 CASJNB edge set
        ds1cmdEdge,              #E3 CASMOD edge set
        count=True
    ), "\n"
)
        

print("Venn values for ds2")
print(
    threeway(
        ds2cgmEdge,              #E1 CASGMM edge set
        ds2cjnEdge,              #E2 CASJNB edge set
        ds2cmdEdge,              #E3 CASMOD edge set
        count=True
    ), "\n"
)
        
print("Venn values for ds4")
print(
    threeway(
        ds4cgmEdge,              #E1 CASGMM edge set
        ds4cjnEdge,              #E2 CASJNB edge set
        ds4cmdEdge,              #E3 CASMOD edge set
        count=True
    ), "\n"
)
        
    
    









