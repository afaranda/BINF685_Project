#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 20:12:14 2020

@author: adam
"""
import timeit as tm
import pandas as pd
resdir="results"
#### Define Enironment (lots of code in a string)
envir = """
import pandas as pd
import numpy as np
import random as rn
from copy import deepcopy as dc
from sklearn.metrics import mutual_info_score as mis
from sklearn.mixture import GaussianMixture as gm
from jenkspy import jenks_breaks
from network import net
from network import export_pom
from network import score_pom

from generate_networks import ds1, ds2, ds4

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
        # print(max([mi[v][i] for i in set(mii) - Cv.keys()]))
        # print((min(list(Cv.values())) if len(Cv) > 0 else 0))
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
        else:
           print('skip')
            
    return E
ds1_mi = pop_mi(ds1)
ds2_mi = pop_mi(ds2)
ds4_mi = pop_mi(ds4)
"""

#### define timers:
ds1cgm = tm.Timer('CASGMM(ds1, ds1_mi)', setup=envir)
ds1cmd = tm.Timer('CASMOD(ds1, ds1_mi)', setup=envir)
ds1cjn = tm.Timer('CASJNB(ds1, ds1_mi)', setup=envir)
ds2cgm = tm.Timer('CASGMM(ds2, ds2_mi)', setup=envir)
ds2cmd = tm.Timer('CASMOD(ds2, ds2_mi)', setup=envir)
ds2cjn = tm.Timer('CASJNB(ds2, ds2_mi)', setup=envir)
ds4cgm = tm.Timer('CASGMM(ds4, ds4_mi)', setup=envir)
ds4cmd = tm.Timer('CASMOD(ds4, ds4_mi)', setup=envir)
ds4cjn = tm.Timer('CASJNB(ds4, ds4_mi)', setup=envir)

timing = pd.DataFrame({
    'ds1cgm':ds1cgm.repeat(50, 1),
    'ds1cmd':ds1cmd.repeat(50, 1),
    'ds1cjn':ds1cjn.repeat(50, 1),
    'ds2cgm':ds2cgm.repeat(50, 1),
    'ds2cmd':ds2cmd.repeat(50, 1),
    'ds2cjn':ds2cjn.repeat(50, 1),
    'ds4cgm':ds2cgm.repeat(50, 1),
    'ds4cmd':ds2cmd.repeat(50, 1),
    'ds4cjn':ds2cjn.repeat(50, 1)
})
timing.to_csv(resdir + "Timing_Results.csv", index=False)


