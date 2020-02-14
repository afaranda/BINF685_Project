#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 12:16:11 2020

@author: adam

"""
##############################################################################
# 
# This script defines a function that generates a matrix of mutual
# mutual information estimates for a given data set.  
#
#

import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score as mi
from sklearn.metrics import normalized_mutual_info_score as nmi


def mi_matrix(df):
    mat = pd.DataFrame(
        index=df.columns, 
        columns=df.columns, 
        data=0
    )
    for i in df.columns:
        for j in df.columns:
            mat.loc[i,j] = mi(df[i], df[j])
            
    return mat



def nmi_matrix(df):
    mat = pd.DataFrame(
        index=df.columns, 
        columns=df.columns, 
        data=0
    )
    for i in df.columns:
        for j in df.columns:
            mat.loc[i,j] = nmi(df[i], df[j])
            
    return mat

data = pd.DataFrame({
    'G1':np.repeat([1,2,3,4],5),
    'G2':np.repeat([1,2],10),
    'G3':np.repeat([1,2,1,2],5),
    'G4':np.repeat([1,1,1,2],5),
    'G5':np.repeat([1,1,2,1],5),
    'G6':np.repeat([2,2,3,1],5),
    'G7':np.repeat([1,2,3,4,5],4)
    })

# My function is equivalent to using the the Pandas 'corr' method with 
# nmi as the metric. 
data.corr(method=nmi) - nmi_matrix(data)







