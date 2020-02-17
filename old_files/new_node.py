#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 15:25:14 2020

@author: adam
"""

import pandas as pd
import numpy as np
import random as rn
import pygraphviz as pgv
import pomegranate as pm
#import itertools as it
#from math import log
#from copy import deepcopy as dc


n1=node(1,label='n1')
n2=node(2,label='n2', outcomes=('A', 'B', 'C'))
n3=node(3, label='n3', outcomes=('X', 'Y'))
n2.add_parents([n1])
n2.add_parents([n3])


df=pd.DataFrame({
    'n1':[0,0,0,0,1,1,1,1],
    'n2':['A','A','A','A','B','B','C','C'],
    'n3':['X','X','X','X','X','X','Y','Y']  
    })

x=n2.cpd_index()
idx = pd.MultiIndex.from_tuples(
                    x, names = df.columns
                )


class node:
    '''
    A node represents a single variable in the Bayes Net. Each node stores a
    conditional probability table that enumerates the probability of all 
    possible outcomes given the state of each parent.  
    '''
    def __init__(self, 
                idx, 
                label=None, 
                parents=[], 
                outcomes=(0,1)):

        self.idx = idx
        self.label = label
        self.par = parents
        self.ocm = outcomes
        self.cpt = None
        self.cpd = {}
    
    def cpd_index(self):
        var = self.par + [self]
        if len(var) == 1:
            idx = [(o,) for o in var[0].ocm]
            return(idx)
        
        else:
            combos=[]
            for v in var:
                if len(combos)  == 0:
                    # Initialize with a list of outcomes
                    combos = [list([i]) for i in v.ocm]
            
                else:
                    # Iterate over each existing combination
                    # for each existing combination, add one new combo
                    # for each outcome in the next variable
               
                    new = []
                    for c in combos:
                        for o in v.ocm:
                            merge = c + [o]
                            new.append(merge)
                            
                    combos=new
            combos = [tuple(i) for i in combos]
            
            return(combos)
    
    def cpt_index(self, by='index'):
        """
        Construct a Pandas index or MultiIndx corresponding to  all possible
        combinations of parents by outcomes for this node. 
        """
        if by == 'index':
            var = self.par + [self]
            if len(var) == 1:
                idx = pd.Index(list(self.ocm)).set_names(var[0].idx)
                return(idx)
            
            else:
                combos=[]
                for v in var:
                    if len(combos)  == 0:
                        # Initialize with a list of outcomes
                        combos = [list([i]) for i in v.ocm]
                
                    else:
                        # Iterate over each existing combination
                        # for each existing combination, add one new combo
                        # for each outcome in the next variable
                   
                        new = []
                        for c in combos:
                            for o in v.ocm:
                                merge = c + [o]
                                new.append(merge)
                                
                        combos=new
                combos = [tuple(i) for i in combos]
                idx = pd.MultiIndex.from_tuples(
                    combos, names = [v.idx for v in var]
                )
                return(idx)
            
        elif by == 'label':
            var = self.par + [self]
            if len(var) == 1:
                lbl = pd.Index(list(self.ocm)).set_names(var[0].label)
                return(lbl)
            
            else:
                combos=[]
                for v in var:
                    if len(combos)  == 0:
                        # Initialize with a list of outcomes
                        combos = [list([i]) for i in v.ocm]
                
                    else:
                        # Iterate over each existing combination
                        # for each existing combination, add one new combo
                        # for each outcome in the next variable
                   
                        new = []
                        for c in combos:
                            for o in v.ocm:
                                merge = c + [o]
                                new.append(merge)
                                
                        combos=new
                combos = [tuple(i) for i in combos]
                lbl = pd.MultiIndex.from_tuples(
                    combos, names = [v.label for v in var]
                )
                return(lbl)
        else: 
                return None
                
    
    def parent_idx(self):
        idx = []
        for i in self.par:
            idx.append(i.idx)
        return(idx)
    
    def parent_lbl(self):
        lbl = []
        for i in self.par:
            lbl.append(i.label)
        return(lbl)
    
    def add_parents(self, parents):
        new = []
        for p in parents:
            if self in p.par:
                print("Cant have a parent that is also a child")
                return(False)
            
            elif self == p:
                print("No Self Parenting Allowed")
                return False
            
            elif p in self.par:
                print("Already has this parent")
                return(False)
            
            else:
                new.append(p)
                self.par = self.par + new
                return(True)
    
    def del_parents(self, parents):
        x = set(self.par).difference(parents)
        self.par = list(x)
        

    def node_probs(self, data, alpha = 0.001, by='index'):
        """
        Estimate the probabilities for each possible outcome of the variable
        represented by this node, given its parents by summing observations in
        a data set over all possible parent-child outcomes and normalizing 
        by the number of child outcomes. 
        
        """
                
        index = self.cpt_index(by=by)
        if by == 'index':
            var = self.parent_idx() + [self.idx]
            par = self.parent_idx()
            
            # Copy data, add a column to calcuate conditional probabilities
            cpt = data[var].copy()
            cpt['Prob'] = 0
        
        elif by == 'label':
            var = self.parent_lbl() + [self.label]
            par = self.parent_lbl()
            
            # Copy data, add a column to calcuate conditional probabilities
            cpt = data[var].copy()
            cpt['Prob'] = 0

        # # Flatten cpt to counts over observed outcomes
        cpt = cpt.groupby(var).count().copy()

        # For possible, but unobserved outcomes, convert NaN to 0
        cpt = cpt.reindex(index).fillna(0)
        
        cpt = cpt.reset_index()
        
       
        #Use a rather complicated lambda to calculate frequencies
        if len(par) == 0:
            cpt['Prob'] = cpt['Prob'].transform(
            lambda x: (x + alpha) / (x.sum() +(x.count()*alpha)))
        else:
            cpt['Prob'] = cpt.groupby(par)['Prob'].transform(
                lambda x: (x + alpha) / (x.sum() +(x.count()*alpha))
            )
            
        self.cpt = cpt
    
    def empty_cpt(self, by='index'):
        """
        Calculate total CPT size based on number of parents, 
        outcomes.  Build empty table with columns for each parent's
        possible outcomes and this nodes possible outcome.  Add a 
        column to store the probability for each record.
        """
        df = pd.DataFrame(
            index=self.cpt_index(by=by),
            columns=['Prob']
        )
        df['Prob'] = 0.0
        return df.reset_index()