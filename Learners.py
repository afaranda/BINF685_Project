#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 12:51:42 2020

@author: adam
"""
import pandas as pd
import numpy as np
import random as rn
from sklearn.metrics import mutual_info_score as mi
from sklearn.mixture import GaussianMixture as gm
from jenkspy import jenks_breaks
from network import net
from network import export_pom
from network import score_pom

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

class greedy():
    def __init__(self, data, alpha=0.000001):
        if not isinstance(data, pd.DataFrame):
            return None
        else:
            self.alpha = alpha
            self.data = data.copy()
            self.net = net(data = self.data)
            self.net.calc_cpt(self.data)
            self.scores = {}
            print(self.net.nds.keys())
            
    def train(self, iterations = 10, maxmiss=10):
        """
        Train using Recommended Greedy Algorithm
        """
        
        
        scores = {
            'Iteration':[],
            'Network':[],
            'Score':[]
        }
        score_check = maxmiss
        niter = iterations 
        nodes = [i for i in self.data.columns]
        best = score_pom(export_pom(self.net, by='label'), self.data)
        
        #print("START LOOP")
        while score_check > 0 and niter > 0:
            n = net(data = self.data)
            n.import_dag(self.net.export_dag())
            
            ops = [n.add_edge, n.del_edge, n.rev_edge]
            for f in ops:
                edge = np.random.choice(nodes, size = 2, replace=False)
                f(edge[0], edge[1])
              
            if n.acyclic():
                n.calc_cpt(self.data, alpha = self.alpha)
                score = score_pom(export_pom(n, by='label'), self.data)
                scores['Iteration'].append(iterations - niter)
                scores['Network'].append(n)
                scores['Score'].append(score)
                #print(best, score, niter, score_check)
                if score > best:
                    self.net = n
                    best = score
                    niter = niter - 1
                    score_check = maxmiss
                    continue
                else:
                    score_check = score_check - 1
                    niter = niter -1
                    continue
            else:
                niter = niter - 1
                continue
        self.scores = scores


class CASGMM():
    def __init__(self, data, alpha=0.000001):
        if not isinstance(data, pd.DataFrame):
            return None
        else:
            self.data = data.copy()
            self.net = net(data = self.data)
            self.net.calc_cpt(self.data)
            self.scores = {}
            self.alpha = alpha
            self.mi = self.pop_mi()
            self.E = self.CAS()
            

    def pop_mi(self):
        x = self.data.corr(mi)
        f = lambda x, i: (
            x[i].drop([i]) / 1
        )
            
        return {i:f(x,i) for i in x.columns}
        
    def CAS(self):
        V = self.data.columns
        E = set()

        for v in V:
            miv = self.mi[v]
            mii = self.mi[v].index
            C = list(mii)
            m = best_model(miv.values.reshape(-1,1))
            
            if m.n_components > 1:
                print("Split", v)
                if m.means_[0] > m.means_[1]:
                    C = m.predict(miv.values.reshape(-1,1))
                    print("Keep 0:", C)
                    C = [mii[i] for i in range(0,len(C)) if C[i] == 0]
            
                elif m.means_[0] < m.means_[1]:
                    C = m.predict(miv.values.reshape(-1,1))
                    print("Keep 1:",C)
                    C = [mii[i] for i in range(0,len(C)) if C[i] == 1]
         
            for c in C:
                E.add((v,c))
                E.add((c,v))
                
        return E
    
    def train(self, iterations = 10, maxmiss=10):
        """
        Train using Recommended Greedy Algorithm
        """
        
        
        scores = {
            'Iteration':[],
            'Network':[],
            'Score':[]
        }
        score_check = maxmiss
        niter = iterations 
        best = score_pom(export_pom(self.net, by='label'), self.data)
        
        #print("START LOOP")
        while score_check > 0 and niter > 0:
            n = net(data = self.data)
            n.import_dag(self.net.export_dag())
            
            ops = [n.add_edge, n.del_edge, n.rev_edge]
            for f in ops:
                edge = rn.sample(self.E,1)[0]
                f(edge[0], edge[1])
              
            if n.acyclic():
                n.calc_cpt(self.data, alpha = self.alpha)
                score = score_pom(export_pom(n, by='label'), self.data)
                scores['Iteration'].append(iterations - niter)
                scores['Network'].append(n)
                scores['Score'].append(score)
                #print(best, score, niter, score_check)
                if score > best:
                    self.net = n
                    best = score
                    niter = niter - 1
                    score_check = maxmiss
                    continue
                else:
                    score_check = score_check - 1
                    niter = niter -1
                    continue
            else:
                niter = niter - 1
                continue
        self.scores = scores      


class CASJNB():
    def __init__(self, data, alpha=0.000001):
        if not isinstance(data, pd.DataFrame):
            return None
        else:
            self.data = data.copy()
            self.net = net(data = self.data)
            self.net.calc_cpt(self.data)
            self.alpha = alpha
            self.scores = {}
            self.mi = self.pop_mi()
            self.E = self.CAS()
            

    def pop_mi(self):
        x = self.data.corr(mi)
        f = lambda x, i: (
            x[i].drop([i]) / 1
        )
            
        return {i:f(x,i) for i in x.columns}
        
    def CAS(self):
        V = self.data.columns
        E = set()

        for v in V:
            miv = sorted(
                (v, k) for (k, v) in 
                self.mi[v].to_dict().items()
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
    
    def train(self, iterations = 10, maxmiss=10):
        """
        Train using Recommended Greedy Algorithm
        """
        
        
        scores = {
            'Iteration':[],
            'Network':[],
            'Score':[]
        }
        score_check = maxmiss
        niter = iterations 
        best = score_pom(export_pom(self.net, by='label'), self.data)
        
        #print("START LOOP")
        while score_check > 0 and niter > 0:
            n = net(data = self.data)
            n.import_dag(self.net.export_dag())
            
            ops = [n.add_edge, n.del_edge, n.rev_edge]
            for f in ops:
                edge = rn.sample(self.E,1)[0]
                f(edge[0], edge[1])
              
            if n.acyclic():
                n.calc_cpt(self.data, alpha = self.alpha)
                score = score_pom(export_pom(n, by='label'), self.data)
                scores['Iteration'].append(iterations - niter)
                scores['Network'].append(n)
                scores['Score'].append(score)
                #print(best, score, niter, score_check)
                if score > best:
                    self.net = n
                    best = score
                    niter = niter - 1
                    score_check = maxmiss
                    continue
                else:
                    score_check = score_check - 1
                    niter = niter -1
                    continue
            else:
                niter = niter - 1
                continue
        self.scores = scores  


class CASMOD():
    def __init__(self, data, alpha=0.000001):
        if not isinstance(data, pd.DataFrame):
            return None
        else:
            self.data = data.copy()
            self.net = net(data = self.data)
            self.net.calc_cpt(self.data)
            self.scores = {}
            self.alpha = alpha
            self.mi = self.pop_mi()
            self.E = self.CAS()
            

    def pop_mi(self):
        x = self.data.corr(mi)
        f = lambda x, i: (
            x[i].drop([i]) / x[i].drop([i]).sum()
        )
            
        return {i:f(x,i) for i in x.columns}
        
    def CAS(self):
        V = self.data.columns
        E = set()

        for v in V:
            miv = self.mi[v]
            mii = self.mi[v].index
            C = list(mii)
            # get current edges with this node, with corresponding mi
            Cv = {
                i[1]:self.mi[v][i[1]] 
                for i in E if i[0] == v
            }
            
            # Skip Expectation Maximization if there are no "unused" nodes 
            # with an mi against 'v' > than the current candidates of 'v'
            if not (
                    min(list(Cv.values()) + [0]) >
                    max([self.mi[v][i] for i in set(mii) - Cv.keys()])
            ):
                m = best_model(miv.values.reshape(-1,1))
            
                if m.n_components > 1:
                    print("Split", v)
                    if m.means_[0] > m.means_[1]:
                        C = m.predict(miv.values.reshape(-1,1))
                        print("Keep 0:", C)
                        C = [mii[i] for i in range(0,len(C)) sample_net, if C[i] == 0]
                
                    elif m.means_[0] < m.means_[1]:
                        C = m.predict(miv.values.reshape(-1,1))
                        print("Keep 1:",C)
                        C = [mii[i] for i in range(0,len(C)) if C[i] == 1]
             
                for c in C:
                    E.add((v,c))
                    E.add((c,v))
                
        return E
    
    def train(self, iterations = 10, maxmiss=10):
        """
        Train using Recommended Greedy Algorithm
        """
        
        
        scores = {
            'Iteration':[],
            'Network':[],
            'Score':[]
        }
        score_check = maxmiss
        niter = iterations 
        best = score_pom(export_pom(self.net, by='label'), self.data)
        
        #print("START LOOP")
        while score_check > 0 and niter > 0:
            n = net(data = self.data)
            n.import_dag(self.net.export_dag())
            
            ops = [n.add_edge, n.del_edge, n.rev_edge]
            for f in ops:
                edge = rn.sample(self.E,1)[0]
                f(edge[0], edge[1])
              
            if n.acyclic():
                n.calc_cpt(self.data, alpha = 0.001)
                score = score_pom(export_pom(n, by='label'), self.data)
                scores['Iteration'].append(iterations - niter)
                scores['Network'].append(n)
                scores['Score'].append(score)
                #print(best, score, niter, score_check)
                if score > best:
                    self.net = n
                    best = score
                    niter = niter - 1
                    score_check = maxmiss
                    continue
                else:
                    score_check = score_check - 1
                    niter = niter -1
                    continue
            else:
                niter = niter - 1
                continue
        self.scores = scores     




class greedy_nmi_weighted():
    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            return None
        else:
            self.data = data.copy()
            self.net = net(data = self.data)
            self.net.calc_cpt(self.data)
            self.scores = {}
            self.mi_weights = {}
            print(self.net.nds.keys())
            self.calc_norm_mi_Weights()
    def calc_norm_mi_Weights(self):
        x = self.data.corr(mi)
        f = lambda x, i: (
            x[i].drop([i]) / x[i].drop([i]).sum()
        )
            
        self.mi_weights={i:f(x,i) for i in x.columns}
      
        
            
    def train(self, iterations = 10, maxmiss=10):
        """
        Train using Recommended Greedy Algorithm
        """
        scores = {
            'Iteration':[],
            'Network':[],
            'Score':[]
        }
        score_check = maxmiss
        niter = iterations 
        nodes = [i for i in self.data.columns]
        best = score_pom(export_pom(self.net, by='label'), self.data)
        
        #print("START LOOP")
        while score_check > 0 and niter > 0:
            n = net(data = self.data)
            n.import_dag(self.net.export_dag())
            
            ops = [n.add_edge, n.del_edge, n.rev_edge]
            
            for f in ops:
                  # Choose the first node in a uniform, random way
                  v1 = np.random.choice(nodes)
                 
                  # Choose the second with probabilities weighted by mi
                  v2 = np.random.choice(
                      self.mi_weights[v1].index,
                      p=self.mi_weights[v1]
                  )
                  f(v1, v2)
                  
            if n.acyclic():
                n.calc_cpt(self.data, alpha = 0.001)
                score = score_pom(export_pom(n, by='label'), self.data)
                scores['Iteration'].append(iterations - niter)
                scores['Network'].append(n)
                scores['Score'].append(score)
                #print(best, score, niter, score_check)
                if score > best:
                    self.net = n
                    best = score
                    niter = niter - 1
                    score_check = maxmiss
                    continue
                else:
                    score_check = score_check - 1
                    niter = niter -1
                    continue
            else:
                niter = niter - 1
                continue
        self.scores = scores
        
#     def get_prob(self, query, target):
#         qr = query.copy()
#         tg = target
#         par = self.net.nds[tg].parent_idx()
#         ocm = self.net.nds[tg].ocm
#         cpt = self.net.nds[tg].cpt.copy()
#         if len(par) > 0:
#             idx = pd.MultiIndex.from_tuples(
#                 [tuple([qr[j] for j in par + [tg]])],
#                 names = par + [tg]
#             )
#         else:
#             idx = pd.Index([qr[tg]]).set_names(tg)
                
#         pr = cpt.set_index(par + [tg]).loc[idx].values[0][0] 
#         return(pr)

#     def call_target(self, query, target):
#         """
#         Predict the status of a target gene (0 or 1) given the status of
#         other genes in the network.  This function retrieves the CPT for
#         the target gene and returns the most likely outcome based on
#         probabilities estimated from the training set.
#         """
#         qr = query.copy()
#         tg = target
#         par = self.net.nds[tg].parent_idx()
#         ocm = self.net.nds[tg].ocm
#         cpt = self.net.nds[tg].cpt.copy()
#         if len(par) > 0:
#             idx = pd.MultiIndex.from_tuples(
#                 [tuple([qr[j] for j in par] + [i]) for i in ocm ],
#                 names = par + [tg]
#             )
#         else:
#             idx = pd.Index(ocm).set_names(tg)
#         call = cpt.set_index(par + [tg]).loc[idx]
#         call = call.reset_index()
#         call = call[tg].loc[call['Prob'] == max(call.Prob)].values[0]
#         return(call)


#     def predict(self, query, target, alpha = 0):
#         """
#         Predict the status of a target gene (0 or 1) given the status of
#         one or more query genes, using samples generated via direct sampling.
#         """
#         qr = query.copy()
#         ocm = self.net.nds[target].ocm
#         cpt = self.samples[qr.index].copy()
#         cpt['Prob'] = 0

#         # # Flatten cpt to counts over observed outcomes
#         cpt = cpt.groupby(list(qr.index)).count().copy()
#         if len(qr.drop(target).index) > 0:
#             idx = pd.MultiIndex.from_tuples(
#                 [tuple([qr[j] for j in qr.drop(target).index] + [i]) for i in ocm],
#                 names = list(qr.index)
#             )
#         else:
#             idx =pd.Index(ocm).set_names(target)

        
#         # For possible, but unobserved outcomes, convert NaN to 0
#         cpt = cpt.reindex(idx).fillna(0)
        
#         cpt = cpt.reset_index()
        
#         # Use a rather complicated lambda to calculate frequencies
#         #if len(qr.drop(target).index) == 0:
#         #    cpt['Prob'] = cpt.transform(
#         #    lambda x: (x + alpha) / (x.sum() +(x.count()*alpha)))['Prob']
#         #else:
#         #    cpt['Prob'] = cpt.groupby(list(qr.drop(target).index)).transform(
#         #        lambda x: (x + alpha) / (x.sum() +(x.count()*alpha))
#         #    )['Prob']
        

#         print(cpt)
#         return(cpt.loc[cpt['Prob'] == max(cpt['Prob']), target])




        
#         #call = cpt.set_index(par + [tg]).loc[idx]
#         #call = call.reset_index()
#         #call = call[tg].loc[call['Prob'] == max(call.Prob)].values[0]
#         #return(call)

#     def test_accuracy(self, test, target = 6):
#         data=test
#         data['Pred_Prob'] = data.apply(self.get_prob, args=(target,), axis=1)
#         data['Predict'] = data.apply(self.call_target, args=(target,), axis=1)
#         data['Correct'] = data.apply(lambda x: x[0] == x['Predict'], axis=1)
#         return(data)
        