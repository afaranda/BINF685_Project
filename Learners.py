#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 12:51:42 2020

@author: adam
"""
import pandas as pd
import numpy as np
from network import net
from network import export_pom
from network import score_pom

class greedy():
    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            return None
        else:
            self.data = data
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
        