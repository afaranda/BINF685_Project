import numpy as np
import pandas as pd
import random as rn
#from network import greedy
from network import node
from network import net
from network import val
from network import topoSort
from network import sample_net

from matplotlib import pyplot as plt
from pyitlib import discrete_random_variable as drv
from pomegranate import *



######### Validate Sampler on Randomly Generated Data:
data={
      'G2':[rn.choices([0,1], weights=[0.5, 0.5])[0] for i in range(0,2000)] 
      }


# Using this function, G3 is dependent on G1 -- independent of G2
def ofun(a):
    if a==1:
        return rn.choices([0,1], weights=[0.9,0.1])[0]
    else:
        return rn.choices([0,1], weights=[0.1,0.9])[0]
    

data['G1'] = [
    ofun(data['G2'][i]) 
    for i in range(0,len(data['G2']))
    ]

data['G3'] = [
    rn.choices(
        ['A', 'B', 'C'], 
        weights=[0.15, 0.3, 0.55])[0] 
    for i in range(0,2000)
] 



data = pd.DataFrame(data)
data = data[['G1', 'G2', 'G3']]

n = net(data=data)
n.add_edge(1,0)
n.calc_cpt(data)

data2 = sample_net(n, 2000)

m=net(data=data2)
m.add_edge(1,0)
m.calc_cpt(data2)






# ###### Pandas Testing

# df=pd.DataFrame({
#     0:[0,0,0,0,1,1,1,1],
#     2:[0,0,1,1,0,0,1,1],
#     1:[0,1,0,1,0,1,0,1],
#     'Prob':[10/20, 10/20, 1/10, 9/10, 9/10, 1/10, 4/10, 6/10 ]
    
    
#     })

# gr=df.groupby([0,2]).groups


# rn.choices(
#     df.loc[gr[(0,0)]][1].values,
#     df.loc[gr[(0,0)]]['Prob'].values
#     )[0]   


    
# r = 2    
# df = []
# x = 0
# while x < r:
#     l = topoSort(n.export_nds())
#     row = {}
#     for m in l:
#         if len(m.par) > 1:
#             gr = m.cpt.groupby([j.idx for j in m.par]).groups
#             po = tuple(
#                 [row[k] for k in [j.idx for j in m.par] ][0]
#             )
#             print(po)
#             row[m.idx] = rn.choices(
#                     m.cpt.loc[gr[po]][m.idx].values,
#                     m.cpt.loc[gr[po]]['Prob'].values
#             )[0]
            
#         elif len(m.par) > 0:
#             gr = m.cpt.groupby([j.idx for j in m.par]).groups
#             po = row[m.par[0].idx]
#             print(po)
#             row[m.idx] = rn.choices(
#                     m.cpt.loc[gr[po]][m.idx].values,
#                     m.cpt.loc[gr[po]]['Prob'].values
#             )[0]
            
#         else:
#             print("noPar")
#             row[m.idx] = rn.choices(
#                 m.cpt[m.idx],
#                 m.cpt['Prob']
#             )[0]
#     df.append(
#         [row[k] for k in sorted(row.keys())]  
#     )
#     x=x+1
    
    
    
# r = 2    
# df = []
# x = 0
# while x < r:
#     l = topoSort(n.export_nds())
#     row = {}
#     for m in l:
#         if len(m.par) > 1:
#             gr = m.cpt.groupby([j.label for j in m.par]).groups
#             po = tuple(
#                 [row[k] for k in [j.label for j in m.par] ]
#             )
#             print(po)
#             row[m.label] = rn.choices(
#                     m.cpt.loc[gr[po]][m.label].values,
#                     m.cpt.loc[gr[po]]['Prob'].values
#             )[0]
            
#         elif len(m.par) > 0:
#             gr = m.cpt.groupby([j.label for j in m.par]).groups
#             po = row[m.par[0].label]
#             print(po)
#             row[m.label] = rn.choices(
#                     m.cpt.loc[gr[po]][m.label].values,
#                     m.cpt.loc[gr[po]]['Prob'].values
#             )[0]
            
#         else:
#             print("noPar")
#             row[m.label] = rn.choices(
#                 m.cpt[m.label],
#                 m.cpt['Prob']
#             )[0]
            
#     df.append([row[k] for k in
#         [{i.idx:i.label for i in l}[j] 
#         for j in sorted(
#                 {i.idx:i.label for i in l}.keys()
#             )]
#         ]) 
#     x=x+1


# def sample_net(net, r):
#     """
#     Using a simple 'prior sampling' approach, generate 
#     n samples from this 'net' object.
#     """

#     return df
            
    # For each node in a topologically sorted node list
    #     Sample the outcome of the node, based on it's CPT,
    #     Given the outcomes of it's parents (If any)
    #     Add the sampled outcome to a node-outcome dictionary
    
    # Sort the Node-Outcome dictionary in order of index
    # Append the list of outcomes to a growing table of samples
    # Return the table of samples when all
    
    
    # if self.by == 'index':
    #     while x < n:
    #         l = topoSort(self.export_nds())
    #         row = {}
    #         for i in l:
    #             var = i.cpt.columns.drop('Prob')
    #             prior = {k:row[k] for k in row.keys() if k in var}
    #             c = i.cpt.copy()
    #             c = c.sort_values(by=list(c.columns))

    #             # Subset cpt
    #             #print(prior)
    #             for j in prior.keys():
    #                     c = c[c[j] == prior[j]]

    #             s = rn.uniform()
    #             #print("NODE:", i.idx, "Draw:", s)
    #             #print(c)
    #             #print(c.loc[c[i.idx] == 0, 'Prob'])
                
    #             if s < c.loc[c[i.idx] == 0, 'Prob'].values[0]:
    #                 row[i.idx] = 0
    #             else:
    #                 row[i.idx] = 1
    #             #print(row.keys(), row.values())
    #         df.append([row[m] for m in sorted(row.keys())])
    #         x = x + 1
    #     df = pd.DataFrame(df)
    #     return(df)








# # Validate Sort using Randomly Generated Acyclic Networks
# for i in range(0,100):
#     n = net(size=20, outcomes=(0,1))
#     used = []
#     for i in n.nds.keys():
#         used.append(i)
#         for j in set(n.nds.keys()).difference(used):
#             if rn.choices([True, False])[0]:
#                 n.add_edge(i, j)      
    
#        #print(net[i].idx, [ j.idx for j in net[i].par])
#     l = n.export_nds()
#     rn.shuffle(l)
#     m = topoSort(l)
#     if val(n.nds):
#         for i in n.nds.keys():
#             print(n.nds[i].idx, [ j.idx for j in n.nds[i].par])     
#     print(val(n.nds))
    
    
    
    
# n = net().fill_un

# n.add_edge(0,1)
# n.add_edge(0,2)
# n.add_edge(3,5)
# n.add_edge(5,6)
# n.add_edge(6,4)
# n.add_edge(4,0)
# n.rev_edge(0,2)
# n.add_edge(2,5)

# G1 = node(1, label="FOO")
# G2 = node(2, label="BAR")
# G3 = node(3, label="BIF")

# G1.add_parents([G2])
# G2.add_parents([G3])
# G3.del_parents([G1])

# net = {1:G1, 2:G2, 3:G3}

##### Validated On Binary Outcomes

# G1 = node(1, label="G1")
# G2 = node(2, label="G2")
# G3 = node(3, label="G3", parents=[G2, G1])






# data={
#       'G1':[rn.choices([0,1], weights=[0.5, 0.5])[0] for i in range(0,1000)],
#       'G2':[rn.choices([0,1], weights=[0.2, 0.8])[0] for i in range(0,1000)]  
#       }

# def pfun(a,b):
#     if a==1:
#         if b == 1:
#             return rn.choices([0,1], weights=[0.9,0.1])[0]
#         elif b == 0:
#             return rn.choices([0,1], weights=[0.1,0.9])[0]
        
#     elif a==0:
#         if b == 1:
#             return rn.choices([0,1], weights=[0.1,0.9])[0]
#         elif b == 0:
#             return rn.choices([0,1], weights=[0.9,0.1])[0]
    
# data['G3'] = [
#     pfun(data['G1'][i], data['G2'][i]) 
#     for i in range(0,len(data['G1']))
#     ]

# data = pd.DataFrame(data)
# G1.node_probs(data, alpha=0.00001, by='label')
# G2.node_probs(data, alpha=0.00001, by='label')
# G3.node_probs(data, alpha=0.00001, by='label')

# data={
#       1:[rn.choices([0,1], weights=[0.5, 0.5])[0] for i in range(0,1000)],
#       2:[rn.choices([0,1], weights=[0.2, 0.8])[0] for i in range(0,1000)]  
#       }

# def pfun(a,b):
#     if a==1:
#         if b == 1:
#             return rn.choices([0,1], weights=[0.9,0.1])[0]
#         elif b == 0:
#             return rn.choices([0,1], weights=[0.1,0.9])[0]
        
#     elif a==0:
#         if b == 1:
#             return rn.choices([0,1], weights=[0.1,0.9])[0]
#         elif b == 0:
#             return rn.choices([0,1], weights=[0.9,0.1])[0]
    
# data[3] = [
#     pfun(data[1][i], data[2][i]) 
#     for i in range(0,len(data[1]))
#     ]

# data = pd.DataFrame(data)
# G1.node_probs(data, alpha=0.00001)
# G2.node_probs(data, alpha=0.00001)
# G3.node_probs(data, alpha=0.00001)

# ###### Validate node on Multinomial Outcomes
# G1 = node(1, label="G1")
# G2 = node(2, label="G2", outcomes=('A', 'B', 'C'))
# G3 = node(3, label="G3", parents=[G2, G1])


# data={
#       'G1':[rn.choices([0,1], weights=[0.5, 0.5])[0] for i in range(0,2000)],
#       'G2':[rn.choices(['A', 'B', 'C'], weights=[0.15, 0.3, 0.55])[0] for i in range(0,2000)]  
#       }

# def pfun(a,b):
#     if a==1:
#         if b == 'A':
#             return rn.choices([0,1], weights=[0.9,0.1])[0]
#         elif b == 'B':
#             return rn.choices([0,1], weights=[0.1,0.9])[0]
#         elif b =='C':
#             return rn.choices([0,1], weights=[0.9,0.1])[0]
    
#     if a==0:
#         if b == 'A':
#             return rn.choices([0,1], weights=[0.1,0.9])[0]
#         elif b == 'B':
#             return rn.choices([0,1], weights=[0.1,0.9])[0]
#         elif b =='C':
#             return rn.choices([0,1], weights=[0.1,0.9])[0]
# # def ofun(a):
# #     if a==1:
# #         return rn.choices([0,1], weights=[0.9,0.1])[0]
# #     else:
# #         return rn.choices([0,1], weights=[0.1,0.9])[0]
    
# data['G3'] = [
#     pfun(data['G1'][i], data['G2'][i]) 
#     for i in range(0,len(data['G1']))
#     ]

# # data['G3'] = [
# #     ofun(data['G1'][i]) 
# #     for i in range(0,len(data['G1']))
# #     ]

# data = pd.DataFrame(data)

# n = net(data=data)

# n.add_edge(2,0)
# n.add_edge(0,1)
# n.add_edge(2,1)


# # data['G2'] = data['G2'].astype('str')
# # G1.node_probs(data, alpha=0.00001, by='label')
# # G2.node_probs(data, alpha=0.00001, by='label')
# # G3.node_probs(data, alpha=0.00001, by='label')

# bn = BayesianNetwork.from_samples(data)



# def fn(x):
#     if x == 'A':
#         return 0
#     if x == 'B':
#         return 1
#     if x == 'C':
#         return 2
#     else:
#         return 3
# data['G2'] = data['G2'].apply(fn)




# # Load Data Tables
# train = pd.read_csv(
#     'hw2_train.data', 
#     header=None, 
#     sep="\t"
# )

# test = pd.read_csv(
#     'hw2_test.data', 
#     header=None, 
#     sep="\t"
# )

# n=net(3)
# n.add_edge(0,1)
# n.add_edge(2,1)

# n.nds[0].cpt=pd.DataFrame({0:[0,1], 'Prob':[0.5, 0.5]})
# n.nds[2].cpt=pd.DataFrame({0:[0,1], 'Prob':[0.6, 0.4]})

# n.nds[1].cpt=pd.DataFrame({
#     0:[0,0,0,0,1,1,1,1],
#     2:[0,0,1,1,0,0,1,1],
#     1:[0,1,0,1,0,1,0,1],
#     'Prob':[0.2, 0.8, 0.8, 0.2, 0.8, 0.2, 0.2, 0.8]})



# ############ Validate Sampler on Multinomial Model
# n=net(3, outcomes= )
# n.add_edge(0,1)
# n.add_edge(2,1)








# n = net(8)
# n.add_edge(0,1)
# n.add_edge(0,2)
# n.add_edge(0,3)
# n.add_edge(2,5)
# n.add_edge(7,5)
# n.add_edge(5,6)
# n.calc_cpt(train)

# train.columns = ["G"+ str(c) for c in train.columns]
# p=n.export_pom()[2]

# p.log_probability(train)






# n = net(8, outcomes=('U', 'D'))
# n.add_edge(0,1)
# n.add_edge(0,2)
# n.add_edge(0,3)
# n.add_edge(2,5)
# n.add_edge(7,5)
# n.add_edge(5,6)


# train.columns = ["G"+ str(c) for c in train.columns]
# for c in train.columns:
#     train[c] = train[c].apply(lambda x: 'U' if x == 1 else 'D')
# n.calc_cpt(train)



# n.score_net(train)
# p=n.export_pom()[2]
# p.fit(train)

# for i in s:
#     print(i.idx)
#     print([j.idx for j in i.par])


# def is_dpord(l):
#         """
#         Function to check a node list is ordered parent to child
#         """
#         for i in range(0, len(l) -1):
#             pi = l[i].par
#             for j in pi:
#                 for k in range(i+1, len(l)):
#                     if j == l[k]:
#                         return False
#                     else:
#                         continue
#         return True

# def sort_nodes(l):
#     stop = 0
#     l = l #list(self.nds.values())
#     p = 0
#     while (not is_dpord(l)) and stop <100000:
#         pi = l[p].par
#         for i in range(p, len(l)):
#             if l[i] in pi:
#                 l.insert(len(l), l.pop(p))
#                 break
#             elif i+1 == len(l):
#                 p = p + 1

#         stop = stop + 1
#         print(stop)
#     return(l)

# is_dpord(s)
# s=sort_nodes(s)


# mivector=np.empty((0,8))
# row=[]
# for i in train.columns:
 
#     for j in train.columns:    
#         row.append(drv.information_mutual(train[i], train[j]))
#     print(row)
#     np.append(mivector, row, axis=0)


# def val(net):
#     """
#     Depth first search algorithm to validate that
#     a given DAG contains no cycles
#     """
#     # Initialize Depth Markers
#     rec, vis = {}, {}
#     vertices = list(net.nds.values())
#     for v in vertices:
#         rec[v] = False
#         vis[v] = False

#     for v in vertices:
#         print(v.idx)
#         if not net.dfs(v, rec, vis):
#             return(True)
#         else:
#             return(False)
  
# def dfs(net, v, rec, vis):
#     if not vis[v]:
#         vis[v] = True
#         rec[v] = True
        
#     print("foo")
#     for c in v.chl:
#         if (not vis[c]) and net.dfs(c, rec, vis):
#             return(True)
#         elif rec[c]:
#             return(True)
#     else:
#         rec[v] = False
#         return(False)



# # Convert Names from integers to characters
# for d in [train, test]:
#     d.columns = ["G"+str(g) for g in list(d.columns)]
    
# # Learn The Network from data with the greedy algorithm
# bn = BayesianNetwork.from_samples(train)

# # Predict Outcomes for Gene 6
# G6 = test['G6']
# test['G6'] = None
# pred=pd.DataFrame(bn.predict(test.values))
# pred.columns = ["P"+str(g) for g in list(pred.columns)]
# pred = pred.assign(G6 = G6)[['G6', 'P6']]
# pred['Match'] = pred[['G6', 'P6']].apply(
#     lambda x:
#        1 if  x['P6'] == x['G6'] else 0,
#      axis = 1
#     )
# bn.plot()
# plt.show()
# print("Correct Predictions:", pred.Match.sum())



# # Learn The Network from data, with the chow-liu algorithm
# bn = BayesianNetwork.from_samples(train, algorithm='chow-liu')

# # Predict Outcomes for Gene 6

# pred=pd.DataFrame(bn.predict(test.values))
# pred.columns = ["P"+str(g) for g in list(pred.columns)]
# pred = pred.assign(G6 = G6)[['G6', 'P6']]
# pred['Match'] = pred[['G6', 'P6']].apply(
#     lambda x:
#        1 if  x['P6'] == x['G6'] else 0,
#      axis = 1
#     )
# bn.plot()
# plt.show()
# print("Correct Predictions:", pred.Match.sum())

# # Learn The Network from data, with the exact algorithm
# bn = BayesianNetwork.from_samples(train, algorithm='exact')

# # Predict Outcomes for Gene 6
# pred=pd.DataFrame(bn.predict(test.values))
# pred.columns = ["P"+str(g) for g in list(pred.columns)]
# pred = pred.assign(G6 = G6)[['G6', 'P6']]
# pred['Match'] = pred[['G6', 'P6']].apply(
#     lambda x:
#        1 if  x['P6'] == x['G6'] else 0,
#      axis = 1
#     )
# bn.plot()
# plt.show()
# print("Correct Predictions:", pred.Match.sum())

# ## Fix Node Sorting
# l = [5,1,3,2,4]


# for m in l[0:len(l) - 1]:
#     pm = [l[i] for i in range(0, len(l)) if l[i] < m]
#     mi = [i for i in range(0, len(l)) if l[i] == m]
#     print("Outer Loop:", m, mi, pm)
#     for n in range(mi[0]+1, len(l)):
#         if l[n] in pm:
#             mi_new = n
            
#     if mi != mi_new:
#         node = l.pop(mi[0])
#         l.insert(mi_new, node)

# #l = [5,1,3,2,4]
# l = [5,1,3,2,4,7,6,10,9,8]
# new = []
# for m in l:
#     ni = len(new)
#     for n in range(0, len(new)):
#         if m < new[n]:
#             print(n, m, new[n])
#             ni = n
#             break
#     new.insert(ni, m)
#     print(new)





# n = net(size = 8)
# n.add_edge(0,1)
# n.add_edge(1,2)
# n.add_edge(3,1)
# n.add_edge(0,2)
# n.add_edge(4,2)
# n.add_edge(0,4)

# n.calc_cpt(train)


# def fl(l):
#     return [
#         int(l[j]) 
#         if j % len(l) < len(l) - 1 
#         else l[j] for j in range(0, len(l))
#     ]
    

# x=n.export_pom()
# # for n in dep:
            
#         #     # Convert cpt to appropriate format
#         #     cpt = [list(i) for i in list(n.cpt.values)]
            
#         #     # Vector of ID for each parent
#         #     par_id = ["G"+str(i.idx) for i in n.par ]
#         #     print(par_id)
            
#         #     # Validate that all parents have been processed
#         #     for p  in par_id:
#         #         if (not p in topStates.keys()) and (not p in depStates.keys()):
#         #             print("Problem with parent:",p, "of node:",n.idx)
#         #             return [topStates, depStates]
            
#         #     # Get all parents found in the topStates dict
#         #     par = [topStates[i] for i in par_id if i in topStates.keys()]
            
#         #     # Add all parents in the depStates dict
#         #     par = par + [depStates[i] for i in par_id if i in depStates.keys()]
#         #     print("Parents:", par)
#         #     print(cpt)
#         #     print(topStates.keys(), depStates.keys())
#         #     cpt = pm.ConditionalProbabilityTable(
#         #         cpt,
#         #         par 
#         #     )
#         #     depStates["G"+str(n.idx)] = pm.Node(
#         #         cpt, 
#         #         name = "G"+str(n.idx)
#         #     )
        
#         # Assemble and "Bake" model



# cpt = [list(i) for i in list(n.nds[4].cpt.values)]
# G0 = DiscreteDistribution(n.nds[0].cpt['Prob'].to_dict())
# G0 = Node(G0, name="G0")
# G4 = ConditionalProbabilityTable(cpt, [G0])


# cpt = [list(i) for i in list(n.nds[1].cpt.values)]
# G3 =  DiscreteDistribution(n.nds[3].cpt['Prob'].to_dict())
# G3 = Node(G3, name="G3")
# G1 = ConditionalProbabilityTable(cpt, [G3])


# [
#  "G"+str(i)
#   for i in [0,1,3]
#  ]


# #### Try Validating with data generated for a simple 3 node problem
# n = net(size=3)
# n.add_edge(0, 2)
# n.add_edge(1, 2)


# # Assign Probabilities
# cpt0 = n.nds[0].empty_cpt()
# cpt0['Prob'] = [0.5, 0.5]
# n.nds[0].cpt = cpt0

# cpt1= n.nds[1].empty_cpt()
# cpt1['Prob'] = [0.4, 0.6]
# n.nds[1].cpt = cpt1

# cpt2 = n.nds[2].empty_cpt()
# cpt2['Prob'] = [0.9, 0.1, 0.1, 0.9, 0.5, 0.5, 0.9, 0.1]
# n.nds[2].cpt = cpt2

# # Fit a second model on data 1000 random samples from the initial model
# data = n.sample_net(3000)

# m = net(size = 3)
# m.add_edge(0,2)
# m.add_edge(1,2)
# m.calc_cpt(data, alpha = 0.0001)

# # Plot graph of first and second model using pomegranate
# o=BayesianNetwork.from_samples(data)
# o.plot()
# plt.show()


# #### Try running the Monty Hall Simulation
# n = net(size=3, outcomes = (0,1,2))
# n.add_edge(0,2)
# n.add_edge(1,2)

# cpt0 = n.nds[0].empty_cpt()
# cpt0['Prob'] = [1./3, 1./3, 1./3]
# n.nds[0].cpt = cpt0

# cpt1 = n.nds[1].empty_cpt()
# cpt1['Prob'] = [1./3, 1./3, 1./3]
# n.nds[1].cpt = cpt1

# cpt2 = n.nds[2].empty_cpt()
# cpt2['Prob'] = [ 
#     0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
#     0.0, 0.0, 1.0, 0.5, 0.0, 0.5, 1.0, 0.0, 0.0,
#     0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0
# ]
# n.nds[2].cpt = cpt2

# data = n.sample_net(1000)

# m = net(size=3, outcomes = (0,1,2))
# m.add_edge(0,2)
# m.add_edge(1,2)
# m.calc_cpt(data)


# x=tuple([0,1])
# y=tuple(['A', 'B', 'C'])

# z = [y,x]
# combos =[]

# for v in z:
#     if len(combos)  == 0:
#         # Initialize with a list of outcomes
#         combos = [list([i]) for i in v]

#     else:
#         # Iterate over each existing combination
#         # for each existing combination, add one new combo
#         # for each outcome in the next variable
   
#         new = []
#         for c in combos:
#             for o in v:
#                 merge = c + [o]
#                 new.append(merge)
                
#         combos=new
        
    

