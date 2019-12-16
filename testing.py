import numpy as np
import pandas as pd
from network import greedy
from network import net
from matplotlib import pyplot as plt
from pyitlib import discrete_random_variable as drv
from pomegranate import *

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

n=net(3)
n.add_edge(0,1)
n.add_edge(2,1)

n.nds[0].cpt=pd.DataFrame({0:[0,1], 'Prob':[0.5, 0.5]})
n.nds[2].cpt=pd.DataFrame({0:[0,1], 'Prob':[0.6, 0.4]})

n.nds[1].cpt=pd.DataFrame({
    0:[0,0,0,0,1,1,1,1],
    2:[0,0,1,1,0,0,1,1],
    1:[0,1,0,1,0,1,0,1],
    'Prob':[0.2, 0.8, 0.8, 0.2, 0.8, 0.2, 0.2, 0.8]})



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
        
    

