import numpy as np
import pandas as pd
import random as rn
from network import export_pom
from network import net





######### Validate Sampler on Randomly Generated Data:
data={
      'G2':[rn.choices(['A','B'], weights=[0.5, 0.5])[0] for i in range(0,2000)] 
      }


# Using this function, G3 is dependent on G1 -- independent of G2
def ofun(a):
    if a=='A':
        return rn.choices(['A','B'], weights=[0.9,0.1])[0]
    else:
        return rn.choices(['A','B'], weights=[0.1,0.9])[0]

def pfun(a):
    if a=='A':
        return rn.choices(['A','B'], weights=[0.9,0.1])[0]
    elif a=='B':
        return rn.choices(['A','B'], weights=[0.1,0.9])[0]
    else:
        return rn.choices(
            ['A','B','C'],
            weights=[1./9, 1./9, 7./9]
        )[0]
    

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

data['G3'] = [
    rn.choices(
        ['A', 'B', 'C'], 
        weights=[0.15, 0.3, 0.55])[0] 
    for i in range(0,2000)
] 

data['G4'] = [
    pfun(data['G3'][i])
    for i in range(0,len(data['G3']))
]


data = pd.DataFrame(data)
data = data[['G1', 'G2', 'G3', 'G4']]

a = net(data=data)
a.add_edge(1,0)
a.add_edge(2,3)
a.calc_cpt(data)
m = export_pom(a, by='label')

v = [i.name for i in m.states]


# Function that calculates the probability of a row given a set of nodes
def f(r, nds):
    pr = []
    for n in nds:
        c=n.cpt.columns.drop('Prob')
        pr.append(
            #np.log(
                n.cpt.set_index(list(c)).loc[tuple(r[c])].values[0]
            #    )
            )
    return(np.array(pr).prod())


# Visually, these calculations generate the same results. 
data['PomProb'] = m.probability(data[v])
data['MyProb'] = data[v].apply(f, nds=a.export_nds(), axis=1)

# Difference is rounded to 10 decimal places since we don't care about 
# floating point error. 
data['diff'] = round(data['PomProb'] - data['MyProb'], 10)

# NEXT:  Validate Pom Export For a Bunch for Random Networks











