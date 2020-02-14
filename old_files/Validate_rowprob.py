import numpy as np
import pandas as pd
import random as rn
#from network import greedy
from network import net
from network import topoSort
from network import sample_net
from network import fl
import pomegranate as pm


# Implelment Function that converts a net object to a pomegranate object
def export_pom(net, by='index'):
    '''
    Returns
    -------
    pomegranate BN Model based on given DAG.
    Assume my "sort" function correctly returns a list where
    children are allways ranked higher than parents
    '''
    s = topoSort(net.export_nds())
    model = pm.BayesianNetwork("DIY_GRN")
    
    
    # Convert Top Level nodes to Discrete distributions
    top = [i for i in s if len(i.par) == 0]
    topStates = {}
    
    for n in top:
        pr = n.cpt['Prob'].to_dict()
        if by == 'index':
            va = n.cpt[n.idx].to_dict()
        else:
            va = n.cpt[n.label].to_dict()
        dist = {}
        for v in va.keys():
            dist[va[v]] = pr[v]
            
        dist=pm.DiscreteDistribution(dist)
        if by == 'index':
            state = pm.Node(dist, name = str(n.idx))
            topStates[str(n.idx)] = state
        else:
            state = pm.Node(dist, name = str(n.label))
            topStates[str(n.label)] = state
            

            
        model.add_state(state)

    # Convert Depent Nodes to Conditional Distributions
    dep = [i for i in s if len(i.par) != 0]
    depStates = {}
   
    for n in dep:
        
        # Convert floats cpt outcome levels to integers if needed
        if isinstance(n.cpt.iloc[0,0], np.int64):
            cpt = [fl(l) for l in n.cpt.values.tolist()]
        
        else:
            cpt = n.cpt.values.tolist()
            
        # Vector of ID for each parent
        if by == 'index':
            par_id = [str(i.idx) for i in n.par ]
        else:
            par_id = [str(i.label) for i in n.par ]
    
        
        # Validate that all parents have been processed
        for p  in par_id:
            if (not p in topStates.keys()) and (not p in depStates.keys()):
                print("Problem with parent:",p, "of node:",n.idx)
                return [topStates, depStates]
        
        # Get all parents found in the topStates dict
        par = [ 
                topStates[i]
                for i in par_id if i in topStates.keys()
        ]
        
        
        # Add all parents in the depStates dict
        par = par + [
            depStates[i]
            for i in par_id if i in depStates.keys()
        ]
    
        cpt = pm.ConditionalProbabilityTable(
            cpt,
            [p.distribution for p in par] 
        )
        
        if by == 'index':
            state =  pm.Node(cpt, name = str(n.idx))
            depStates[str(n.idx)] = state
            
        else:
            state =  pm.Node(cpt, name = str(n.label))
            depStates[str(n.label)] = state
            
        # Add node to model
        model.add_state(state)
        
        # Add edges from parent to this node
        for p in par:
            model.add_edge(p, state)
        
    
    # Assemble and "Bake" model
    model.bake()
    return (topStates, depStates, model)




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

a = net(data=data)
a.add_edge(1,0)
a.del_edge(1,0)
a.calc_cpt(data)


c=a.nds[0].cpt.columns.drop('Prob')

def t(x, c):
    return(tuple(x[c]))

data.apply(t,c=c, axis=1)


pr=[]
for n in a.export_nds():
    c=n.cpt.columns.drop('Prob')
    pr.append(
        n.cpt.set_index(list(c)).loc[t(data.loc[1],c)].values[0]
    )
    

def f(r, nds):
    pr = []
    for n in nds:
        c=n.cpt.columns.drop('Prob')
        pr.append(
            np.log(
                n.cpt.set_index(list(c)).loc[tuple(r[c])].values[0]
                )
            )
    return(np.array(pr).sum())





