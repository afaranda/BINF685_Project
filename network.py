import pandas as pd
import numpy as np
import random as rn
import pomegranate as pm
#import itertools as it
#from math import log
#from copy import deepcopy as dc

#############################################################################
#                                                                           #
#                           Module Level Functions                          #
#                                                                           #
#############################################################################

def fl(l):
    '''
    Helper Function that converts CPT Variable-Outcomes from "float64"
    to "int" after extraction via "n.cpt.values.tolist()"
    '''
    return [
        int(l[j])
        if j % len(l) < len(l) - 1 
        else l[j] for j in range(0, len(l))
    ]

def dfs(node, vis):
    
    vis[node] = "G"
    
    for p in node.par:
        
        if vis[p] == "B":
            continue
        
        if vis[p] == "G":
            #print("Cycle Detected", p.idx)
            return True
        
        if dfs(p, vis):
            #print("From", node.idx, "Cycle Detected Recursively", p.idx)
            return(True)
  
    vis[node]="B"
    return(False)

def val(net):
    vis = {i:'W' for i in net.values()}
    
    for node in vis.keys():
        if vis[node] == "W":
            if dfs(node, vis):
                return True
        else:
            continue
    return False

def topoSort(net):
    vis = []
    stack = []
    
    for node in net:
        if not node in vis:
            topo_dfs(node, vis, stack)
            
    return stack
    
def topo_dfs(node, vis, stack):
    if not node in vis:
        vis.append(node)
    
    for p in node.par:
        if p in stack:
            continue

        else:
            topo_dfs(p, vis, stack)
    stack.append(node)

def is_dpord(l):
    
    """
    Function to check a node list is ordered parent to child
    """
    for i in range(0, len(l) -1):
        pi = l[i].par
        for j in pi:
            for k in range(i+1, len(l)):
                if j == l[k]:
                    return False
                else:
                    continue
    return True



def sample_net(net, r=10):
    df = []
    x = 0
    l = topoSort(net.export_nds())
    if net.by == 'index':
        while x < r:
            row = {}
            for m in l:
                if len(m.par) > 1:
                    gr = m.cpt.groupby([j.idx for j in m.par]).groups
                    po = tuple(
                        [row[k] for k in [j.idx for j in m.par] ]
                    )
                    row[m.idx] = rn.choices(
                            m.cpt.loc[gr[po]][m.idx].values,
                            m.cpt.loc[gr[po]]['Prob'].values
                    )[0]
                    
                elif len(m.par) > 0:
                    gr = m.cpt.groupby([j.idx for j in m.par]).groups
                    po = row[m.par[0].idx]
                    row[m.idx] = rn.choices(
                            m.cpt.loc[gr[po]][m.idx].values,
                            m.cpt.loc[gr[po]]['Prob'].values
                    )[0]
                    
                else:
                    row[m.idx] = rn.choices(
                        m.cpt[m.idx],
                        m.cpt['Prob']
                    )[0]
            df.append(
                [row[k] for k in sorted(row.keys())]  
            )
            x=x+1
        return pd.DataFrame(df)
    
    elif net.by == 'label':
        while x < r:
            row = {}
            for m in l:
                if len(m.par) > 1:
                    gr = m.cpt.groupby([j.label for j in m.par]).groups
                    po = tuple(
                        [row[k] for k in [j.label for j in m.par] ]
                    )
                    row[m.label] = rn.choices(
                            m.cpt.loc[gr[po]][m.label].values,
                            m.cpt.loc[gr[po]]['Prob'].values
                    )[0]
                    
                elif len(m.par) > 0:
                    gr = m.cpt.groupby([j.label for j in m.par]).groups
                    po = row[m.par[0].label]
                    row[m.label] = rn.choices(
                            m.cpt.loc[gr[po]][m.label].values,
                            m.cpt.loc[gr[po]]['Prob'].values
                    )[0]
                    
                else:
                    row[m.label] = rn.choices(
                        m.cpt[m.label],
                        m.cpt['Prob']
                    )[0]
                    
            df.append([row[k] for k in
                [{i.idx:i.label for i in l}[j] 
                for j in sorted(
                        {i.idx:i.label for i in l}.keys()
                    )]
                ]) 
            x=x+1
           
        return pd.DataFrame(df, columns=[{i.idx:i.label for i in l}[j] 
                for j in sorted({i.idx:i.label for i in l}.keys())])
    
def export_pom(net, by='index'):
    '''
    Returns
    -------
    pomegranate BN Model based on given DAG.
    Assume my "sort" function correctly returns a list where
    children are allways ranked higher than parents. If Pommegranate is used
    to estimate model likelihood, all outcomes must be of the same data type. 
    Either All int or all string. 
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
    return (model)


def score_net(net, data):
    lp = data.apply(data_prob, nds=net.export_nds(), axis=1)
    return(sum(lp))
    

def data_prob(r, nds):
    '''
    This function is intended to be used with the 'apply' method for a pandas
    data frame, containing the data to be learned

    Parameters
    ----------
    r : Pandas Series
        Row from a data set, with labels corresponding to node labels in nds
    nds : list of nodes
        list of nodes from a bayesian network.  each node stores a conditional
        probability table

    Returns
    -------
    log probability of observation 'r'

    '''
    pr = []
    for n in nds:
        c=n.cpt.columns.drop('Prob')
        pr.append(
            np.log(
                n.cpt.set_index(list(c)).loc[tuple(r[c])].values[0]
                )
            )
    return(np.array(pr).sum())

def score_pom(model, data):
    v = [i.name for i in model.states]
    return model.log_probability(data[v]).sum()

def edges(net):
    
    edges = set()
    if net.by == 'index':
        for i in net.nds.keys():
            for j in net.nds[i].par:
                edges.add((j.idx, i))
                
    elif net.by == 'label':
        for i in net.nds.keys():
            for j in net.nds[i].par:
                edges.add((j.label, i))
                
    else:
        print('Invalid Network')
        return None
    return edges

       
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

        
class net:
    def __init__(self, size=None, outcomes=None, data=None):
        self.siz = None
        self.nds = {}
        self.by =None
        
        if not (size == None and outcomes == None) and data == None:
            self.fill_uniform(size, outcomes)
            self.by = 'index'
            
        elif isinstance(data, pd.DataFrame):
            self.fill_data(data, by='label')
            self.by = 'label'
    
    def fill_uniform(self, size=5, outcomes=(0,1)):
        self.siz = size
        self.nds = {}
        for i in range(0, self.siz):
            self.nds[i] = node(i, label=str(i), outcomes = outcomes)
        return self
    
    def fill_data(self, data, by='index'):
        self.siz = data.shape[1]
        if by == 'index':
            for i in range(0, self.siz):
                self.nds[i] = node(
                    i, label = data.columns[i], 
                    outcomes = data[data.columns[i]].unique().tolist()
                )
        elif by == 'label':
            for i in range(0, self.siz):
                self.nds[data.columns[i]] = node(
                    i, label = data.columns[i], 
                    outcomes = data[data.columns[i]].unique().tolist()
                )
        else:
            print("Can only use 'index' or 'label' as node keys")
            return None
            
    def reset(self):
        for i in self.nds.keys():
    
            self.nds[i] = node(
                self.nds[i].idx, label = self.nds[i].label, 
                outcomes = self.nds[i].ocm
            )

        
    def add_edge(self, p_idx, c_idx):
        print("parent:", p_idx, "child:", c_idx)
        if p_idx != c_idx:
            self.nds[c_idx].add_parents([self.nds[p_idx]])
        
    def del_edge(self, p_idx, c_idx):
        self.nds[c_idx].del_parents([self.nds[p_idx]])
        
    def rev_edge(self, p_idx, c_idx):
        if p_idx != c_idx:
            self.del_edge(p_idx, c_idx)
            self.add_edge(c_idx, p_idx)

    def acyclic(self):
        return not val(self.nds)
          
    def calc_cpt(self, data, alpha=0.001):
        """
        Calculate CPT probabilities for all nodes in the
        network given the data. If dolumn names in the data do not match
        the index, invoke with: " by='label' " to 
        """
        for k in self.nds.keys():
            n = self.nds[k]
            n.node_probs(data = data, alpha = alpha, by=self.by)
             
    def export_nds(self):
        return list(self.nds.values())
    
    def export_dag(self):
        dag = np.zeros([self.siz, self.siz], int)
        for k in self.nds.keys():
            for p in self.nds[k].par:
                dag[p.idx, self.nds[k].idx] = 1
        return(dag)
    
    def import_dag(self, dag):
        dg = dag
        ndi = {self.nds[i].idx:i for i in self.nds.keys()}
        self.reset()
        if not isinstance(dg, np.ndarray):
            return None
        if not dg.shape[0] == dg.shape[1]:
            return None
        
        for i in range(0, len(dg)):
            for j in range(0, len(dg)):
                if dg[i, j] == 1:
                    self.add_edge(ndi[i], ndi[j])
                    
    def print_nodes(self):
        for k in self.nds.values():
            print(
                "Node:", k.idx,
                "Parents:", k.parent_idx(),
                "Children:", [j.idx for j in self.nds.values() if k in j.par]
            )


# class greedy():
#     def __init__(self, data):
#         if not isinstance(data, pd.DataFrame):
#             return None
#         else:
#             self.data = data
#             self.net = None
#             self.scores = {}
#             self.samples = None
            
#     def train(self, iterations = 10, maxmiss=10):
#         """
#         Train using Recommended Greedy Algorithm
#         """
#         self.net = net(size = len(self.data.columns))
#         self.net.calc_cpt(self.data)
#         scores = {
#             'Iteration':[],
#             'Network':[],
#             'Score':[]
#         }
#         score_check = maxmiss
#         niter = iterations 
#         nodes = [i for i in self.data.columns]
#         best = self.net.score_net(self.data)
    
#         #print("START LOOP")
#         while score_check > 0 and niter > 0:
#             n = net(size = len(self.data.columns))
#             n.import_dag(self.net.export_dag())
            
#             ops = [n.add_edge, n.del_edge, n.rev_edge]
#             for f in ops:
#                 edge = np.random.choice(nodes, size = 2, replace=False )
#                 f(edge[0], edge[1])
              
#             if n.val():
#                 n.calc_cpt(self.data, alpha = 0.001)
#                 score = n.score_net(self.data)
#                 scores['Iteration'].append(iterations - niter)
#                 scores['Network'].append(n)
#                 scores['Score'].append(score)
#                 #print(best, score, niter, score_check)
#                 if score > best:
#                     self.net = n
#                     best = score
#                     niter = niter - 1
#                     score_check = maxmiss
#                     continue
#                 else:
#                     score_check = score_check - 1
#                     niter = niter -1
#                     continue
#             else:
#                 niter = niter - 1
#                 continue
#         self.scores = scores
#         self.samples = self.net.sample_net(2000)
        
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
        
        
            


        
        

        
    









            



            


          
            
 
        
        

    
  
    


        

                    
                 




             

            




        
    

        
        



            

        
################################################################################
#                                                                              #
#            Validation Section: Comment out when no longer needed             #
#                                                                              #
################################################################################

# d = dag(10)
# print(d.dg)
# full_net = [
#   (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), 
#   (0, 6), (0, 7), (0, 8), (0, 9), (1, 2), 
#   (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), 
#   (1, 8), (1, 9), (2, 3), (2, 4), (2, 5), 
#   (2, 6), (2, 7), (2, 8), (2, 9), (3, 4), 
#   (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
#   (4, 5), (4, 6), (4, 7), (4, 8), (4, 9),
#   (5, 6), (5, 7), (5, 8), (5, 9), (6, 7), 
#   (6, 8), (6, 9), (7, 8), (7, 9), (8, 9)
# ]

# for e in full_net:
#     p, c = e
#     d.add(p, c)


# print(d.val())

# d.add(4, 2)
# print(d.val())

# net = [(0, 1), (2, 1), (3, 1), (1, 4), (3, 4), (4, 5), (5, 3)]
# d = dag(6)
# for e in net:
#     p, c = e
#     d.add(p, c)

# print(d.val())

# def node_probs(self, data, alpha = 0):
#         """
#         Fill CPT with corresponding observational counts
#         """
#         cpt = self.cpt
#         var = [c for c in cpt.columns if not c == 'Prob']
#         par = self.parent_idx()
#         if len(par) == 0:
#             par = ['dummy']
#             cpt['dummy'] = 0

        
#         # For each case listed in the cpt, count the number of
#         # corresponding rows in the data

#         # Store the sum of observations for each case
#         for i in cpt.T.columns:
#             s = 0
#             for j in data.T.columns:
#                 m = True
#                 for v in var:
#                     x = cpt.loc[[i],[v]].values[0][0]
#                     y = data.loc[[j], [v]].values[0][0]
#                     if x != y:
#                         m = False
#                         break
#                 if m:
#                     s = s + 1   
#             cpt.loc[[i], ['Prob']]=s
#         # Use a rather complicated lambda to calculate frequencies
#         cpt['Prob'] = cpt.groupby(par).transform(
#             lambda x: (x + alpha) / (x.sum() +(x.count()*alpha))
#         )['Prob']
        
#         # Dump the dummy group 
#         if par[0] == 'dummy':
#             SELF.CPT = cpt.drop(columns = ['dummy'])


# # Old Gibbs Sampler Code
    # def gibbs(self, n):
    #     # Randomly initialize values for all possible variables
    #     val = []
    #     samples={}
    #     for k in self.nds.keys():
    #         nd=self.nds[k]
    #         val.append(np.random.choice(nd.ocm))
    #         samples[k]=[]
            
    #     l = self.sort_nodes(list(self.nds.values()))
    #     #print(val)
    #     # Generate "n" samples by rotating through each variable
    #     while n > 0:          
    #         pos = n % len(self.nds)
    #         node= l[pos]        
    #         par = self.nds[pos].parent_idx()
    #         ocm = self.nds[pos].ocm
    #         cpt = self.nds[pos].cpt.copy()
    #         if len(par) > 0:
    #             idx = pd.MultiIndex.from_tuples(
    #                 [tuple([val[j] for j in par] + [o]) for o in ocm],
    #                 names = par + [pos]
    #             )
    #         else:
    #             idx = pd.Index(ocm).set_names(pos)

    #         oc = cpt.set_index(par + [pos]).loc[idx].reset_index()[pos].values
    #         pr = cpt.set_index(par + [pos]).loc[idx].reset_index()['Prob'].values
    #         new_value = np.random.choice(oc, p=pr)
    #         val[pos] = new_value
    #         for k in samples.keys():
    #             samples[k].append(val[k])
    #         n = n - 1
            
    #     return(pd.DataFrame(samples))