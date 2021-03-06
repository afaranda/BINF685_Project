import pandas as pd
import numpy as np
import pomegranate as pm
import itertools as it
from math import log
from copy import deepcopy as dc

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


class node:
    def __init__(self, 
                idx, 
                name=None, 
                parents=[], 
                children = [], 
                outcomes=(0,1)):

        self.idx = idx
        self.par = parents
        self.chl = children
        self.ocm = outcomes
        self.cpt = None
        
    def old_cpt_index(self):
        """
        Calculate total CPT size based on number of parents, 
        outcomes.  Build empty table with columns for each parent's
        possible outcomes and this nodes possible outcome.  Add a 
        column to store the probability for each record.
        """
        var = self.parent_idx() + [self.idx]
        seed = []
        if len(var) == 1:
            idx = pd.Index(list(self.ocm)).set_names(var[0])
            return(idx)

        for p in range(0,len(var)):
            if len(seed) == 0:
                seed.append(list(self.ocm))
            else:
                for s in range(0,len(seed)):
                    old = []
                    for k in range(0, len(self.ocm)):
                        old = old + seed[s]
                        seed[s] = old
                l = len(seed[0]) // len(self.ocm)
                new = []
                for o in self.ocm:
                    for j in range(0, l):
                        new.append(o)
                seed.insert(0,new)
        tpl = list(zip(*seed))
        idx = pd.MultiIndex.from_tuples(tpl, names = var)
        return(idx)
    
    def cpt_index(self):
        """
        Calculate total CPT size based on number of parents, 
        outcomes.  Build empty table with columns for each parent's
        possible outcomes and this nodes possible outcome.  Add a 
        column to store the probability for each record.
        """
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
            idx = pd.MultiIndex.from_tuples(combos, names = [v.idx for v in var])
            return(idx)
    
    def parent_idx(self):
        idx = []
        for i in self.par:
            idx.append(i.idx)
        return(idx)
    
    def add_parents(self, parents):
        new = []
        for p in parents:
            if p in self.chl:
                print("Cant have a parent that is also a child")
                return(False)

            elif p in self.par:
                print("Already has this child")
                return(False)
            else:
                new.append(p)
                self.par = self.par + new
                return(True)
    
    def add_children(self, children):
        new = []
        for c in children:
            if c in self.par:
                print("Cant have a child that is also a parent")
                return(False)
            elif c in self.chl:
                print("Already has this parent")
                return(False)
            else:
                new.append(c)
                self.chl = self.chl + new
                return(True)

    def del_parents(self, parents):
        x = set(self.par).difference(parents)
        self.par = list(x)
        

    def del_children(self, children):
        x = set(self.chl).difference(children)
        self.chl = list(x)

    def old_node_probs(self, data, alpha = 0.001):
        """
        Fill CPT with corresponding observational counts
        """
        index = self.cpt_index()
        var = self.parent_idx() + [self.idx]
        par = self.parent_idx()
        
        # Copy data, add a column to calcuate conditional probabilities
        cpt = data[var].copy()
        cpt['Prob'] = 0

        # # Flatten cpt to counts over observed outcomes
        cpt = cpt.groupby(var).count().copy()

        # For possible, but unobserved outcomes, convert NaN to 0
        cpt = cpt.reindex(index).fillna(0)
        
        cpt = cpt.reset_index()
  
        # Use a rather complicated lambda to calculate frequencies
        if len(par) == 0:
            cpt['Prob'] = cpt.transform(
            lambda x: (x + alpha) / (x.sum() +(x.count()*alpha)))['Prob']
        else:
            cpt['Prob'] = cpt.groupby(par).transform(
                lambda x: (x + alpha) / (x.sum() +(x.count()*alpha))
            )['Prob']
            
        # # Dump the dummy group 
        # if par[0] == 'dummy':
        #     self.cpt = cpt.drop(columns = ['dummy'])
        self.cpt = cpt
        
    def node_probs(self, data, alpha = 0.001):
        """
        Fill CPT with corresponding observational counts
        """
        index = self.cpt_index()
        var = self.parent_idx() + [self.idx]
        par = self.parent_idx()
        
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
    
    def empty_cpt(self):
        """
        Calculate total CPT size based on number of parents, 
        outcomes.  Build empty table with columns for each parent's
        possible outcomes and this nodes possible outcome.  Add a 
        column to store the probability for each record.
        """
        df = pd.DataFrame(
            index=self.cpt_index(),
            columns=['Prob']
        )
        df['Prob'] = 0.0
        return df.reset_index()

        

class net:
    def __init__(self, size=5, outcomes = (0,1)):
        self.siz = size
        self.nds = {}
        for i in range(0, self.siz):
            self.nds[i] = node(i, outcomes=outcomes)


    def reset(self):
        self.siz = self.siz
        self.nds = {}
        for i in range(0, self.siz):
            self.nds[i] = node(i)
        
    def add_edge(self, p_idx, c_idx):
        if p_idx != c_idx:
            if self.nds[p_idx].add_children([self.nds[c_idx]]):
                self.nds[c_idx].add_parents([self.nds[p_idx]])
        
    
    def del_edge(self, p_idx, c_idx):
        self.nds[p_idx].del_children([self.nds[c_idx]])
        self.nds[c_idx].del_parents([self.nds[p_idx]])
        
    
    def rev_edge(self, p_idx, c_idx):
        if p_idx != c_idx:
            if c_idx in [j.idx for j in self.nds[p_idx].chl]:
                print("is_child")
                self.del_edge(p_idx, c_idx)
                self.add_edge(c_idx, p_idx)

    def val(self):
        """
        Depth first search algorithm to validate that
        a given DAG contains no cycles
        """
        # Initialize Depth Markers
        rec, vis = {}, {}
        vertices = list(self.nds.values())
        for v in vertices:
            rec[v] = False
            vis[v] = False

        for v in vertices:
            if not self.dfs(v, rec, vis):
                return(True)
            else:
                return(False)
      
    def dfs(self, v, rec, vis):
        if not vis[v]:
            vis[v] = True
            rec[v] = True
        
        for c in v.chl:
            if (not vis[c]) and self.dfs(c, rec, vis):
                return(True)
            elif rec[c]:
                return(True)
        else:
            rec[v] = False
            return(False)
        
    def calc_cpt(self, data, alpha=0.001):
        """
        Calculate CPT probabilities for all nodes in the
        network given the data.
        """
        for k in self.nds.keys():
            n = self.nds[k]
            n.node_probs(data = data, alpha = alpha)

    def is_dpord(self, l):
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

    def sort_nodes(self, l):
        stop = 0
        l = l #list(self.nds.values())
        p = 0
        while (not self.is_dpord(l)) and stop <1000:
            pi = l[p].par
            for i in range(p, len(l)):
                if l[i] in pi:
                    l.insert(len(l), l.pop(p))
                    break
                elif i+1 == len(l):
                    p = p + 1

            stop = stop + 1
        return(l)
    
    def new_sort_nodes(self, l):
        stop = 0
        l = l #list(self.nds.values())
        p = 0
        while (not self.is_dpord(l)) and stop <1000:
            pi = l[p].par
            for i in range(p, len(l)):
                if l[i] in pi:
                    l.insert(len(l), l.pop(p))
                    break
                elif i+1 == len(l):
                    p = p + 1

            stop = stop + 1
        return(l)
    
    def calc_prob(self, query):
        l = list(self.nds.values())
        if(len(query) != self.siz):
            print('Invalid Query')
            return False
        prob = []
        # Assume that the variables in the query correspond
        # to their position eg query[0] is Var 0, query[1] = Var 1 etc. . .
        
        for i in l:
            var = i.cpt.columns.drop('Prob')
            val = [query[j] for j in var ]
            c = i.cpt.copy()
            for j in range(0, len(val)):
                c = c[c[var[j]] == val[j]]
            prob.append(c['Prob'].values[0])
        return(np.prod(prob))

    def sample_net(self, n):
        """
        Using a simple 'prior sampling' approach, generate 
        n samples from this 'net' object.
        """
        df = []
        x = 0
        while x < n:
            l = self.sort_nodes(list(self.nds.values()))
            row = {}
            for i in l:
                var = i.cpt.columns.drop('Prob')
                prior = {k:row[k] for k in row.keys() if k in var}
                c = i.cpt.copy()
                c = c.sort_values(by=list(c.columns))

                # Subset cpt
                #print(prior)
                for j in prior.keys():
                        c = c[c[j] == prior[j]]

                s = np.random.uniform()
                #print("NODE:", i.idx, "Draw:", s)
                #print(c)
                #print(c.loc[c[i.idx] == 0, 'Prob'])
                
                if s < c.loc[c[i.idx] == 0, 'Prob'].values[0]:
                    row[i.idx] = 0
                else:
                    row[i.idx] = 1
                #print(row.keys(), row.values())
            df.append([row[m] for m in sorted(row.keys())])
            x = x + 1
        df = pd.DataFrame(df)
        return(df)
    
    def gibbs(self, n):
        # Randomly initialize values for all possible variables
        val = []
        samples={}
        for k in self.nds.keys():
            nd=self.nds[k]
            val.append(np.random.choice(nd.ocm))
            samples[k]=[]
            
        l = self.sort_nodes(list(self.nds.values()))
        #print(val)
        # Generate "n" samples by rotating through each variable
        while n > 0:          
            pos = n % len(self.nds)
            node= l[pos]        
            par = self.nds[pos].parent_idx()
            ocm = self.nds[pos].ocm
            cpt = self.nds[pos].cpt.copy()
            if len(par) > 0:
                idx = pd.MultiIndex.from_tuples(
                    [tuple([val[j] for j in par] + [o]) for o in ocm],
                    names = par + [pos]
                )
            else:
                idx = pd.Index(ocm).set_names(pos)

            oc = cpt.set_index(par + [pos]).loc[idx].reset_index()[pos].values
            pr = cpt.set_index(par + [pos]).loc[idx].reset_index()['Prob'].values
            new_value = np.random.choice(oc, p=pr)
            val[pos] = new_value
            for k in samples.keys():
                samples[k].append(val[k])
            n = n - 1
            
        return(pd.DataFrame(samples))
        
       
    def score_net(self, data):
        """
        Score this network against a data set
        """
        # Flatten data set to a table of unique rows with corresponding
        # Frequencies. 

        #df = data.copy()
        #gr = [i for i in df.columns]
        #df['Count'] = 0
        #df = df.groupby(gr, as_index=False).count()
        #df['Model'] = df[gr].apply(self.calc_prob, axis=1)
        #return(df.Model.transform(log).sum())
        
        # Reformat data for pomegranate scoring
        data.columns = ["G" + str(i) for i in data.columns]
        net = self.export_pom()[2]
        pscore = net.log_probability(data).sum()
        return pscore
        
    
    def export_dag(self):
        dag = np.zeros([self.siz, self.siz], int)
        for k in self.nds.keys():
            for c in self.nds[k].chl:
                dag[self.nds[k].idx, c.idx] = 1
        return(dag)
    
    def import_dag(self, dag):
        dg = dag
        self.reset()
        if not isinstance(dg, np.ndarray):
            return None
        if not dg.shape[0] == dg.shape[1]:
            return None
        
        for i in range(0, len(dg)):
            for j in range(0, len(dg)):
                if dg[i, j] == 1:
                    self.add_edge(i, j)

    def export_pom(self):
        '''
        Returns
        -------
        pomegranate BN Model based on given DAG.
        Assume my "sort" function correctly returns a list where
        children are allways ranked higher than parents
        '''
        s = self.sort_nodes( l = list(self.nds.values()))
        model = pm.BayesianNetwork("DIY_GRN")
        
        
        # Convert Top Level nodes to Discrete distributions
        top = [i for i in s if len(i.par) == 0]
        topStates = {}
        
        for n in top:
            pr = n.cpt['Prob'].to_dict()
            va = n.cpt[n.idx].to_dict()
            dist = {}
            for v in va.keys():
                dist[va[v]] = pr[v]
            
            dist=pm.DiscreteDistribution(dist)
            state = pm.Node(dist, name = "G"+str(n.idx))
                
            
            topStates["G"+str(n.idx)] = state
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
            par_id = ["G"+str(i.idx) for i in n.par ]

            
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
            
            state =  pm.Node(cpt, name = "G"+str(n.idx))
            depStates["G"+str(n.idx)] = state
            
            # Add node to model
            model.add_state(state)
            
            # Add edges from parent to this node
            for p in par:
                model.add_edge(p, state)
            
        
        # Assemble and "Bake" model
        model.bake()
        return (topStates, depStates, model)
                    
    def print_nodes(self):
        for k in self.nds.keys():
            print(
                "Node:", k,
                "Parents:", self.nds[k].parent_idx(),
                "Children:",[j.idx for j in self.nds[k].chl]
            )


class greedy():
    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            return None
        else:
            self.data = data
            self.net = None
            self.scores = {}
            self.samples = None
            
    def train(self, iterations = 10, maxmiss=10):
        """
        Train using Recommended Greedy Algorithm
        """
        self.net = net(size = len(self.data.columns))
        self.net.calc_cpt(self.data)
        scores = {
            'Iteration':[],
            'Network':[],
            'Score':[]
        }
        score_check = maxmiss
        niter = iterations 
        nodes = [i for i in self.data.columns]
        best = self.net.score_net(self.data)
    
        #print("START LOOP")
        while score_check > 0 and niter > 0:
            n = net(size = len(self.data.columns))
            n.import_dag(self.net.export_dag())
            
            ops = [n.add_edge, n.del_edge, n.rev_edge]
            for f in ops:
                edge = np.random.choice(nodes, size = 2, replace=False )
                f(edge[0], edge[1])
              
            if n.val():
                n.calc_cpt(self.data, alpha = 0.001)
                score = n.score_net(self.data)
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
        self.samples = self.net.sample_net(2000)
        
    def get_prob(self, query, target):
        qr = query.copy()
        tg = target
        par = self.net.nds[tg].parent_idx()
        ocm = self.net.nds[tg].ocm
        cpt = self.net.nds[tg].cpt.copy()
        if len(par) > 0:
            idx = pd.MultiIndex.from_tuples(
                [tuple([qr[j] for j in par + [tg]])],
                names = par + [tg]
            )
        else:
            idx = pd.Index([qr[tg]]).set_names(tg)
                
        pr = cpt.set_index(par + [tg]).loc[idx].values[0][0] 
        return(pr)

    def call_target(self, query, target):
        """
        Predict the status of a target gene (0 or 1) given the status of
        other genes in the network.  This function retrieves the CPT for
        the target gene and returns the most likely outcome based on
        probabilities estimated from the training set.
        """
        qr = query.copy()
        tg = target
        par = self.net.nds[tg].parent_idx()
        ocm = self.net.nds[tg].ocm
        cpt = self.net.nds[tg].cpt.copy()
        if len(par) > 0:
            idx = pd.MultiIndex.from_tuples(
                [tuple([qr[j] for j in par] + [i]) for i in ocm ],
                names = par + [tg]
            )
        else:
            idx = pd.Index(ocm).set_names(tg)
        call = cpt.set_index(par + [tg]).loc[idx]
        call = call.reset_index()
        call = call[tg].loc[call['Prob'] == max(call.Prob)].values[0]
        return(call)


    def predict(self, query, target, alpha = 0):
        """
        Predict the status of a target gene (0 or 1) given the status of
        one or more query genes, using samples generated via direct sampling.
        """
        qr = query.copy()
        ocm = self.net.nds[target].ocm
        cpt = self.samples[qr.index].copy()
        cpt['Prob'] = 0

        # # Flatten cpt to counts over observed outcomes
        cpt = cpt.groupby(list(qr.index)).count().copy()
        if len(qr.drop(target).index) > 0:
            idx = pd.MultiIndex.from_tuples(
                [tuple([qr[j] for j in qr.drop(target).index] + [i]) for i in ocm],
                names = list(qr.index)
            )
        else:
            idx =pd.Index(ocm).set_names(target)

        
        # For possible, but unobserved outcomes, convert NaN to 0
        cpt = cpt.reindex(idx).fillna(0)
        
        cpt = cpt.reset_index()
        
        # Use a rather complicated lambda to calculate frequencies
        #if len(qr.drop(target).index) == 0:
        #    cpt['Prob'] = cpt.transform(
        #    lambda x: (x + alpha) / (x.sum() +(x.count()*alpha)))['Prob']
        #else:
        #    cpt['Prob'] = cpt.groupby(list(qr.drop(target).index)).transform(
        #        lambda x: (x + alpha) / (x.sum() +(x.count()*alpha))
        #    )['Prob']
        

        print(cpt)
        return(cpt.loc[cpt['Prob'] == max(cpt['Prob']), target])




        
        #call = cpt.set_index(par + [tg]).loc[idx]
        #call = call.reset_index()
        #call = call[tg].loc[call['Prob'] == max(call.Prob)].values[0]
        #return(call)

    def test_accuracy(self, test, target = 6):
        data=test
        data['Pred_Prob'] = data.apply(self.get_prob, args=(target,), axis=1)
        data['Predict'] = data.apply(self.call_target, args=(target,), axis=1)
        data['Correct'] = data.apply(lambda x: x[0] == x['Predict'], axis=1)
        return(data)
        
        
            


        
        

        
    









            



            


          
            
 
        
        

    
  
    


        

                    
                 




             

            




        
    

        
        



            

        
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
