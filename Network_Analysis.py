import pandas as pd
from Learners import greedy, CASGMM, CASMOD, CASJNB
from Metrics import kldv, ac, sn, sp
from network import net, export_pom
from generate_networks import tn1, ds1, tn2, ds2, tn3, ds3, tn4, ds4
import networkx as nx
from pomegranate import BayesianNetwork as bn


######## Use ds3 to Demonstrate  Sampler accuracy

ln3 = net(data=ds3)                  # Define identical network to tn3
ln3.add_edge('Icy', 'Watson')
ln3.add_edge('Icy', 'Holmes')
ln3.add_edge('Holmes', 'Ambulance')
ln3.add_edge('Watson', 'Ambulance')
ln3.calc_cpt(                        # Estimate CPT's from first 100 samples
    ds3.loc[0:100,], alpha=0.00001
)     

####### Compare original CPT's to sample based estimates
kl={l:kldv(tn3.nds[l].cpt, ln3.nds[l].cpt) 
 for l in [i.label for i in tn3.export_nds()]} 

pd.DataFrame({
    'Table':list(kl.keys()),
    'KL_Divergence':list(kl.values())
    }).to_csv("results/IcyRoads_KL_Divergence.csv", index=False)

for l in [i.label for i in tn3.export_nds()]:
    ofn = 'results/Defined_'+l+'_CPT.csv'
    lfn = 'results/Sampled_'+l+'_CPT.csv'
    tn3.nds[l].cpt.to_csv(ofn, index=False)
    ln3.nds[l].cpt.to_csv(ofn, index=False)

### Run Learners on Network #1: 20 independent nodes

r = False
for i in range(1,6):
    grd = greedy(ds1)
    grd.train(iterations=300, maxmiss = 15)
    
    cgm = CASGMM(ds1)
    cgm.train(iterations=300, maxmiss = 15)
    
    cmd = CASMOD(ds1)
    cmd.train(iterations=300, maxmiss = 15)
    
    cjn = CASJNB(ds1)
    cjn.train(iterations=300, maxmiss = 15)

    if not r:
        t_res = pd.DataFrame(
            grd.scores).assign(Trial = i, Learner="Greedy")
        
        t_res.append(
            pd.DataFrame(cgm.scores).assign(Trial = i, Learner = 'CASGMM')
        )
        
        t_res.append(
            pd.DataFrame(cgs.scores).assign(Trial = i, Learner = 'CASMOD')
        )    
        
         t_res.append(
            pd.DataFrame(cjn.scores).assign(Trial = i, Learner = 'CASMOD')
        )    
        
        r = True
    else:
        t_res = t_res.append(pd.DataFrame(g.scores).assign(Trial = i))

        # Print out The top network from each trial
    print("Most Recent Best Greedy Network for Trial {0}".format(i))
    g.net.print_nodes()
    print(
    	"Best Score observed in Trial {0}: {1}".format(
    	i, max(t_res['Score'].loc[t_res['Trial'] == i])
    	)
    )
    print()        
        
# Print Summary Results
