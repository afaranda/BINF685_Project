import pandas as pd
from Learners import greedy, CASGMM, CASMOD, CASJNB
from Metrics import kldv, accuracy, edge_hits
from network import net, export_pom, plot_net
from generate_networks import tn3, ds3

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
    ln3.nds[l].cpt.to_csv(lfn, index=False)
plot_net(tn3, filename='results/IcyRoads.png')

