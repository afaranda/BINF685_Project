import pandas as pd
from Learners import greedy, CASGMM, CASMOD, CASJNB
from Metrics import kldv, accuracy, edge_hits
from network import net, export_pom, plot_net
from generate_networks import tn1, ds1, tn2, ds2, tn3, ds3, tn4, ds4


# Define the number of iterations learn on, and maximum number of sequential
# misses
niter = 10
maxm = 5

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

### Run Learners on Network #1: 20 independent nodes -- expect no networks
t_res=pd.DataFrame(
    columns=['Iteration','Score','Trial','Learner','Net']
)

g_res=pd.DataFrame(
    columns=['Trial', 'Learner', 'Net', 'TP', 'FP', 'TN', 'FN']
)    

ds1_acc=pd.DataFrame(
    columns=['Trial', 'Learner', 'Net'] + list(ds1.columns)   
)

for i in range(1, 6):
    grd = greedy(ds1.loc[0:499])
    grd.train(iterations=niter, maxmiss=maxm)
    
    cgm = CASGMM(ds1.loc[0:499])
    cgm.train(iterations=niter, maxmiss=maxm)
    
    cmd = CASMOD(ds1.loc[0:499])
    cmd.train(iterations=niter, maxmiss=maxm)
    
    cjn = CASJNB(ds1.loc[0:499])
    cjn.train(iterations=niter, maxmiss=maxm)
    
    t_res = t_res.append(
        pd.DataFrame(
            grd.scores).assign(Trial = i, Learner = 'GREEDY', Net="ds1")
    )
    
    t_res=t_res.append(
        pd.DataFrame(
            cgm.scores).assign(Trial = i, Learner = 'CASGMM', Net="ds1")
    )
    
    t_res=t_res.append(
        pd.DataFrame(
            cmd.scores).assign(Trial = i, Learner = 'CASMOD', Net="ds1")
    )    
    
    t_res=t_res.append(
        pd.DataFrame(
            cjn.scores).assign(Trial = i, Learner = 'CASJNK', Net="ds1")
    )    
    
    g_res = g_res.append(
        pd.DataFrame(
            {
            i[0]:[i[1]]  
            for i in edge_hits(
                export_pom(grd.net, by='label'),
                export_pom(tn1, by='label')
                ).items()
            }
            ).assign(Trial = i, Learner = 'GREEDY', Net="ds1"),
        sort=False
                
    )
    
    g_res = g_res.append(
        pd.DataFrame(
            {
            i[0]:[i[1]]  
            for i in edge_hits(
                export_pom(cgm.net, by='label'),
                export_pom(tn1, by='label')
                ).items()
            }
            ).assign(Trial = i, Learner = 'CASGMM', Net="ds1"),
        sort=False
                
    )
    
    g_res = g_res.append(
        pd.DataFrame(
            {
            i[0]:[i[1]]  
            for i in edge_hits(
                export_pom(cmd.net, by='label'),
                export_pom(tn1, by='label')
                ).items()
            }
            ).assign(Trial = i, Learner = 'CASMOD', Net="ds1"),
        sort=False
                
    )
    
    g_res = g_res.append(
        pd.DataFrame(
            {
            i[0]:[i[1]]  
            for i in edge_hits(
                export_pom(cjn.net, by='label'),
                export_pom(tn1, by='label')
                ).items()
            }
            ).assign(Trial = i, Learner = 'CASJNK', Net="ds1"),
        sort=False
                
    )
    
    ds1_acc = ds1_acc.append(
        pd.DataFrame(
            {
            i[0]:[i[1]]  
            for i in accuracy(
                cjn.net,
                ds1.loc[500:599]
                ).items()
            }
            ).assign(Trial = i, Learner = 'GREEDY', Net="ds1"),
        sort=False
                
    )
    
    ds1_acc = ds1_acc.append(
        pd.DataFrame(
            {
            i[0]:[i[1]]  
            for i in accuracy(
                cgm.net,
                ds1.loc[500:599]
                ).items()
            }
            ).assign(Trial = i, Learner = 'CASGMM', Net="ds1"),
        sort=False
                
    )
    
    ds1_acc = ds1_acc.append(
        pd.DataFrame(
            {
            i[0]:[i[1]]  
            for i in accuracy(
                cmd.net,
                ds1.loc[500:599]
                ).items()
            }
            ).assign(Trial = i, Learner = 'CASMOD', Net="ds1"),
        sort=False
                
    )
    
    ds1_acc = ds1_acc.append(
        pd.DataFrame(
            {
            i[0]:[i[1]]  
            for i in accuracy(
                cjn.net,
                ds1.loc[500:599]
                ).items()
            }
            ).assign(Trial = i, Learner = 'CASJNK', Net="ds1"),
        sort=False
                
    )

plot_net(tn1, filename='results/TRUTH_ds1.png')
plot_net(grd.net, filename="results/GREEDY_ds1.png")
plot_net(cgm.net, filename="results/CASGMM_ds1.png")
plot_net(cmd.net, filename="results/CASMOD_ds1.png")
plot_net(cjn.net, filename="results/CASJNK_ds1.png")


### Run Learners on Network #2: 5 independent groups -- expect clusters
ds2_acc=pd.DataFrame(
    columns=['Trial', 'Learner', 'Net'] + list(ds2.columns)   
)

for i in range(1, 6):
    grd = greedy(ds2.loc[0:499])
    grd.train(iterations=niter, maxmiss=maxm)
    
    cgm = CASGMM(ds2.loc[0:499])
    cgm.train(iterations=niter, maxmiss=maxm)
    
    cmd = CASMOD(ds2.loc[0:499])
    cmd.train(iterations=niter, maxmiss=maxm)
    
    cjn = CASJNB(ds2.loc[0:499])
    cjn.train(iterations=niter, maxmiss=maxm)
    
    t_res = t_res.append(
        pd.DataFrame(
            grd.scores).assign(Trial = i, Learner = 'GREEDY', Net="ds2")
    )
    
    t_res=t_res.append(
        pd.DataFrame(
            cgm.scores).assign(Trial = i, Learner = 'CASGMM', Net="ds2")
    )
    
    t_res=t_res.append(
        pd.DataFrame(
            cmd.scores).assign(Trial = i, Learner = 'CASMOD', Net="ds2")
    )    
    
    t_res=t_res.append(
        pd.DataFrame(
            cjn.scores).assign(Trial = i, Learner = 'CASJNK', Net="ds2")
    )    
    
    g_res = g_res.append(
        pd.DataFrame(
            {
            i[0]:[i[1]]  
            for i in edge_hits(
                export_pom(grd.net, by='label'),
                export_pom(tn1, by='label')
                ).items()
            }
            ).assign(Trial = i, Learner = 'GREEDY', Net="ds2"),
        sort=False
                
    )
    
    g_res = g_res.append(
        pd.DataFrame(
            {
            i[0]:[i[1]]  
            for i in edge_hits(
                export_pom(cgm.net, by='label'),
                export_pom(tn2, by='label')
                ).items()
            }
            ).assign(Trial = i, Learner = 'CASGMM', Net="ds2"),
        sort=False
                
    )
    
    g_res = g_res.append(
        pd.DataFrame(
            {
            i[0]:[i[1]]  
            for i in edge_hits(
                export_pom(cmd.net, by='label'),
                export_pom(tn2, by='label')
                ).items()
            }
            ).assign(Trial = i, Learner = 'CASMOD', Net="ds2"),
        sort=False
                
    )
    
    g_res = g_res.append(
        pd.DataFrame(
            {
            i[0]:[i[1]]  
            for i in edge_hits(
                export_pom(cjn.net, by='label'),
                export_pom(tn2, by='label')
                ).items()
            }
            ).assign(Trial = i, Learner = 'CASJNK', Net="ds2"),
        sort=False
                
    )
    
    ds2_acc = ds2_acc.append(
        pd.DataFrame(
            {
            i[0]:[i[1]]  
            for i in accuracy(
                cjn.net,
                ds2.loc[500:599]
                ).items()
            }
            ).assign(Trial = i, Learner = 'GREEDY', Net="ds2"),
        sort=False
                
    )
    
    ds2_acc = ds2_acc.append(
        pd.DataFrame(
            {
            i[0]:[i[1]]  
            for i in accuracy(
                cgm.net,
                ds2.loc[500:599]
                ).items()
            }
            ).assign(Trial = i, Learner = 'CASGMM', Net="ds2"),
        sort=False
                
    )
    
    ds2_acc = ds2_acc.append(
        pd.DataFrame(
            {
            i[0]:[i[1]]  
            for i in accuracy(
                cmd.net,
                ds2.loc[500:599]
                ).items()
            }
            ).assign(Trial = i, Learner = 'CASMOD', Net="ds2"),
        sort=False
                
    )
    
    ds2_acc = ds2_acc.append(
        pd.DataFrame(
            {
            i[0]:[i[1]]  
            for i in accuracy(
                cjn.net,
                ds2.loc[500:599]
                ).items()
            }
            ).assign(Trial = i, Learner = 'CASJNK', Net="ds2"),
        sort=False
                
    )
    
plot_net(tn2, filename='results/TRUTH_ds2.png')
plot_net(grd.net, filename="results/GREEDY_ds2.png")
plot_net(cgm.net, filename="results/CASGMM_ds2.png")
plot_net(cmd.net, filename="results/CASMOD_ds2.png")
plot_net(cjn.net, filename="results/CASJNK_ds2.png")


### Run Learners on Network #4: The 'Alarm' network.  
ds4_acc=pd.DataFrame(
    columns=['Trial', 'Learner', 'Net'] + list(ds4.columns)   
)

for i in range(1, 6):
    grd = greedy(ds4.loc[0:499])
    grd.train(iterations=niter, maxmiss=maxm)
    
    cgm = CASGMM(ds4.loc[0:499])
    cgm.train(iterations=niter, maxmiss=maxm)
    
    cmd = CASMOD(ds4.loc[0:499])
    cmd.train(iterations=niter, maxmiss=maxm)
    
    cjn = CASJNB(ds4.loc[0:499])
    cjn.train(iterations=niter, maxmiss=maxm)
    
    t_res = t_res.append(
        pd.DataFrame(
            grd.scores).assign(Trial = i, Learner = 'GREEDY', Net="ds4")
    )
    
    t_res=t_res.append(
        pd.DataFrame(
            cgm.scores).assign(Trial = i, Learner = 'CASGMM', Net="ds4")
    )
    
    t_res=t_res.append(
        pd.DataFrame(
            cmd.scores).assign(Trial = i, Learner = 'CASMOD', Net="ds4")
    )    
    
    t_res=t_res.append(
        pd.DataFrame(
            cjn.scores).assign(Trial = i, Learner = 'CASJNK', Net="ds4")
    )    
    
    g_res = g_res.append(
        pd.DataFrame(
            {
            i[0]:[i[1]]  
            for i in edge_hits(
                export_pom(grd.net, by='label'),
                export_pom(tn4, by='label')
                ).items()
            }
            ).assign(Trial = i, Learner = 'GREEDY', Net="ds4"),
        sort=False
                
    )
    
    g_res = g_res.append(
        pd.DataFrame(
            {
            i[0]:[i[1]]  
            for i in edge_hits(
                export_pom(cgm.net, by='label'),
                export_pom(tn4, by='label')
                ).items()
            }
            ).assign(Trial = i, Learner = 'CASGMM', Net="ds4"),
        sort=False
                
    )
    
    g_res = g_res.append(
        pd.DataFrame(
            {
            i[0]:[i[1]]  
            for i in edge_hits(
                export_pom(cmd.net, by='label'),
                export_pom(tn4, by='label')
                ).items()
            }
            ).assign(Trial = i, Learner = 'CASMOD', Net="ds4"),
        sort=False
                
    )
    
    g_res = g_res.append(
        pd.DataFrame(
            {
            i[0]:[i[1]]  
            for i in edge_hits(
                export_pom(cjn.net, by='label'),
                export_pom(tn4, by='label')
                ).items()
            }
            ).assign(Trial = i, Learner = 'CASJNK', Net="ds4"),
        sort=False
                
    )
    
    ds4_acc = ds4_acc.append(
        pd.DataFrame(
            {
            i[0]:[i[1]]  
            for i in accuracy(
                cjn.net,
                ds4.loc[500:599]
                ).items()
            }
            ).assign(Trial = i, Learner = 'GREEDY', Net="ds4"),
        sort=False
                
    )
    
    ds4_acc = ds4_acc.append(
        pd.DataFrame(
            {
            i[0]:[i[1]]  
            for i in accuracy(
                cgm.net,
                ds4.loc[500:599]
                ).items()
            }
            ).assign(Trial = i, Learner = 'CASGMM', Net="ds4"),
        sort=False
                
    )
    
    ds4_acc = ds4_acc.append(
        pd.DataFrame(
            {
            i[0]:[i[1]]  
            for i in accuracy(
                cmd.net,
                ds4.loc[500:599]
                ).items()
            }
            ).assign(Trial = i, Learner = 'CASMOD', Net="ds4"),
        sort=False
                
    )
    
    ds4_acc = ds4_acc.append(
        pd.DataFrame(
            {
            i[0]:[i[1]]  
            for i in accuracy(
                cjn.net,
                ds4.loc[500:599]
                ).items()
            }
            ).assign(Trial = i, Learner = 'CASJNK', Net="ds4"),
        sort=False
                
    )
plot_net(tn4, filename='TRUTH_ds4.png')
plot_net(grd.net, filename="results/GREEDY_ds4.png")
plot_net(cgm.net, filename="results/CASGMM_ds4.png")
plot_net(cmd.net, filename="results/CASMOD_ds4.png")
plot_net(cjn.net, filename="results/CASJNK_ds4.png")

t_res.to_csv("results/Training_Results.csv", index=False)
g_res.to_csv("results/Graph_Results.csv", index=False)
ds1_acc.to_csv("results/ds1_accuracy.csv", index=False)
ds2_acc.to_csv("results/ds2_accuracy.csv", index=False)
ds4_acc.to_csv("results/ds4_accuracy.csv", index=False)


  