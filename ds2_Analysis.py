import pandas as pd
from Learners import greedy, CASGMM, CASMOD, CASJNB
from Metrics import kldv, accuracy, edge_hits
from network import net, export_pom, plot_net
from generate_networks import tn2, ds2


# Define the number of iterations learn on, and maximum number of sequential
# misses
niter = 50
maxm = 15
resdir ='results1'

### Run Learners on Network #2: 5 independent groups -- expect clusters
print("Net: ds2")
print("Iterations:",niter)
print("Max Miss:",maxm)

t_res=pd.DataFrame(
    columns=['Iteration','Score','Trial','Learner','Net']
)

g_res=pd.DataFrame(
    columns=['Trial', 'Learner', 'Net', 'TP', 'FP', 'TN', 'FN']
)    

ds2_acc=pd.DataFrame(
    columns=['Trial', 'Learner', 'Net'] + list(ds2.columns)   
)

for i in range(1, 6):
    print('Trial:', i)
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
    
plot_net(tn2, filename=resdir +'/TRUTH_ds2.png')
plot_net(grd.net, filename= resdir +"/GREEDY_ds2.png")
plot_net(cgm.net, filename= resdir + "/CASGMM_ds2.png")
plot_net(cmd.net, filename= resdir + "/CASMOD_ds2.png")
plot_net(cjn.net, filename= resdir + "/CASJNK_ds2.png")

t_res.to_csv(resdir + "/ds2_Training_Results.csv", index=False)
g_res.to_csv(resdir + "/ds2_Graph_Results.csv", index=False)
ds2_acc.to_csv(resdir + "/ds2_accuracy.csv", index=False)



  
