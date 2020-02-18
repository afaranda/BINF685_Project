import pandas as pd
from Learners import greedy, CASGMM, CASMOD, CASJNB
from Metrics import accuracy, edge_hits
from network import net, export_pom, plot_net
from generate_networks import tn4, ds4


# Define the number of iterations learn on, and maximum number of sequential
# misses
niter=150
maxm=25
resdir='results3'


### Run Learners on Network #4: The 'Alarm' network.
print("Net: ds4")
print("Iterations:",niter)
print("Max Miss:",maxm)

t_res=pd.DataFrame(
    columns=['Iteration','Score','Trial','Learner','Net']
)

g_res=pd.DataFrame(
    columns=['Trial', 'Learner', 'Net', 'TP', 'FP', 'TN', 'FN']
)    

ds4_acc=pd.DataFrame(
    columns=['Trial', 'Learner', 'Net'] + list(ds4.columns)   
)

for i in range(1, 6):
    print('Trial:', i)
    print('greedy')
    grd = greedy(ds4.loc[0:499])
    grd.train(iterations=niter, maxmiss=maxm)
    
    print('CASGMM')
    cgm = CASGMM(ds4.loc[0:499])
    cgm.train(iterations=niter, maxmiss=maxm)
    
    print('CASMOD')
    cmd = CASMOD(ds4.loc[0:499])
    cmd.train(iterations=niter, maxmiss=maxm)
    
    print('CASJNB')
    cjn = CASJNB(ds4.loc[0:499])
    cjn.train(iterations=niter, maxmiss=maxm)
    
    print('Scoring')
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
                ds4.loc[500:519]
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
                ds4.loc[500:519]
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
                ds4.loc[500:519]
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
                ds4.loc[500:519]
                ).items()
            }
            ).assign(Trial = i, Learner = 'CASJNK', Net="ds4"),
        sort=False
                
    )
    
    
    
accuracy2( export_pom(cjn.net,by='label'),ds4.loc[500:599])

plot_net(tn4, filename= resdir + '/TRUTH_ds4.png')
plot_net(grd.net, filename= resdir + "/GREEDY_ds4.png")
plot_net(cgm.net, filename= resdir + "/CASGMM_ds4.png")
plot_net(cmd.net, filename= resdir + "/CASMOD_ds4.png")
plot_net(cjn.net, filename= resdir + "/CASJNK_ds4.png")

t_res.to_csv(resdir + "/ds4_Training_Results.csv", index=False)
g_res.to_csv(resdir + "/ds4_Graph_Results.csv", index=False)
ds4_acc.to_csv(resdir + "/ds4_accuracy.csv", index=False)
