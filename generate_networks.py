#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 18:14:46 2020
This file contains network models for testing purposes.
Each network has a corresponding data set.  

ds1 is a set of 20 independent binary variables with the probability of a
'1' ranging from 0.958 to 0.042 

ds2 is a set of 20 binary variables with dependency groups of 5 variables.  
The data was generated such that the first variable in a group has a 50% 
probability of being 1, and each successive variable is equal to the preceding
variable with 95% probability. 

ds3 is an adaptation of the "Icy Roads" Example from:
Finn V. Jensen. An introduction to Bayesian Networks. 
UCL Press Ltd., London, 1996. ds3 has 4 variables in a diamond
shaped dependency structure: Either Watson or Holmes crashing their cars
are indepenent unless road conditions are given; an ambulance call is 
independent of road conditions if both drivers have crashed 
(or arrived safely).




@author: adam
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score as mi
from pomegranate import BayesianNetwork as bn
from network import net
from network import export_pom
from network import sample_net
from numpy.random import choice


# Dependency Function 1
def dpf1(x, p=1):
    if x == 1:
        return(choice([1,0], p=[p, 1-p]))
    else:
        return(choice([1,0], p=[1-p,p]))
 

# Dependency Function 2  
def dpf2(x, y, p=1, q=1):
    if x == 1 and y == 1:
        return(choice([1,0], p=[p, 1-p]))
    elif x != 1 and y == 1:
        return(choice([1,0], p=[q, 1-q]))
    elif x == 1 and y != 1:
        return(choice([1,0], p=[1-q, q]))
    else:
        return(choice([1,0], p=[1-p,p]))


# x=choice([0,1], size=100)
# y=choice([0,1], size=100)
# z=[dpf2(i[0],i[1], q=0.5) for i in zip(x,y)]
# df=pd.DataFrame({'G1':x, 'G2':y, 'G3':z})


# Data Set 1: 20 independent variables, 1000 samples with frequencies
# In descending order
nobs=5100
ds1 = pd.DataFrame(columns = ['G'+str(i) for i in range(0,20)])
for i in range(1,21):
    ds1['G'+str(i-1)]=choice([0,1],size=nobs, p=[i/21, 1-(i/21)])
tn1=net(data=ds1)
tn1.calc_cpt(ds1, alpha = 0.00001)

# Data Set 2: 20 variables with dependency groups of 5, where each variable
# in a group is dependent on the previous varible in increasing order
# eg. G1 depends on G0, G2 depends on G1 etc. . . Each group of five is
# independent from the other groups. 
    
ds2 = pd.DataFrame(columns = ['G'+str(i) for i in range(0,20)])
ds2['G0'] = choice([0,1],size=nobs, p=[0.5, 0.5])
for i in range(1,5):
    ds2['G'+str(i)]=[dpf1(i, 0.95) for i in ds2['G'+str(i-1)]]
    
ds2['G5'] = choice([0,1],size=nobs, p=[0.5, 0.5])
for i in range(6,10):
    ds2['G'+str(i)]=[dpf1(i, 0.95) for i in ds2['G'+str(i-1)]] 
    
ds2['G10'] = choice([0,1],size=nobs, p=[0.5, 0.5])
for i in range(11,15):
    ds2['G'+str(i)]=[dpf1(i, 0.95) for i in ds2['G'+str(i-1)]] 

ds2['G15'] = choice([0,1],size=nobs, p=[0.5, 0.5])
for i in range(16,20):
    ds2['G'+str(i)]=[dpf1(i, 0.95) for i in ds2['G'+str(i-1)]] 
tn2 = net(data=ds2)

for i in range(0,4):
    tn2.add_edge('G'+str(i), 'G'+str(i+1))
    
for i in range(5,9):
    tn2.add_edge('G'+str(i), 'G'+str(i+1))

for i in range(10,14):
    tn2.add_edge('G'+str(i), 'G'+str(i+1))

for i in range(15,19):
    tn2.add_edge('G'+str(i), 'G'+str(i+1))
tn2.calc_cpt(ds2, alpha=0.00001)

    
export_pom(tn1, by='label')

# Data Set 3: 4 Variables generated by sampling from a fixed set of conditional
# probability tables (Example from: Finsen Jenn)

data=pd.DataFrame({
    'Icy':[0,1],  
    'Holmes':[0,1],
    'Watson':[0,1],
    'Ambulance':[0,1]
    })

tn3=net(data=data)
tn3.add_edge('Icy', 'Watson')
tn3.add_edge('Icy','Holmes')
tn3.add_edge('Holmes', 'Ambulance')
tn3.add_edge('Watson', 'Ambulance')

tn3.nds['Icy'].cpt=tn3.nds['Icy'].empty_cpt(
    ).rename(
        {0:'Icy'}, axis=1
    ).assign(Prob=[0.7, 0.3])

tn3.nds['Holmes'].cpt=tn3.nds['Holmes'].empty_cpt(
    ).rename(
        {0:'Icy', 1:'Holmes'}, axis=1
    ).assign(Prob=[0.9,0.1, 0.05, 0.95])

tn3.nds['Watson'].cpt=tn3.nds['Watson'].empty_cpt(
    ).rename(
        {0:'Icy', 2:'Watson'}, axis=1
    ).assign(Prob=[0.9,0.1, 0.2, 0.8])

tn3.nds['Ambulance'].cpt=tn3.nds['Ambulance'].empty_cpt(
    ).rename(
        {1:'Holmes', 2:'Watson', 3:'Ambulance'}, axis=1
    ).assign(Prob=[0.9,0.1, 0.5, 0.5, 0.4, 0.6, 0.2, 0.8])

ds3=sample_net(tn3, nobs)
# Data Set 4:  10 Variables generated by sampling from a 
# fixed set of conditional probability tables, with a maximum of 3 parents
# per table

# Alarm Network Subset
alarm_net_variables={
    'HISTORY':[1,0,0,0],
    'CVP':[0,1,2,2],
    'PCWP':[0,1,2,2],
    'HYPOVOLEMIA':[1,0,0,0],
    'LVEDVOLUME':[0,1,2,2],
    'LVFAILURE':[1,0,0,0],
    'STROKEVOLUME':[0,1,2,2],
    'ERRLOWOUTPUT':[1,0,0,0],
    'HRBP':[0,1,2,2],
    'HREKG':[0,1,2,2],
    'ERRCAUTER':[1,0,0,0],
    'HRSAT':[0,1,2,2],
    'INSUFFANESTH':[1,0,0,0],
    'ANAPHYLAXIS':[1,0,0,0],
    'TPR':[0,1,2,2],
    'EXPCO2':[0,1,2,3],
    'KINKEDTUBE':[1,0,0,0],
    'MINVOL':[0,1,2,3],
    'FIO2':[0,1,1,1],
    'PVSAT':[0,1,2,2],
    'SAO2':[0,1,2,2],
    'PAP':[0,1,2,2],
    'PULMEMBOLUS':[1,0,0,0],
    'SHUNT':[0,1,1,1],
    'INTUBATION':[0,1,2,2],
    'PRESS':[0,1,2,3],
    'DISCONNECT':[1,0,0,0],
    'MINVOLSET':[0,1,2,2],
    'VENTMACH':[0,1,2,3],
    'VENTTUBE':[0,1,2,3],
    'VENTLUNG':[0,1,2,3],
    'VENTALV':[0,1,2,3],
    'ARTCO2':[0,1,2,2],
    'CATECHOL':[0,1,1,1],
    'HR':[0,1,2,2],
    'CO':[0,1,2,2],
    'BP':[0,1,2,2],
}


# alarm_net_variables={
#     'HISTORY':['TRUE', 'FALSE', 'FALSE', 'FALSE'],
#     'CVP':['LOW', 'NORMAL', 'HIGH', 'HIGH'],
#     'PCWP':['LOW', 'NORMAL', 'HIGH', 'HIGH'],
#     'HYPOVOLEMIA':['TRUE', 'FALSE', 'FALSE', 'FALSE'],
#     'LVEDVOLUME':['LOW', 'NORMAL', 'HIGH', 'HIGH'],
#     'LVFAILURE':['TRUE', 'FALSE', 'FALSE', 'FALSE'],
#     'STROKEVOLUME':['LOW', 'NORMAL', 'HIGH', 'HIGH'],
#     'ERRLOWOUTPUT':['TRUE', 'FALSE', 'FALSE', 'FALSE'],
#     'HRBP':['LOW', 'NORMAL', 'HIGH', 'HIGH'],
#     'HREKG':['LOW', 'NORMAL', 'HIGH', 'HIGH'],
#     'ERRCAUTER':['TRUE', 'FALSE', 'FALSE', 'FALSE'],
#     'HRSAT':['LOW', 'NORMAL', 'HIGH', 'HIGH'],
#     'INSUFFANESTH':['TRUE', 'FALSE', 'FALSE', 'FALSE'],
#     'ANAPHYLAXIS':['TRUE', 'FALSE', 'FALSE', 'FALSE'],
#     'TPR':['LOW', 'NORMAL', 'HIGH', 'HIGH'],
#     'EXPCO2':['ZERO','LOW', 'NORMAL', 'HIGH'],
#     'KINKEDTUBE':['TRUE', 'FALSE', 'FALSE', 'FALSE'],
#     'MINVOL':['ZERO','LOW', 'NORMAL', 'HIGH'],
#     'FIO2':['LOW', 'NORMAL', 'NORMAL', 'NORMAL'],
#     'PVSAT':['LOW', 'NORMAL', 'HIGH', 'HIGH'],
#     'SAO2':['LOW', 'NORMAL', 'HIGH', 'HIGH'],
#     'PAP':['LOW', 'NORMAL', 'HIGH', 'HIGH'],
#     'PULMEMBOLUS':['TRUE', 'FALSE', 'FALSE', 'FALSE'],
#     'SHUNT':['NORMAL', 'HIGH', 'HIGH', 'HIGH'],
#     'INTUBATION':['NORMAL', 'ESOPHAGEAL', 'ONESIDED', 'ONESIDED'],
#     'PRESS':['ZERO','LOW', 'NORMAL', 'HIGH'],
#     'DISCONNECT':['TRUE', 'FALSE', 'FALSE', 'FALSE'],
#     'MINVOLSET':['LOW', 'NORMAL', 'HIGH', 'HIGH'],
#     'VENTMACH':['ZERO','LOW', 'NORMAL', 'HIGH'],
#     'VENTTUBE':['ZERO','LOW', 'NORMAL', 'HIGH'],
#     'VENTLUNG':['ZERO','LOW', 'NORMAL', 'HIGH'],
#     'VENTALV':['ZERO','LOW', 'NORMAL', 'HIGH'],
#     'ARTCO2':['LOW', 'NORMAL', 'HIGH', 'HIGH'],
#     'CATECHOL':['NORMAL', 'HIGH', 'HIGH', 'HIGH'],
#     'HR':['LOW', 'NORMAL', 'HIGH', 'HIGH'],
#     'CO':['LOW', 'NORMAL', 'HIGH', 'HIGH'],
#     'BP':['LOW', 'NORMAL', 'HIGH', 'HIGH'],
# }


tn4=net(data=pd.DataFrame(alarm_net_variables))
tn4.nds['HYPOVOLEMIA'].cpt=tn4.nds['HYPOVOLEMIA'].empty_cpt(
    ).rename(
        {3:'HYPOVOLEMIA'}, axis=1
    ).assign(Prob=[0.2, 0.8])
print(tn4.nds['HYPOVOLEMIA'].cpt)

tn4.nds['LVFAILURE'].cpt=tn4.nds['LVFAILURE'].empty_cpt(
    ).rename(
        {5:'LVFAILURE'}, axis=1
    ).assign(Prob=[0.05, 0.95])
print(tn4.nds['LVFAILURE'].cpt)

tn4.nds['ERRLOWOUTPUT'].cpt=tn4.nds['ERRLOWOUTPUT'].empty_cpt(
    ).rename(
        {7:'ERRLOWOUTPUT'}, axis=1
    ).assign(Prob=[0.05, 0.95])
print(tn4.nds['ERRLOWOUTPUT'].cpt)

tn4.nds['ERRCAUTER'].cpt=tn4.nds['ERRCAUTER'].empty_cpt(
    ).rename(
        {10:'ERRCAUTER'}, axis=1
    ).assign(Prob=[0.1, 0.9])
print(tn4.nds['ERRCAUTER'].cpt)

tn4.nds['INSUFFANESTH'].cpt=tn4.nds['INSUFFANESTH'].empty_cpt(
    ).rename(
        {12:'INSUFFANESTH'}, axis=1
    ).assign(Prob=[0.1, 0.9])
print(tn4.nds['INSUFFANESTH'].cpt)

tn4.nds['ANAPHYLAXIS'].cpt=tn4.nds['ANAPHYLAXIS'].empty_cpt(
    ).rename(
        {13:'ANAPHYLAXIS'}, axis=1
    ).assign(Prob=[0.01, 0.99])
print(tn4.nds['ANAPHYLAXIS'].cpt)

tn4.nds['KINKEDTUBE'].cpt=tn4.nds['KINKEDTUBE'].empty_cpt(
    ).rename(
        {16:'KINKEDTUBE'}, axis=1
    ).assign(Prob=[0.04, 0.96])
print(tn4.nds['KINKEDTUBE'].cpt)

tn4.nds['FIO2'].cpt=tn4.nds['FIO2'].empty_cpt(
    ).rename(
        {18:'FIO2'}, axis=1
    ).assign(Prob=[0.05, 0.95])
print(tn4.nds['FIO2'].cpt)

tn4.nds['PULMEMBOLUS'].cpt=tn4.nds['PULMEMBOLUS'].empty_cpt(
    ).rename(
        {22:'PULMEMBOLUS'}, axis=1
    ).assign(Prob=[0.01, 0.99])
print(tn4.nds['PULMEMBOLUS'].cpt)

tn4.nds['INTUBATION'].cpt=tn4.nds['INTUBATION'].empty_cpt(
    ).rename(
        {24:'INTUBATION'}, axis=1
    ).assign(Prob=[0.92, 0.03, 0.05])
print(tn4.nds['INTUBATION'].cpt)

tn4.nds['DISCONNECT'].cpt=tn4.nds['DISCONNECT'].empty_cpt(
    ).rename(
        {26:'DISCONNECT'}, axis=1
    ).assign(Prob=[0.1, 0.9])
print(tn4.nds['DISCONNECT'].cpt)

tn4.nds['MINVOLSET'].cpt=tn4.nds['MINVOLSET'].empty_cpt(
    ).rename(
        {27:'MINVOLSET'}, axis=1
    ).assign(Prob=[0.05, 0.90, 0.05])
print(tn4.nds['MINVOLSET'].cpt)

tn4.add_edge('LVFAILURE', 'HISTORY')
tn4.nds['HISTORY'].cpt=tn4.nds['HISTORY'].empty_cpt(
    ).rename(
        {5:'LVFAILURE', 0:'HISTORY'}, axis=1
    ).assign(Prob=[0.9, 0.1, 0.01, 0.99])
print(tn4.nds['HISTORY'].cpt)

tn4.add_edge('LVEDVOLUME', 'CVP')
tn4.nds['CVP'].cpt=tn4.nds['CVP'].empty_cpt(
    ).rename(
        {4:'LVEDVOLUME', 1:'CVP'}, axis=1
    ).assign(Prob=[
        0.95, 0.04, 0.01,
        0.04, 0.95, 0.01,
        0.01, 0.29, 0.70
    ])
print(tn4.nds['CVP'].cpt)

tn4.add_edge('LVEDVOLUME', 'PCWP')
tn4.nds['PCWP'].cpt=tn4.nds['PCWP'].empty_cpt(
    ).rename(
        {4:'LVEDVOLUME', 2:'PCWP'}, axis=1
    ).assign(Prob=[
        0.95, 0.04, 0.01,
        0.04, 0.95, 0.01,
        0.01, 0.04, 0.95
    ])
print(tn4.nds['PCWP'].cpt)

tn4.add_edge('HYPOVOLEMIA', 'LVEDVOLUME')
tn4.add_edge('LVFAILURE', 'LVEDVOLUME')
tn4.nds['LVEDVOLUME'].cpt=tn4.nds['LVEDVOLUME'].empty_cpt(
    ).rename(
        {3:'HYPOVOLEMIA', 5:'LVFAILURE', 4:'LVEDVOLUME'}, axis=1
    ).assign(Prob=[
        0.95, 0.04, 0.01,
        0.01, 0.09, 0.90,
        0.98, 0.01, 0.01,
        0.05, 0.90, 0.05
    ])
print(tn4.nds['LVEDVOLUME'].cpt)

tn4.add_edge('HYPOVOLEMIA', 'STROKEVOLUME')
tn4.add_edge('LVFAILURE', 'STROKEVOLUME')
tn4.nds['STROKEVOLUME'].cpt=tn4.nds['STROKEVOLUME'].empty_cpt(
    ).rename(
        {3:'HYPOVOLEMIA', 5:'LVFAILURE', 6:'STROKEVOLUME'}, axis=1
    ).assign(Prob=[
        0.98, 0.01, 0.01,
        0.50, 0.49, 0.01,
        0.95, 0.04, 0.01,
        0.05, 0.90, 0.05
    ])
print(tn4.nds['STROKEVOLUME'].cpt)

tn4.add_edge('ERRLOWOUTPUT', 'HRBP')
tn4.add_edge('HR', 'HRBP')
tn4.nds['HRBP'].cpt=tn4.nds['HRBP'].empty_cpt(
    ).rename(
        {7:'ERRLOWOUTPUT', 34:'HR', 8:'HRBP'}, axis=1
    ).assign(Prob=[
        0.98, 0.01, 0.01,
        0.3, 0.4, 0.3,
        0.01, 0.98, 0.01,
        0.40, 0.59, 0.01,
        0.98, 0.01, 0.01,
        0.01, 0.01, 0.98
    ])
print(tn4.nds['HRBP'].cpt)

tn4.add_edge('ERRCAUTER', 'HREKG')
tn4.add_edge('HR', 'HREKG')
tn4.nds['HREKG'].cpt=tn4.nds['HREKG'].empty_cpt(
    ).rename(
        {10:'ERRCAUTER', 34:'HR', 9:'HREKG'}, axis=1
    ).assign(Prob=[
        1./3, 1./3, 1./3,
        1./3, 1./3, 1./3,
        0.01, 0.98, 0.01,
        1./3, 1./3, 1./3,
        0.98, 0.01, 0.01,
         0.01, 0.01, 0.98
        
    ])
print(tn4.nds['HREKG'].cpt)

tn4.add_edge('ERRCAUTER', 'HRSAT')
tn4.add_edge('HR', 'HRSAT')
tn4.nds['HRSAT'].cpt=tn4.nds['HRSAT'].empty_cpt(
    ).rename(
        {10:'ERRCAUTER', 34:'HR', 11:'HRSAT'}, axis=1
    ).assign(Prob=[
        1./3, 1./3, 1./3,
        1./3, 1./3, 1./3,
        0.01, 0.98, 0.01,
        1./3, 1./3, 1./3,
        0.98, 0.01, 0.01,
        0.01, 0.01, 0.98
        
    ])
print(tn4.nds['HRSAT'].cpt)

tn4.add_edge('ANAPHYLAXIS', 'TPR')
tn4.nds['TPR'].cpt=tn4.nds['TPR'].empty_cpt(
    ).rename(
        {13:'ANAPHYLAXIS', 14:'TPR'}, axis=1
    ).assign(Prob=[
        0.98, 0.01, 0.01,
        0.3, 0.4, 0.3
    ])
print(tn4.nds['TPR'].cpt)

tn4.add_edge('ARTCO2', 'EXPCO2')
tn4.add_edge('VENTLUNG', 'EXPCO2')
tn4.nds['EXPCO2'].cpt=tn4.nds['EXPCO2'].empty_cpt(
    ).rename(
        {32:'ARTCO2', 30:'VENTLUNG', 15:'EXPCO2'}, axis=1
    ).assign(Prob=[
        0.97, 0.01, 0.01, 0.01,
        0.01, 0.97, 0.01, 0.01,
        0.01, 0.01, 0.97, 0.01,
        0.01, 0.01, 0.01, 0.97,
        0.01, 0.97, 0.01, 0.01,
        0.97, 0.01, 0.01, 0.01,
        0.01, 0.01, 0.97, 0.01,
        0.01, 0.01, 0.01, 0.97,
        0.01, 0.97, 0.01, 0.01,
        0.01, 0.01, 0.97, 0.01,
        0.97, 0.01, 0.01, 0.01,
        0.01, 0.01, 0.01, 0.97
    ])
print(tn4.nds['EXPCO2'].cpt)


tn4.add_edge('VENTLUNG', 'MINVOL')
tn4.add_edge('INTUBATION', 'MINVOL')

tn4.nds['MINVOL'].cpt=tn4.nds['MINVOL'].empty_cpt(
    ).rename(
        {24:'INTUBATION', 30:'VENTLUNG', 17:'MINVOL'}, axis=1
    ).assign(Prob=[
        0.97, 0.01, 0.01, 0.01,
        0.01, 0.97, 0.01, 0.01,
        0.01, 0.01, 0.97, 0.01,
        0.01, 0.01, 0.01, 0.97,
        0.97, 0.01, 0.01, 0.01,
        0.60, 0.38, 0.01, 0.01,
        0.50, 0.48, 0.01, 0.01,
        0.50, 0.48, 0.01, 0.01,
        0.97, 0.01, 0.01, 0.01,
        0.01, 0.97, 0.01, 0.01,
        0.01, 0.01, 0.97, 0.01,
        0.01, 0.01, 0.01, 0.97,
       
    ])
print(tn4.nds['MINVOL'].cpt)

tn4.add_edge('VENTALV', 'PVSAT')
tn4.add_edge('FIO2', 'PVSAT')
tn4.nds['PVSAT'].cpt=tn4.nds['PVSAT'].empty_cpt(
    ).rename(
        {18:'FIO2', 31:'VENTALV', 19:'PVSAT'}, axis=1
    ).assign(Prob=[
        1.0, 0.0, 0.0,
        0.99, 0.01, 0.00,
        0.95, 0.04, 0.01,
        0.95, 0.04, 0.01,
        1.0, 0.0, 0.0,
        0.95, 0.04, 0.01,
        0.01, 0.95, 0.04,
        0.01, 0.01, 0.98
    ])
print(tn4.nds['PVSAT'].cpt)

tn4.add_edge('SHUNT', 'SAO2')
tn4.add_edge('PVSAT', 'SAO2')
tn4.nds['SAO2'].cpt=tn4.nds['SAO2'].empty_cpt(
    ).rename(
        {19:'PVSAT', 23:'SHUNT', 20:'SAO2'}, axis=1
    ).assign(Prob=[
        0.98, 0.01, 0.01,
        0.01, 0.98, 0.01,
        0.01, 0.01, 0.98,
        0.98, 0.01, 0.01,
        0.98, 0.01, 0.01,
        0.69, 0.30, 0.01
    ])
print(tn4.nds['SAO2'].cpt)

tn4.add_edge('PULMEMBOLUS', 'PAP')
tn4.nds['PAP'].cpt=tn4.nds['PAP'].empty_cpt(
    ).rename(
        {22:'PULMEMBOLUS', 21:'PAP'}, axis=1
    ).assign(Prob=[
        0.01, 0.19, 0.80,
        0.05, 0.90, 0.05
    ])
print(tn4.nds['PAP'].cpt)

tn4.add_edge('PULMEMBOLUS', 'SHUNT')
tn4.add_edge('INTUBATION', 'SHUNT')
tn4.nds['SHUNT'].cpt=tn4.nds['SHUNT'].empty_cpt(
    ).rename(
        {22:'PULMEMBOLUS', 24:'INTUBATION', 23:'SHUNT'}, axis=1
    ).assign(Prob=[
        0.1, 0.9,
        0.1, 0.9,
        0.01, 0.99,
        0.95, 0.05,
        0.95, 0.05,
        0.05, 0.95
    ])
print(tn4.nds['SHUNT'].cpt)


tn4.add_edge('VENTTUBE', 'PRESS')
tn4.add_edge('KINKEDTUBE', 'PRESS')
tn4.add_edge('INTUBATION', 'PRESS')
tn4.nds['PRESS'].cpt=tn4.nds['PRESS'].empty_cpt(
    ).rename(
        {29:'VENTTUBE', 16:'KINKEDTUBE', 24:'INTUBATION', 25:'PRESS'}, axis=1
    ).assign(Prob=[
        0.97, 0.01, 0.01, 0.01,
        0.01, 0.30, 0.49, 0.20,
        0.01, 0.01, 0.08, 0.90,
        0.01, 0.01, 0.01, 0.97,
        0.97, 0.01, 0.01, 0.01,
        0.10, 0.84, 0.05, 0.01,
        0.05, 0.25, 0.25, 0.45,
        0.01, 0.15, 0.25, 0.59,
        0.97, 0.01, 0.01, 0.01,
        0.01, 0.29, 0.30, 0.40,
        0.01, 0.01, 0.08, 0.90,
        0.01, 0.01, 0.01, 0.97,
        0.97, 0.01, 0.01, 0.01,
        0.01, 0.97, 0.01, 0.01,
        0.01, 0.01, 0.97, 0.01,
        0.01, 0.01, 0.01, 0.97,
        0.97, 0.01, 0.01, 0.01,
        0.40, 0.58, 0.01, 0.01,
        0.20, 0.75, 0.04, 0.01,
        0.20, 0.70, 0.09, 0.01,
        0.97, 0.01, 0.01, 0.01,
        0.01, 0.90, 0.08, 0.01,
        0.01, 0.01, 0.38, 0.60,
        0.01, 0.01, 0.01, 0.97
    ])
print(tn4.nds['PRESS'].cpt)

tn4.add_edge('MINVOLSET', 'VENTMACH')
tn4.nds['VENTMACH'].cpt=tn4.nds['VENTMACH'].empty_cpt(
    ).rename(
        {27:'MINVOLSET', 28:'VENTMACH'}, axis=1
    ).assign(Prob=[
        0.05, 0.93, 0.01, 0.01,
        0.05, 0.01, 0.93, 0.01,
        0.05, 0.01, 0.01, 0.93
    ])
print(tn4.nds['VENTMACH'].cpt)


tn4.add_edge('VENTMACH', 'VENTTUBE')
tn4.add_edge('DISCONNECT', 'VENTTUBE')
tn4.nds['VENTTUBE'].cpt=tn4.nds['VENTTUBE'].empty_cpt(
    ).rename(
        {28:'VENTMACH', 26:'DISCONNECT', 29:'VENTTUBE'}, axis=1
    ).assign(Prob=[
        0.97, 0.01, 0.01, 0.01,
        0.97, 0.01, 0.01, 0.01,
        0.97, 0.01, 0.01, 0.01,
        0.97, 0.01, 0.01, 0.01,
        0.97, 0.01, 0.01, 0.01,
        0.01, 0.97, 0.01, 0.01,
        0.01, 0.01, 0.97, 0.01,
        0.01, 0.01, 0.01, 0.97
    ])
print(tn4.nds['VENTTUBE'].cpt)

tn4.add_edge('VENTTUBE', 'VENTLUNG')
tn4.add_edge('KINKEDTUBE', 'VENTLUNG')
tn4.add_edge('INTUBATION', 'VENTLUNG')
tn4.nds['VENTLUNG'].cpt=tn4.nds['VENTLUNG'].empty_cpt(
    ).rename(
        {29:'VENTTUBE', 16:'KINKEDTUBE', 24:'INTUBATION', 30:'VENTLUNG'}, axis=1
    ).assign(Prob=[
        0.97, 0.01, 0.01, 0.01,
        0.95, 0.03, 0.01, 0.01,
        0.40, 0.58, 0.01, 0.01,
        0.30, 0.68, 0.01, 0.01,
        0.97, 0.01, 0.01, 0.01,
        0.97, 0.01, 0.01, 0.01,
        0.97, 0.01, 0.01, 0.01,
        0.97, 0.01, 0.01, 0.01,
        0.97, 0.01, 0.01, 0.01,
        0.95, 0.03, 0.01, 0.01,
        0.50, 0.48, 0.01, 0.01,
        0.30, 0.68, 0.01, 0.01,
        0.97, 0.01, 0.01, 0.01,
        0.01, 0.97, 0.01, 0.01,
        0.01, 0.01, 0.97, 0.01,
        0.01, 0.01, 0.01, 0.97,
        0.97, 0.01, 0.01, 0.01,
        0.97, 0.01, 0.01, 0.01,
        0.97, 0.01, 0.01, 0.01,
        0.97, 0.01, 0.01, 0.01,
        0.97, 0.01, 0.01, 0.01,
        0.01, 0.97, 0.01, 0.01,
        0.01, 0.01, 0.97, 0.01,
        0.01, 0.01, 0.01, 0.97
    ])
print(tn4.nds['VENTLUNG'].cpt)


tn4.add_edge('VENTLUNG', 'VENTALV')
tn4.add_edge('INTUBATION', 'VENTALV')
tn4.nds['VENTALV'].cpt=tn4.nds['VENTALV'].empty_cpt(
    ).rename(
        {30:'VENTLUNG', 24:'INTUBATION', 31:'VENTALV'}, axis=1
    ).assign(Prob=[
        0.97, 0.01, 0.01, 0.01,
        0.01, 0.97, 0.01, 0.01,
        0.01, 0.01, 0.97, 0.01,
        0.01, 0.01, 0.01, 0.97,
        0.97, 0.01, 0.01, 0.01,
        0.01, 0.97, 0.01, 0.01,
        0.01, 0.01, 0.97, 0.01,
        0.01, 0.01, 0.01, 0.97,
        0.97, 0.01, 0.01, 0.01,
        0.03, 0.95, 0.01, 0.01,
        0.01, 0.94, 0.04, 0.01,
        0.01, 0.88, 0.10, 0.01
    ])
print(tn4.nds['VENTALV'].cpt)


tn4.add_edge('VENTALV', 'ARTCO2')
tn4.nds['ARTCO2'].cpt=tn4.nds['ARTCO2'].empty_cpt(
    ).rename(
        {31:'VENTALV', 32:'ARTCO2'}, axis=1
    ).assign(Prob=[
        0.01, 0.01, 0.98,
        0.01, 0.01, 0.98,
        0.04, 0.92, 0.04,
        0.90, 0.09, 0.01
    ])
print(tn4.nds['ARTCO2'].cpt)


tn4.add_edge('TPR', 'CATECHOL')
tn4.add_edge('SAO2', 'CATECHOL')
tn4.add_edge('INSUFFANESTH', 'CATECHOL')
tn4.add_edge('ARTCO2', 'CATECHOL')
tn4.nds['CATECHOL'].cpt=tn4.nds['CATECHOL'].empty_cpt(
    ).rename(
        {14:'TPR', 20:'SAO2', 12:'INSUFFANESTH',32:'ARTCO2', 33:'CATECHOL'}, 
        axis=1
    ).assign(Prob=[
        0.01, 0.99,
        0.01, 0.99,
        0.01, 0.99,
        0.01, 0.99,
        0.01, 0.99,
        0.01, 0.99,
        0.01, 0.99,
        0.01, 0.99,
        0.01, 0.99,
        0.01, 0.99,
        0.01, 0.99,
        0.01, 0.99,
        0.01, 0.99,
        0.01, 0.99,
        0.01, 0.99,
        0.05, 0.95,
        0.05, 0.95,
        0.01, 0.99,
        0.01, 0.99,
        0.01, 0.99,
        0.01, 0.99,
        0.05, 0.95,
        0.05, 0.95,
        0.01, 0.99,
        0.05, 0.95,
        0.05, 0.95,
        0.01, 0.99,
        0.05, 0.95,
        0.05, 0.95,
        0.01, 0.99,
        0.05, 0.95,
        0.05, 0.95,
        0.01, 0.99,
        0.05, 0.95,
        0.05, 0.95,
        0.01, 0.99,
        0.7, 0.3,
        0.7, 0.3,
        0.1, 0.9,
        0.7, 0.3,
        0.7, 0.3,
        0.1, 0.9,
        0.7, 0.3,
        0.7, 0.3,
        0.1, 0.9,
        0.95, 0.05,
        0.99, 0.01,
        0.3, 0.7,
        0.95, 0.05,
        0.99, 0.01,
        0.3, 0.7,
        0.95, 0.05,
        0.99, 0.01,
        0.3, 0.7
    ])
print(tn4.nds['CATECHOL'].cpt)

# probability ( HR | CATECHOL ) {
#   (NORMAL) 0.05, 0.90, 0.05;
#   (HIGH) 0.01, 0.09, 0.90;
# }
tn4.add_edge('CATECHOL', 'HR')
tn4.nds['HR'].cpt=tn4.nds['HR'].empty_cpt(
    ).rename(
        {33:'CATECHOL', 34:'HR'}, axis=1
    ).assign(Prob=[
        0.05, 0.90, 0.05,
        0.01, 0.09, 0.90
    ])
print(tn4.nds['HR'].cpt)


tn4.add_edge('STROKEVOLUME', 'CO')
tn4.add_edge('HR', 'CO')
tn4.nds['CO'].cpt=tn4.nds['CO'].empty_cpt(
    ).rename(
        {6:'STROKEVOLUME', 34:'HR', 35:'CO'}, axis=1
    ).assign(Prob=[
        0.98, 0.01, 0.01,
        0.95, 0.04, 0.01,
        0.80, 0.19, 0.01,
        0.95, 0.04, 0.01,
        0.04, 0.95, 0.01,
        0.01, 0.04, 0.95,
        0.30, 0.69, 0.01,
        0.01, 0.30, 0.69,
        0.01, 0.01, 0.98
    ])
print(tn4.nds['CO'].cpt)

tn4.add_edge('TPR', 'BP')
tn4.add_edge('CO', 'BP')
tn4.nds['BP'].cpt=tn4.nds['BP'].empty_cpt(
    ).rename(
        {14:'TPR', 35:'CO', 36:'BP'}, axis=1
    ).assign(Prob=[
        0.98, 0.01, 0.01,
        0.95, 0.04, 0.01,
        0.80, 0.19, 0.01,
        0.95, 0.04, 0.01,
        0.04, 0.95, 0.01,
        0.01, 0.04, 0.95,
        0.30, 0.69, 0.01,
        0.01, 0.30, 0.69,
        0.01, 0.01, 0.98
    ])
print(tn4.nds['BP'].cpt)
ds4=sample_net(tn4, 1000)

for c in ds4.columns:
    ds4[c]=ds4[c].astype('category')














