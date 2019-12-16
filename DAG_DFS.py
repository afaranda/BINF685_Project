#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 21:18:09 2019

@author: adam

Adapted from example video:
    https://www.youtube.com/watch?v=rKQaZuoUR4M
    Published by Tushar Roy - Coding Made Simple 
    on Jan 15, 2016
"""
import random as rn
class node:
    def __init__(self, idx):
        self.par=[]
        self.idx=idx
    
    def add_par(self, p):
        if not p in self.par:
            self.par.append(p)
    
    def del_par(self, p):
        if p in self.par:
            for i in range(0,len(self.par)):
                if p == self.par[i]:
                    self.par.pop(i)
                    break
        else:
            for i in range(0,len(self.par)):
                if p == self.par[i].idx:
                    self.par.pop(i)
                    break

############### White Grey Black DFS Cycle Detector

## Validate on example network
net={1+i:node(1+i) for i in range(0,6)}
net[1].add_par(net[2])
net[1].add_par(net[3])
net[2].add_par(net[3])
net[4].add_par(net[1])
net[4].add_par(net[5])
net[5].add_par(net[6])
net[6].add_par(net[4])

vis = {i:'W' for i in net.values()}


def dfs(node, vis):
    
    vis[node] = "G"
    
    for p in node.par:
        
        if vis[p] == "B":
            continue
        
        if vis[p] == "G":
            #print("Cycle Detected", p.idx)
            return True
        
        if wgbdfs(p, vis):
            #print("From", node.idx, "Cycle Detected Recursively", p.idx)
            return(True)
  
    vis[node]="B"
    return(False)

dfs(net[1], vis)
dfs(net[4], vis)

def val(net):
    vis = {i:'W' for i in net.values()}
    
    for node in vis.keys():
        if vis[node] == "W":
            if wgbdfs(node, vis):
                return True
        else:
            continue
    return False

print("Cycles Exist", val(net))




# Validate Acyclic Network
net={1+i:node(1+i) for i in range(0,6)}
net[1].add_par(net[2])
net[1].add_par(net[3])
net[2].add_par(net[3])
net[4].add_par(net[1])
net[4].add_par(net[5])
net[5].add_par(net[6])
net[4].add_par(net[6])
print("Cycles Exist", val(net))


# Validate Self Cyclic Node
net={1+i:node(1+i) for i in range(0,6)}
net[1].add_par(net[2])
net[1].add_par(net[3])
net[2].add_par(net[3])
net[4].add_par(net[1])
net[4].add_par(net[5])
net[5].add_par(net[6])
net[4].add_par(net[4])
print("Cycles Exist", val(net))

# Validate Randomly Generated Acyclic Networks
for i in range(0,100):
    net = [node(1+i) for i in range(0,1000)]
    used =[]
    for i in net:
        used.append(i)
        for j in set(net).difference(used):
            if rn.choices([True, False])[0]:
                i.add_par(j)
    net={i+1:net[i] for i in range(0, len(net))}        
    
    
       #print(net[i].idx, [ j.idx for j in net[i].par])
    if val(net):
        for i in net.keys():
            print(net[i].idx, [ j.idx for j in net[i].par])     
    print(val(net))

# Validate Randomly Generated Cyclic Networks
for i in range(0, 10):
    net = [node(1+i) for i in range(0,10)]
    used =[]
    for k in net:
        used.append(k)
        for j in used:
            if rn.choices([True, False])[0]:
                k.add_par(j)
    net={i+1:net[i] for i in range(0, len(net))}
    if not val(net):
        for l in net.keys():
           print(net[l].idx, [ j.idx for j in net[l].par])
          
    print(val(net))
           

# Validate Randomly Generated Sparse, Non-Self Cyclic Networks
# Note that this code may not allways generate a non-cyclic network
for i in range(0, 10):
    net = [node(1+i) for i in range(0,1000)]
    used =[]
    for k in net:
        used.append(k)
        for j in set(used).difference([k]):
            if rn.choices([True, False], [0.5,0.5])[0]:
                k.add_par(j)
        
    for m in net:
        if len(m.par) > 2:
            for p in m.par:
                if rn.choices([True, False], [0.5,0.5])[0]:
                    m.del_par(p)
                    p.add_par(m)
            
                    
                
    net={i+1:net[i] for i in range(0, len(net))}
    if not val(net):
        for l in net.keys():
           print(net[l].idx, [ j.idx for j in net[l].par])
          
    print(val(net))

# Examine the last network -- verify Cycle (Don't bother for nets > 10 nodes)
for l in net.keys():
    print(net[l].idx, [ j.idx for j in net[l].par])


















            
            
            
            
