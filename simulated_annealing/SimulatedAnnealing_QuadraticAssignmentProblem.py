#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 10:31:57 2018

@author: matthewbuckley
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def obj_func_QAP(distance_arr, flow_df):
    return pd.DataFrame(distance_arr * flow_df)


Dist = pd.DataFrame([[0,1,2,3,1,2,3,4],[1,0,1,2,2,1,2,3],[2,1,0,1,3,2,1,2],
                      [3,2,1,0,4,3,2,1],[1,2,3,4,0,1,2,3],[2,1,2,3,1,0,1,2],
                      [3,2,1,2,2,1,0,1],[4,3,2,1,3,2,1,0]],
                    columns=["A","B","C","D","E","F","G","H"],
                    index=["A","B","C","D","E","F","G","H"])

Flow = pd.DataFrame([[0,5,2,4,1,0,0,6],[5,0,3,0,2,2,2,0],[2,3,0,0,0,0,0,5],
                      [4,0,0,0,5,2,2,10],[1,2,0,5,0,10,0,0],[0,2,0,2,10,0,5,1],
                      [0,2,0,2,0,5,0,10],[6,0,5,10,0,1,10,0]],
                    columns=["A","B","C","D","E","F","G","H"],
                    index=["A","B","C","D","E","F","G","H"])

T0 = 1500
M = 250
N = 20
alpha = 0.9

x0 = ["B", "D", "A", "E", "C", "F", "G", "H"]

Dists_DF_0 = Dist.reindex(columns=x0, index=x0)
Dists_0 = np.array(Dists_DF_0)

objective_values_arr = np.array(obj_func_QAP(Dists_0, Flow))

starting_obj_value = sum(sum(objective_values_arr))

print(starting_obj_value)

temperatures = []
min_costs = []

for i in range(M):
    for j in range(N):
        ran_1 = np.random.randint(0, len(x0))
        ran_2 = np.random.randint(0, len(x0))
        
        while ran_1 == ran_2:
            ran_2 = np.random.randint(0, len(x0))
            
        xt = [department for department in x0]
        xf = []
            
        xt[ran_2] = x0[ran_1]
        xt[ran_1] = x0[ran_2]
            
        dist_df_0 = Dist.reindex(columns=x0, index=x0)
        dist_arr_0 = np.array(dist_df_0)
            
        dist_df_t = Dist.reindex(columns=xt, index=xt)
        dist_arr_t = np.array(dist_df_t)
            
        obj_vals_0 = np.array(obj_func_QAP(dist_arr_0, Flow))
            
        obj_vals_t = np.array(obj_func_QAP(dist_arr_t, Flow))
            
        obj_value_0 = sum(sum(obj_vals_0))
        obj_value_t = sum(sum(obj_vals_t))
            
        rand1 = np.random.rand()
        formula = 1 / (np.exp(obj_value_t - obj_value_0)/ T0)
            
        if (obj_value_t < obj_value_0) or (rand1 <= formula):
            x0 = xt
            obj_value_0 = obj_value_t
            
    temperatures.append(T0)
    min_costs.append(obj_value_0)
    
    T0 = alpha * T0
    
print()
print("Final Solution: {}".format(x0))
print("Minimized Cost: {}".format(obj_value_0))

plt.plot(temperatures, min_costs)
plt.plot("Cost vs. Temperature", fontsize=20, fontweight="bold")
plt.xlabel("Temperature", fontsize=18, fontweight="bold")
plt.ylabel("Cost", fontsize=18, fontweight="bold")
plt.xlim(1500, 0)

plt.xticks(np.arange(min(temperatures), max(temperatures), 100), fontweight="bold")
plt.yticks(fontweight="bold")
plt.show()
            
            