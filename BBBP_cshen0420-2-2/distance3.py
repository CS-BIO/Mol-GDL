# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 19:44:32 2021

@author: 申聪
"""
import numpy as np
import heapq

class mol_dis_sim(object):
    def Calculate_distance(Coor):
        Num_atoms = len(Coor)
    
        all_atoms = []
        Distance_matrix_real = np.zeros((Num_atoms,Num_atoms),dtype=float) 
        Distance_matrix_real2 = np.zeros((Num_atoms,Num_atoms),dtype=float) 
        Distance_matrix = np.zeros((Num_atoms,Num_atoms),dtype=float)
        for i in range(Num_atoms):
            all_atoms.append(Coor[i].split("\t")[0])
            for j in range(i+1,Num_atoms):
                # print("这是j的值：",j)
                # print("这是Coordinate_value[j]：",Coordinate_value[j])
                x_i = float(Coor[i].split("\t")[1])
                y_i = float(Coor[i].split("\t")[2])
                z_i = float(Coor[i].split("\t")[3])
                
                x_j = float(Coor[j].split("\t")[1])
                y_j = float(Coor[j].split("\t")[2])
                z_j = float(Coor[j].split("\t")[3])
    
                dis = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2 + (z_i - z_j) ** 2)
    
    #                eta = args.eta[a]
    #                exp_k = 4
    #                Similarity_matrix[i][j] = 1/(1+np.power(abs(dis)/eta,exp_k))
    #                Similarity_matrix[j][i] = 1/(1+np.power(abs(dis)/eta,exp_k))
                Distance_matrix_real[i][j] = dis
                Distance_matrix_real[j][i] = dis
                
#                if dis <= int(arr_cutoff[0]) or dis >= int(arr_cutoff[1]):
#                    Distance_matrix[i][j] = 0
#                    Distance_matrix[j][i] = 0
#                else:
#                    Distance_matrix[i][j] = 1
#                    Distance_matrix[j][i] = 1
            index_ = heapq.nlargest(5, range(len(Distance_matrix_real[i,:])), Distance_matrix_real[i,:].take)
#            random_value = 7
#            if len(Distance_matrix_real[i,:]) >= random_value:
#                index_ = np.random.choice(a=len(Distance_matrix_real[i,:]), size=random_value, replace=False, p=None)
#                while i in index_:                
#                    if len(Distance_matrix_real[i,:]) <= random_value:
#                        break
#                    index_ = np.random.choice(a=len(Distance_matrix_real[i,:]), size=random_value, replace=False, p=None)
#            else:
#                index_ = np.random.choice(a=len(Distance_matrix_real[i,:]), size=len(Distance_matrix_real[i,:]), replace=False, p=None)

            Distance_matrix[i,index_] = 1.0
            Distance_matrix_real2[i,index_] = Distance_matrix_real[i,index_]
            Distance_matrix_real2[i,i] = 0.0

        return Distance_matrix,Distance_matrix_real2,all_atoms


