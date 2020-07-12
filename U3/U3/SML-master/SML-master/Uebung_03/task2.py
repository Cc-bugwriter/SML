# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    data = np.loadtxt(f"D:\\Test\\dataSets\\{filename}.txt")
    C_1 = np.mat(data[:50, :]).T
    C_2 = np.mat(data[50:93,:]).T
    C_3 = np.mat(data[93:, :]).T
    return C_1, C_2 ,C_3, data

def normal_vec_w (c_1,c_2):
    mean_c_1 = np.mean(c_1,1)
    mean_c_2 = np.mean(c_2,1)
    diff_w = np.dot ((c_1 - mean_c_1), ((c_1 - mean_c_1).T)) + np.dot ((c_2 - mean_c_2), ((c_2 - mean_c_2).T))
    mat_w = np.dot(np.linalg.inv(diff_w), (mean_c_2 - mean_c_1)) 
    normal_w = mat_w[0] / mat_w[1]
    return mat_w, normal_w

def offset(w, c_1, c_2):
    c1_re = np.dot(w.T , c_1)
    c2_re = np.dot(w.T , c_2)
    mean_c1_re = np.mean(c1_re)
    mean_c2_re = np.mean(c2_re)
    var_c1_re = np.var(c1_re)
    var_c2_re = np.var(c2_re)
    r1 = 1 / (2 * var_c1_re) - 1 / (2 * var_c2_re)
    r2 = (mean_c2_re / var_c2_re) - (mean_c1_re / var_c1_re)
    r3 = mean_c1_re ** 2 / (2 * var_c1_re) - mean_c2_re ** 2 / (2 * var_c2_re) - np.log(np.sqrt(var_c2_re/var_c1_re))
    
    root_r = np.roots([r1, r2, r3])
    for i in range (len(root_r)):
        if root_r[i] < max(mean_c1_re, mean_c2_re) and root_r[i] > min(mean_c1_re , mean_c2_re):
            w0 = root_r[i]
            break 
    return w0

if __name__ == "__main__":
    C_1, C_2, C_3, data = load_data('ldaData')
    a = C_1.T
    b = C_2.T
    c = C_3.T
    
    mat_w12, normal_w12 = normal_vec_w(C_1,C_2)
    mat_w13, normal_w13 = normal_vec_w(C_1,C_3)
    mat_w23, normal_w23 = normal_vec_w(C_2,C_3)
    
    w0_12 = offset(mat_w12,C_1, C_2)
    w0_13 = offset(mat_w13,C_1, C_3)
    w0_23 = offset(mat_w23,C_2, C_3)
    
    #without LDF
    for i in range(len(a)):
        plt.scatter(a[i, 0], a[i, 1] , marker = 'v' , color = 'red')
        i = i + 1
        
    for i in range(len(b)):
        plt.scatter(b[i, 0], b[i, 1] , marker = 'x' , color = 'yellow')
        i = i + 1
        
    for i in range(len(c)):
        plt.scatter(c[i, 0], c[i, 1] , marker = 'o' , color = 'blue')
        i = i + 1
        
    plt.show()

    # with LDF
    for i in range(len( data )):
        if np.dot(mat_w12.T, np.mat(data[i, :]).T) - w0_12 < 0:
            plt.scatter(data[i, 0], data[i, 1] , marker = 'v' , color = 'red')
        elif np.dot(mat_w23.T, np.mat(data[i, :]).T) - w0_23 < 0:
            plt.scatter(data[i, 0], data[i, 1], marker = 'x', color = 'yellow')
        else:
            plt.scatter(data[i, 0], data[i, 1], marker = 'o', color = 'blue')
        i = i + 1
    plt.show()