# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    data = np.loadtxt(f"D:\\Test\\dataSets\\{filename}.txt" , delimiter = ',')
    return data

def normalization(dataset):
    x , y = np.shape(dataset)
    for i in range( y - 1):
        dataset[: , i] = dataset[: , i] - np.mean(dataset[: , i])
        dataset[: , i] = dataset[: ,i] * np.sqrt(x / float(np.dot(np.mat(dataset[: , i]), np.mat(dataset[: , i]).T)))
        return x, y, dataset
    
if __name__ == "__main__":
    data_raw = load_data('iris')
    N, M, data_norm = normalization(data_raw)
    
    data_cal = np.mat(data_norm[: , 0 : 4]).T
    cov_data = np.cov(data_cal)
    lambda_data , W = np.linalg.eig(cov_data)
    
    #3b
    lambda_norm = lambda_data / np.sum(lambda_data)
    a = np.linspace(1, M-1, M-1)
    b = np.zeros(M-1)
    for i in range(M-1):
        b[i] = np.sum(lambda_norm[: i+1])
    plt.plot(a, b)
    plt.show()
    
    #3c
    B = np.mat(W[: , 0: 2]) 
    class_2d = np.dot(B.T, (data_cal - np.mean(data_cal, 1)))
    for i in range(N):
        if data_norm[i, 4] == 0:
            plt.scatter (class_2d[0 , i], class_2d[1 , i], marker = 'v', color = 'red')
        elif data_norm[i, 4] == 1:
            plt.scatter (class_2d[0 , i], class_2d[1 , i], marker = 'x', color = 'yellow')
        else:
            plt.scatter (class_2d[0 , i], class_2d[1 , i], marker = 'o', color = 'blue')
    plt.show()
    
    #3d
    data_cal_raw = np.mat(data_raw[: , 0 : 4]).T
    error = np.mat(np.zeros((4,4)))
    cov_data_raw = np.cov(data_cal_raw)
    lambda_data_raw , W_raw = np.linalg.eig(cov_data_raw)
    
    x_rot = np.dot(W_raw.T , data_cal_raw)
    x_pca_white = x_rot / (cov_data_raw + np.sqrt(1/np.power(10,5)))
    x_zca = np.dot(W_raw, x_pca_white)

    
    for i in range(M - 1):
        B_raw = np.mat(W_raw[:, 0: i+1])
        class_2d_raw = np.dot(B_raw.T, (data_cal_raw - np.mean(data_cal_raw, 1)))
        datau_cal = np.mean(data_cal_raw, 1) + np.dot(B_raw ,class_2d_raw)
        error[i, :] = np.sqrt(np.sum(np.power((data_cal_raw - datau_cal), 2), 1) / N).T
    print(error)