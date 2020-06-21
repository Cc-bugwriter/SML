import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp


# task 3a
def plot_histogram(data_set, bin_size):
    # assign number of bar
    num = ((int(np.max(data_set)) + 1) - int(np.min(data_set))) // bin_size + 1

    # define bar list
    bin_list = np.linspace(int(np.min(data_set)), int(np.max(data_set)) + 1, int(num))

    # plot
    plt.figure()
    plt.hist(data_set, bins=bin_list)
    plt.title('bin_size ={size}'.format(size=bin_size))
    plt.ylabel('times')
    plt.xlabel('data')
    plt.grid()
    plt.show()


#  task 3b
def kernel_density_estimate(data_set, x_list, sigma):
    # define gaussian kernel function
    def gaussian_kernel(data_set, x, sigma):
        p_arr = np.exp(- np.linalg.norm(data_set.reshape(-1, 1) - x, axis=1) ** 2 / (2 * sigma ** 2))
        return p_arr

    # assign dimension
    try:
        d = data_set.shape[1]
    except IndexError:
        d = 1

    # assign number
    num = data_set.shape[0]

    # initial return argument
    kde = []

    for x in x_list:
        # compute local probability density
        p_x = np.sum(gaussian_kernel(data_set, x, sigma)) / (num * (np.sqrt(2*np.pi*sigma**2)) ** d)

        kde.append(p_x)

    return kde


def kde_log_likelihood(data_set, x_list, sigma):
    # assign number
    num = data_set.shape[0]

    # initial return argument
    likelihood_list = []

    for x in x_list:
        value = logsumexp(-np.linalg.norm(data_set.reshape(-1, 1) - x, axis=1) ** 2 /
                          (2*sigma**2) - np.log(np.sqrt(2*np.pi*sigma**2) * num))
        likelihood_list.append(value)

    return np.sum(likelihood_list)


# task 3c
def K_Nearest_Neighbors(data_set, x_list, k):
    # assign number
    num = data_set.shape[0]

    # initial return argument
    knn_list = []

    for x in x_list:
        u = np.linalg.norm(data_set.reshape(-1, 1) - x, axis=1)
        r = np.sort(u)[k-1]
        v = 2 * r
        p = k / (num * v)
        knn_list.append(p)

    return knn_list, np.sum(np.log(knn_list))


# task 3d
def Comparison_NPM(train_dataset, test_dataset, sigma_list, k_list):

    for sigma in sigma_list:
        print(f'log-likelihood of KDE by test data set (sigma = {sigma}): '
              f'{kde_log_likelihood(train_dataset, test_dataset, sigma)}')

    for k in k_list:
        _, likelihood = K_Nearest_Neighbors(train_dataset, test_dataset, k)
        print(f'log-likelihood of KNN by test data set(k={k}): {likelihood}')


if __name__ == '__main__':
    # load dataset
    train_dataset = np.loadtxt('./dataSets/nonParamTrain.txt')
    test_dataset = np.loadtxt('./dataSets/nonParamTest.txt')

    # 3a
    for bar_size in [0.02, 0.5, 2.]:
        plot_histogram(train_dataset, bar_size)

    # 3b
    # assign sigma list
    sigma_list = [0.03, 0.2, 0.8]

    # assign independent variable
    x_list = np.linspace(-4, 8, train_dataset.shape[0])

    # assign color space for plot
    col_space = ['r', 'g', 'b']

    plt.figure()
    for i, sigma in enumerate(sigma_list):
        kde = kernel_density_estimate(train_dataset, x_list, sigma)
        plt.plot(x_list, kde, color=col_space[i], label=f'sigma = {sigma}')
        print(f'log-likelihood(sigma = {sigma}): {kde_log_likelihood(train_dataset, train_dataset, sigma)}')
    plt.grid()
    plt.legend(loc='best')
    plt.title("Kernel Density Estimate (Gaussian) ", fontsize=15)
    plt.show()

    # 3c
    # assign k list
    k_list = [2, 8, 35]

    plt.figure()
    for i, k in enumerate(k_list):
        knn, likelihood = K_Nearest_Neighbors(train_dataset, x_list, k)
        plt.plot(x_list, knn, color=col_space[i], label=f'K = {k}')
        print(f'log-likelihood of KNN by training dataset(k={k}): {likelihood}')
    plt.grid()
    plt.legend(loc='best')
    plt.ylim(ymin=0, ymax=1)
    plt.title("KNN Density Estimation", fontsize=15)
    plt.show()

    # 3d
    Comparison_NPM(train_dataset, test_dataset, sigma_list, k_list)