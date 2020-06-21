import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp


# task 3a
def plot_histogram(data_set, bin_size):
    num = (np.max(data_set)-np.min(data_set)) // bin_size + 1
    bin_list = np.linspace(np.min(data_set), np.max(data_set), num)
    plt.hist(data_set, bins=bin_list)
    plt.title('bin_size ={size}'.format(size=bin_size))
    plt.ylabel('times')
    plt.xlabel('data')
    plt.grid()
    plt.show()


#  task 3b
def gaussian(x,sigma):
    gaus = np.exp(-x ** 2 / (2*sigma**2)) / (np.sqrt(2*np.pi*sigma**2))
    return gaus


def kernel_density_estimate(data, sigma):
    num = data.shape[0]
    x_list = np.linspace(-4, 8, num)
    kde = list()
    for x in x_list:
        kde.append(np.sum(gaussian(data-x, sigma)) / num)
    return x_list, kde


def kde_log_likelihood(data, X, sigma):
    num = data.shape[0]
    values_list = list()
    for x_n in X:
        value = logsumexp(-(data - x_n) ** 2 / (2*sigma**2) - np.log(np.sqrt(2*np.pi*sigma**2) * num))
        values_list.append(value)
    return np.sum(values_list)


# task 3c
def KNN(data, k):
    num = data.shape[0]
    knn_list = list()
    x_list = np.linspace(-4, 8, num)
    for x in x_list:
        u = np.abs(data - x)
        r = np.sort(u)[k-1]
        v = 2 * r
        p = k / (num * v)
        knn_list.append(p)
    return x_list, knn_list


def KNN_likelihood(data, X, k):
    num = data.shape[0]
    values_list = list()
    for x_n in X:
        u = np.abs(data - x_n)
        r = np.sort(u)[k-1]
        v = 2 * r
        p = k / (num * v)
        values_list.append(p)
    return np.sum(np.log(values_list))



# task 3d
def Comparison_NPM(train_dataset, test_dataset):
    print('log-likelihood of KDE by train dataset(sigma=0.03): {}'.format(
        kde_log_likelihood(train_dataset, train_dataset, 0.03)))
    print('log-likelihood of KDE by train dataset(sigma=0.2): {}'.format(
        kde_log_likelihood(train_dataset, train_dataset, 0.2)))
    print('log-likelihood of KDE by train dataset(sigma=0.8): {}'.format(
        kde_log_likelihood(train_dataset, train_dataset, 0.8)))
    print('log-likelihood of KDE by test dataset(sigma=0.03): {}'.format(
        kde_log_likelihood(train_dataset, test_dataset, 0.03)))
    print('log-likelihood of KDE by test dataset(sigma=0.2): {}'.format(
        kde_log_likelihood(train_dataset, test_dataset, 0.2)))
    print('log-likelihood of KDE by test dataset(sigma=0.8): {}'.format(
        kde_log_likelihood(train_dataset, test_dataset, 0.8)))

    print('log-likelihood of KNN by training dataset(k=2): {}'.format(KNN_likelihood(train_dataset, train_dataset, 2)))
    print('log-likelihood of KNN by training dataset(k=8): {}'.format(KNN_likelihood(train_dataset, train_dataset, 8)))
    print(
        'log-likelihood of KNN by training dataset(k=35): {}'.format(KNN_likelihood(train_dataset, train_dataset, 35)))
    print('log-likelihood of KNN by test dataset(k=2): {}'.format(KNN_likelihood(train_dataset, test_dataset, 2)))
    print('log-likelihood of KNN by test dataset(k=8): {}'.format(KNN_likelihood(train_dataset, test_dataset, 8)))
    print('log-likelihood of KNN by test dataset(k=35): {}'.format(KNN_likelihood(train_dataset, test_dataset, 35)))



# plt.figure()
# plt.plot(x_list_train, np.log(kde_train), color='g', label='kde_train')
# plt.plot(x_list_test, np.log(kde_test), color='r', label='kde_test')
# plt.plot(x_list_train, np.log(knn_train), color='b', label='knn_train')
# plt.plot(x_list_test, np.log(knn_test), color='y', label='knn_test')
# plt.legend(loc='best')
#
# plt.ylim(ymax = 1)
# plt.title("log likelihood of KNN density estimation ", fontsize=15)
# plt.legend()
# plt.show()


if __name__ == '__main__':
    # load dataset
    train_dataset = np.loadtxt('./dataSets/nonParamTrain.txt')
    test_dataset = np.loadtxt('./dataSets/nonParamTest.txt')

    # 3a
    plot_histogram(train_dataset, 0.02)
    plot_histogram(train_dataset, 0.5)
    plot_histogram(train_dataset, 2.0)

    # 3b
    # log likelihood
    sigma = [0.03, 0.2, 0.8]

    plt.figure()
    x_list1, kde_1 = kernel_density_estimate(train_dataset, 0.03)
    x_list2, kde_2 = kernel_density_estimate(train_dataset, 0.2)
    x_list3, kde_3 = kernel_density_estimate(train_dataset, 0.8)
    plt.plot(x_list1, kde_1, color='r', label='sigma = 0.03')
    plt.plot(x_list2, kde_2, color='g', label='sigma = 0.2')
    plt.plot(x_list3, kde_3, color='b', label='sigma = 0.8')
    plt.legend(loc='best')
    plt.grid()
    plt.title("Kernel Density Estimate (Gaussian) ", fontsize=15)
    plt.show()

    # compute  log likelihood of the data
    print('log-likelihood(sigma=0.03): {}'.format(kde_log_likelihood(train_dataset, train_dataset, 0.03)))
    print('log-likelihood(sigma=0.2): {}'.format(kde_log_likelihood(train_dataset, train_dataset, 0.2)))
    print('log-likelihood(sigma=0.8): {}'.format(kde_log_likelihood(train_dataset, train_dataset, 0.8)))

    # 3c
    plt.figure()
    x_list1, knn_1 = KNN(train_dataset, 2)
    x_list2, knn_2 = KNN(train_dataset, 8)
    x_list3, knn_3 = KNN(train_dataset, 35)
    plt.plot(x_list1, knn_1, color='g', label='K = 2')
    plt.plot(x_list1, knn_2, color='r', label='K = 8')
    plt.plot(x_list1, knn_3, color='b', label='K = 35')
    plt.legend(loc='best')
    plt.grid()
    plt.ylim(ymax=2)
    plt.title("KNN Density Estimation", fontsize=15)
    plt.show()
