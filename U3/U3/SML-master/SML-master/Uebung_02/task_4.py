import numpy as np
import matplotlib.pyplot as plt


# task 4b
def multi_Gaussian(x, cov, mu):
    # convert vektor 1x2 to 2*1
    x = np.array(x).reshape((2, -1))
    p_x = np.exp((-0.5*(x - mu).T @ np.linalg.inv(cov) @ (x - mu))) / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(cov))
    return p_x


def plot_contour(iter_num, K, data, mu, cov):
    plt.figure()
    plt.title(f'iteration = {iter_num}')
    plt.scatter(data[:, 0], data[:, 1])

    # find limit of plot
    x_min, x_max = plt.gca().get_xlim()
    y_min, y_max = plt.gca().get_ylim()

    # assign contour variable
    num = 50
    x = np.linspace(x_min, x_max, num)
    y = np.linspace(y_min, y_max, num)
    X, Y = np.meshgrid(x, y)

    # define color space
    colors = ['red', 'blue', 'green', 'black']

    for k in range(K):
        # initial probability
        Z = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                z = multi_Gaussian(np.array([[x[i]], [y[j]]]), cov[k], mu[k])
                Z[j, i] = z

        plt.contour(X, Y, Z, colors=colors[k])

        plt.grid()
        plt.show()


def update_EMA(iter_num, data):
    ## initialization
    # number of class
    K = 4
    # number of matrix
    N = data.shape[0]
    # priori probability
    pi = 1 /K * np.ones(K)
    # mean of points
    mu = [np.random.rand(2, 1) for i in range(K)]
    # covariance Matrix
    cov = [np.eye(2) for i in range(K)]
    # posterior distribution
    R = np.zeros((N, K))

    # assign sigma
    sigma = np.empty((N, 2, 2))
    for i in range(N):
        d = data[i, :].reshape(2, -1)
        sigma[i] = np.dot(d, d.T)

    # initial list of pd
    likelihood_list = []

    # update iteration
    for l in range(iter_num):
        # initial likelihood
        likelihood = np.zeros(N)
        # E step
        for i in range(N):
            for j in range(K):
                p_phi_ij = pi[j] * multi_Gaussian(data[i, :], cov[j], mu[j])
                R[i, j] = p_phi_ij
                likelihood[i] += p_phi_ij
            # normalization
            R[i, :] = R[i, :] / np.sum(R[i, :])

        likelihood_list.append(np.sum(np.log(likelihood)))

        # M step
        N_j = np.sum(R, axis=0)

        # update pi mu cov
        for k in range(K):
            pi[k] = N_j[k] / N
            mu[k] = np.array((data.T @ R[:, k])).reshape((2, 1)) / N_j[k]
            cov[k] = np.sum(R[:, k].reshape((N, 1, 1)) * sigma, axis=0) / N_j[k] \
                     - (mu[k] @ mu[k].T).reshape(cov[k].shape)

    plot_contour(iter_num, K, data, mu, cov)

    if iter_num == 30:
        plt.figure()
        plt.plot(np.arange(1, 31), np.array(likelihood_list))
        plt.title('log-likelihood for every iteration')
        plt.xlabel('iteration')
        plt.ylabel('log-likelihood')
        plt.grid()
        plt.show()


if __name__ == '__main__':
    # load data
    data = np.loadtxt('./dataSets/gmm.txt')

    # assign iteration list
    t = [1, 3, 5, 10, 30]

    # 4b
    for iteration in t:
        update_EMA(iteration, data)
