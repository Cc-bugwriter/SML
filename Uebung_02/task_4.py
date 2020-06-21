import numpy as np
import matplotlib.pyplot as plt


def Gaussian(x, cov, mu):
    x = np.array(x).reshape((2, -1))
    return np.exp((-0.5*(x-mu).T @ np.linalg.inv(cov) @ (x-mu))) / np.sqrt((2*np.pi) ** 2 * np.linalg.det(cov))

def plot_gmm(iter_num,K,data,mu,cov):
    plt.figure()
    plt.title('iteration = {}'.format(iter_num))
    plt.scatter(data[0,:], data[1,:])
    ax = plt.gca()
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    num = 100
    x = np.linspace(x_min, x_max, num)
    y = np.linspace(y_min, y_max, num)
    X, Y = np.meshgrid(x, y)
    colors = ['red','blue','green','orange']
    for k in range(K):
        Z = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                z = Gaussian(np.array([[x[i]], [y[j]]]), cov[k], mu[k])
                Z[j, i] = z
        plt.contour(X, Y, Z, colors=colors[k])




sigma = np.empty((N,2,2))
for i in range(N):
    d = data[:,i].reshape(2,-1)
    sigma[i] = np.dot(d, d.T)

likelihood = np.zeros((30,1))
for l in range(30):
    # E step
    for i in range(N):
        for j in range(K):
            R[i, j] = pi[j] * Gaussian(data[:, i], cov[j], mu[j])
        s = np.sum(R[i, :])
        R[i, :] = R[i, :] / s
    # M step
    N_j = np.sum(R, axis=0)
    # update pi mu cov
    for k in range(K):
        pi[k] = N_j[k] / N
        mu[k] = np.array((data @ R[:,k])).reshape((2,1)) / N_j[k]
        a = data - mu[k]
        # cov[k] = np.sum((data-mu[k]) @ (data-mu[k]).T * R[:, k].reshape((N,1,1)), axis=0) / N_j[k]
        # cov[k] = np.sum(np.array([(data[:,i] - mu[k]) @ (data[:,i] - mu[k]).T * R[:, k].reshape((N, 1, 1))
        #                           for i in range(N)]), axis=0) / N_j[k]
        cov[k] = np.sum(R[:,k].reshape((N,1,1)) * sigma, axis=0) / N_j[k] - (mu[k] @ mu[k].T).reshape(cov[k].shape)
    if l+1 in {1,3,5,10,30}:
        plot_gmm(l+1,K,data,mu,cov)

# plt.figure()
# plt.title('log')
# plt.xlabel('iteration')
# plt.ylabel('log-likelihood')
plt.grid()
plt.show()

if __name__ == '__main__':
    # load data
    data = np.loadtxt('./dataSets/gmm.txt').T

    # initialization
    K = 4
    N = data.shape[1]
    pi = np.array([0.25, 0.25, 0.25, 0.25])
    mu = [np.random.rand(2, 1) for i in range(K)]
    cov = [np.eye(2) for i in range(K)]
    R = np.zeros((N, K))

