import numpy as np
import os
from cvxopt import solvers
from cvxopt import matrix as mx
import matplotlib.pyplot as plt


def load_data(filename: str):
    file = os.path.join("dataSets/", filename)
    data = np.loadtxt(file)
    return data


def data_split(data):
    input = np.array(data[:, :2]).T

    label = np.zeros(N)
    for i in range(N):
        if data[i, 2] == 0:
            # ‘Setosa’
            label[i] = -1.0
        elif data[i, 2] == 2.0000:
            # ‘Virginica’
            label[i] = 1.0
    label = np.array(label).reshape((-1, N))

    return input, label


def RBF(x, y, sigma=1e-2):
    similarity = np.linalg.norm(x - y)
    return np.exp(-similarity/(2*np.square(sigma)))


def opt(input, label, sigma, slack):
    # initialize kern term
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = RBF(input[:, i], input[:, j], sigma)

    # assign quadratic coefficients
    P = mx(- (label.T @ label) * K)
    # assign linear coefficients
    q = mx(np.ones(N).astype('float'), (N, 1))
    # assign inequality constraints
    I = mx(0.0, (N, N))
    I[::N + 1] = 1.0
    G = mx([-I, I])
    h = mx([mx(0.0, (N, 1)), mx(slack, (N, 1))])
    # assign equality constraints
    A = mx(label.astype('float'), (1, N))
    b = mx([0.0])

    # solve opt problem
    sol = solvers.qp(P, q, G, h, A, b)

    print("final solution:")
    print(sol['x'])
    return sol


def svm(input, label, model:str):
    X = input.T
    y = label.reshape(-1)

    # step size in the mesh
    h = .02
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    color_map = {-1: (0, 0, .9), 0: (0, 0, .9), 1: (1, 0, 0)}

    if model == "diy":
        # SVM
        sol = opt(input, label, sigma=5e-2, slack=1.6)
        # boundary
        w = np.array(sol['x']).T * label @ input.T
        y_pre = w @ input + w[:, 0]
        # classification
        for i in range(N):
            if y_pre[:, i] > 0:
                y_pre[:, i] = 1
            else:
                y_pre[:, i] = -1
        # classification boundary
        Z = w @ np.c_[xx.ravel(), yy.ravel()].T + w[:, 0]
        for i in range(Z.shape[1]):
            if Z[:, i] > 0:
                Z[:, i] = 1
            else:
                Z[:, i] = -1
        Z = Z.reshape(xx.shape)
        # support vector
        alpha = np.array(sol['x'])
        for i in range(N):
            if alpha[i] < alpha.mean():
                alpha[i] = 0
            else:
                alpha[i] = 1
        support_vec = alpha * X
        mask = np.all(np.isnan(support_vec) | np.equal(support_vec, 0), axis=1)
        support_vec = support_vec[~mask]
        # wrong classification
        error_data = []
        for i in range(N):
            if y[i] != y_pre.reshape(-1)[i]:
                error_data.append(X[i, :])
        error_data = np.array(error_data)
    else:
        from sklearn import svm
        from sklearn.metrics import accuracy_score
        rbf_svc = svm.SVC(kernel='rbf', gamma=.05).fit(X, y)
        # support vector
        support_vec = rbf_svc.support_vectors_
        # prediction
        y_pre = rbf_svc.predict(X)
        # classification boundary
        Z = rbf_svc.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # wrong classification
        error_data = []
        for i in range(N):
            if y[i] != y_pre.reshape(-1)[i]:
                error_data.append(X[i, :])
        error_data = np.array(error_data)

    # plot support vector
    plt.figure(1)
    # Put the result into a color plot
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis('off')

    # Plot also the training points
    colors = [color_map[y] for y in y_pre.reshape(-1)]
    plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='black', s=50)
    plt.scatter(support_vec[:, 0], support_vec[:, 1], marker="x", c='w')
    plt.title("support vector")

    # plot support vector
    plt.figure(2)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis('off')
    # Plot also the training points
    colors = [color_map[y] for y in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='black', s=50)
    plt.scatter(error_data[:, 0], error_data[:, 1], marker="x", c='w')
    plt.title("misclassified samples")

    plt.show()
    return error_data


if __name__ == "__main__":
    # load data
    test = load_data("iris-pca.txt")
    # define dimension
    N, _ = test.shape
    # data split
    input, label = data_split(test)

    # SVM
    error_sample = svm(input, label, model='diy')
    # error_sample = svm(input, label, model='scikit')
    print(f"misclassified samples: {error_sample}")
