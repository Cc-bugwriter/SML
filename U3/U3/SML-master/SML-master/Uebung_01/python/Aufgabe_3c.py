import numpy as np
import matplotlib.pyplot as plt


def benchmark_function(array_x=np.random.random((20, 1))):
    """
    evaluate Rosenbrock’s function under array_x
    :param array_x: [array], multidimensional variable (default value: np.random.random((20, 1)))
    :return:
    cost [float], the value of Rosenbrock’s function
    """
    # initial list of intermediate variables
    value_list = []

    # compute
    for i in range(array_x.shape[0]-1):
        f_x = 100*(array_x[i+1] - array_x[i]**2)**2 + (array_x[i] - 1)**2
        value_list.append(f_x)

    return sum(value_list)


def gradient_descent(array_x):
    """
    compute gradient  in Rosenbrock’s function
    :param array_x: [array], input array
    :return:
    [array], the gradient matrix,  size(n x 1)
    """
    # initial list of intermediate variables
    row_list = []

    # compute
    for i in range(array_x.shape[0]):
        # initial for each for
        row = []
        for j in range(array_x.shape[0] - 1):
            if i == j:
                d_fx = 200*(array_x[i+1] - array_x[i]**2)*(-2*array_x[i]) + 2*array_x[i]
                row.append(float(d_fx))
            elif i == j+1:
                d_fx = 200*(array_x[i] - array_x[i-1]**2)
                row.append(float(d_fx))
            else:
                row.append(0)
        row_list.append(row)

    # convert list to array, size(n-1 x n)
    gradient_matrix = np.array(row_list)

    # merge matrix in vector
    gradient_vector = np.sum(gradient_matrix, axis=1, dtype='float64').reshape(array_x.shape[0], 1)

    return gradient_vector


def var_updating(array_x, alpha=1e-3, epoch=500):
    """
    update variable with gradient descent in iteration
    :param array_x: [array], initial variable, size(n x 1)
    :param alpha: [float], learning rate (default value: 1e-3)
    :param epoch: [int] , time of iteration
    :return:
    """
    # initial cost record
    cost_hist = [benchmark_function(array_x)]

    # initial iteration record
    iteration = 0

    for i in range(epoch):
        # compute gradient
        gradient_vector = gradient_descent(array_x)

        # update array
        array_x = array_x - alpha * gradient_vector

        # compute cost
        cost = benchmark_function(array_x)

        cost_hist.append(cost)

        # assign necessary iteration
        if i> 10 and iteration == 0 and \
                (cost_hist[i-1]-cost_hist[i])/cost_hist[i] <= 1e-80:
            iteration = i
            continue

    return cost_hist, iteration


def plot_cost(cost_hist, alpha=1e-3, iteration=500):
    """
    plot evaluation history of bacterium with mutation
    :param cost_hist: , the record of evaluation history
    :param alpha: [float], learning rate (default value: 1e-3)
    :return:
    """
    # assign plot data in dict form
    data = {"cost": cost_hist,
            "iteration": np.linspace(0, len(cost_hist), len(cost_hist), dtype=np.int16)}

    # initial figure
    plt.plot("iteration", "cost", data=data, label=f'alpha = {alpha}, iteration = {iteration}')
    plt.grid()
    plt.xlabel("iteration")
    plt.ylabel("Rosenbrock’s function")
    plt.title("learning curve")

    return plt


if __name__ == '__main__':
    # set random seed
    np.random.seed(233)

    # set search space
    candidate_alpha = np.logspace(-10, -1, 10)

    plt.figure(1)

    for alpha in candidate_alpha:
        array_x = np.random.random((20, 1))
        hist, iteration = var_updating(array_x)
        plt = plot_cost(hist, alpha=alpha, iteration=iteration)

    plt.legend()
    plt.grid()
    plt.show()