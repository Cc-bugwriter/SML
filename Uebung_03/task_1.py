import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    """
    load data for linear regression
    :param filename: [str]
    :return data: [array]
    """
    data = np.loadtxt(f"./dataSets/{filename}.txt")
    return data


# task a, b and c
def ridge_regression(data_train, data_test, coe_=0.01, n=1, plot=True):
    """
    ridge_regression use linear features
    :param data_train: [array], N*2 train data
    :param data_test: [array], N*2 test data
    :param coe_: [float], ridge coefficient
    :param n: [int], polynomials of degrees
    :param plot: [boolean], plot or not
    :return prediction: [array], N*1 prediction result of training data
    :return rmse_train: [float], root mean squared error of the train data
    :return rmse_test: [float], root mean squared error of the test data
    """
    def data_split(data, n=n):
        """
        split input data and target data,
        add bias in input data
        :param data: [array], N*2 input regression data
        :return X: [array], N*n input data with bias
        :return y: [array], N*1 target data
        """
        try:
            # input, target split
            X = data[:, 0].reshape(-1, 1)
            y = data[:, 1].reshape(-1, 1)

            # initial
            X_poly = np.ones(X.shape)
            # add a bias in input
            for i in range(1, n + 1):
                X_poly = np.hstack((X_poly, np.power(X, i)))
            return X_poly, y

        except IndexError:
            # prepare for plot
            X = data.reshape(-1, 1)

            # initial
            X_poly = np.ones(X.shape)
            # add a bias in input
            for i in range(1, n + 1):
                X_poly = np.hstack((X_poly, np.power(X, i)))
            return X_poly

    X_train, y_train = data_split(data_train)
    X_test, y_test = data_split(data_test)

    # calculate the weight matrix with training data set
    w = np.linalg.pinv(X_train.T @ X_train + coe_ * np.eye(X_train.shape[1])) @ X_train.T @ y_train

    # def multivariate_gaussian(data, mue, alpha):
    #     """
    #     Gaussian conditional probability
    #     :param data: [array], N*2 input data
    #     :param mue: [array], N*2 prior mean array
    #     :param alpha: [float], inverse variance coefficient
    #     :return W: [array], N*N Gaussian prior
    #     """
    #     # assign dimension
    #     N = data.shape[1]
    #
    #     # assign deviation
    #     corvariance_inv = alpha * np.eye(N)
    #     corvariance = np.linalg.inv(corvariance_inv)
    #
    #     # compute Gaussian conditional probability
    #     W_gaussian = 1/(np.sqrt(np.pi ** N * np.linalg.det(corvariance))) * \
    #         np.exp(-1/2 * (data - mue) @ corvariance_inv @ (data - mue).T)
    #
    #     return W_gaussian

    def loss(X, y, w, coe_=coe_):
        """
        loss function of ridge regression
        :param X: [array], N*n input data with bias
        :param w: [array], n*1 weight matrix
        :param coe_: output data
        :return phi: [float], loss function result
        :return rmse: [float], root mean squared error
        """
        phi = 1/2 * np.linalg.norm(X @ w - y) ** 2 + coe_/2 * np.linalg.norm(w) ** 2
        rmse = np.sqrt(np.linalg.norm(X @ w - y) ** 2 / len(y))
        return phi, rmse

    _, rmse_train = loss(X_train, y_train, w)
    _, rmse_test = loss(X_test, y_test, w)

    if plot:
        def plot_result(X, y, w):
            """
            plot prediction vs. target
            :param X: [array], N*n input data with bias
            :param y: [array], N*1 target data
            :param pre: [array], N*1 prediction data
            :return: None
            """
            # define independent variable
            x = np.linspace(X[:, 1].min(), X[:, 1].max(), 200)
            x = data_split(x)
            ax = plt.figure().gca()
            ax.plot(X[:, 1], y, 'ko', label='training data')
            ax.plot(x[:, 1], x @ w, 'b', label='predicted function')
            plt.title(f"prediction vs. target in training data,  polynomials of degrees={n}")
            plt.legend()
            plt.show()

        plot_result(X_train, y_train, w)

    return rmse_train, rmse_test


# task c
def fold_cross_validation(data_train, data_test, n=1, fold=5):
    """
    5 fold cross validation
    :param data_train: [array], N*2 train data
    :param data_test: [array], N*2 test data
    :param n: [int], polynomials of degrees
    :param fold: [int], fold number
    :return:
    """
    # assign fold length
    len_set = int(len(data_train)/fold)

    # initial return
    rmse_train_list = []
    rmse_val_list = []
    rmse_test_list = []

    # cross validation
    for i in range(1, fold+1):
        mask = np.zeros(data_train.shape)
        mask[(i-1)*len_set: i*len_set, :] = 1

        data_cv_test = data_train[mask.astype("bool")].reshape(-1, 2)
        data_cv_test = data_cv_test[np.all(data_cv_test != 0, axis=1)]

        data_cv_train = data_train[~mask.astype("bool")].reshape(-1, 2)
        data_cv_train = data_cv_train[np.all(data_cv_train != 0, axis=1)]

        rmse_train, rmse_val = ridge_regression(data_cv_train, data_cv_test, n=n, plot=False)
        _, rmse_test = ridge_regression(data_cv_train, data_test, n=n, plot=False)

        rmse_train_list.append(rmse_train)
        rmse_val_list.append(rmse_val)
        rmse_test_list.append(rmse_test)

    return np.array(rmse_train_list), np.array(rmse_val_list), np.array(rmse_test_list)


# task d, e, and f
def Bayesian_linear_ridge_regression(data_train, data_test, alpha=0.01, sigma=0.1, coe_=0.01,
                                     beta=10, SE=False, plot=True):
    """
    Implement Bayesian linear ridge regression
    :param data_train: [array], N*2 train data
    :param data_test: [array], N*2 test data
    :param alpha: [float], a single precision parameter
    :param sigma: [float], inverse noise precision parameter
    :param coe_: [float], ridge coefficient
    :param beta: [float], scale factor in squared exponential (SE) features
    :param SE: [boolean], whether use the squared exponential (SE) features
    :param SE: [boolean], whether use the squared exponential (SE) features
    :return:
    """
    def data_split(data, n=1):
        """
        split input data and target data,
        add bias in input data
        :param data:
        :return X: [array], N*n input data with bias
        :return y: [array], N*1 target data
        """
        # input, target split
        X = data[:, 0].reshape(-1, 1)
        y = data[:, 1].reshape(-1, 1)

        # initial
        X_poly = np.ones(X.shape)
        # add a bias in input
        if not SE:
            for i in range(1, n + 1):
                X_poly = np.hstack((X_poly, np.power(X, i)))
        else:
            k = 20
            for j in range(1, k + 1):
                X_poly = np.hstack((X_poly, np.exp(-beta/2*np.power(X-(j*0.1 -1), 2))))
        return X_poly, y

    X_train, y_train = data_split(data_train)
    X_test, y_test = data_split(data_test)

    # compute ridge coefficient or single precision parameter
    if not SE:
        # linear features,
        coe_ = alpha*sigma
    else:
        print(f"beta in squared exponential (SE) features: {beta}")
        # squared exponential (SE) features
        alpha = coe_/sigma
    # calculate the weight matrix with training data set
    w = np.linalg.pinv(X_train.T @ X_train + coe_ * np.eye(X_train.shape[1])) @ X_train.T @ y_train

    def prediction(X):
        """
        predict result
        :param X: [array], N*n input data
        :return: pre_mue, [array], N*1 prediction mean value
        :return: pre_sigma, [array], N*1 prediction standard deviations value
        """
        # assign mean value of prediction
        pre_mue = X @ w

        # calculate intermediate variables S_N
        S_N = np.linalg.pinv(alpha * np.eye(X.shape[1]) + 1 / sigma * X.T @ X)

        # calculate corvariance matrix
        pre_sigma_matrix = X @ S_N @ X.T
        pre_sigma_matrix += sigma * np.eye(pre_sigma_matrix.shape[1])

        # assign variance value of prediction
        pre_sigma = pre_sigma_matrix.diagonal().reshape(-1, 1)

        return pre_mue.reshape(-1, 1), np.sqrt(pre_sigma).reshape(-1, 1)

    # prediction regression result
    pre_mue_train, pre_sigma_train = prediction(X_train)
    pre_mue_test, pre_sigma_test = prediction(X_test)

    def metric(pre, y):
        """
        Compute metric of regression result
        :param pre: [array], N*1 prediction data
        :param y: [array], N*1 target data
        :return:
        """
        # compute RMSE
        rmse = np.sqrt(np.linalg.norm(pre - y) ** 2 / len(y))

        # compute average log-likelihood
        log_likeli = .0
        for i in range(len(y)):
            log_likeli += y.shape[1]/2 * np.log(1/(2*np.pi * sigma)) - 1/(2*sigma) * np.linalg.norm(y[i, :] - pre[i, :])
        log_likeli = log_likeli/len(y)

        return rmse, log_likeli

    # compute RMSE and average log-likelihood
    rmse_train, log_likeli_train = metric(pre_mue_train, y_train)
    print(f"RMSE of training data: {rmse_train}")
    print(f"average log-likelihood of training data: {log_likeli_train}")
    rmse_test, log_likeli_test = metric(pre_mue_test, y_test)
    print(f"RMSE of test data: {rmse_test}")
    print(f"average log-likelihood of test data: {log_likeli_test}")

    def plot_result(X, y, pre_mue, pre_var):
        """
        plot prediction vs. target
        :param X: [array], N*n input data with bias
        :param y: [array], N*1 target data
        :param pre_mue: [array], N*1 mean value of prediction data
        :param pre_var: [array], N*1 i standard deviations value of prediction data
        :return: None
        """
        ax = plt.figure().gca()
        ax.plot(X[:, 0], y, 'ko', label='training data')
        ax.plot(X[:, 0], pre_mue, 'b', label='mean of the predictive distribution')
        for i in range(1, 4):
            ax.fill_between(X[:, 0], (pre_mue + i * pre_var).reshape(-1),
                            (pre_mue - i * pre_var).reshape(-1), color='blue', alpha=0.15)
        if SE:
            plt.title(f"Bayesian linear ridge regression, beta={beta}")
        else:
            plt.title("Bayesian linear ridge regression")
        plt.legend()
        plt.show()

    if plot:
        sorted_indices = np.argsort(data_train[:, 0], axis=0)
        plot_result(data_train[sorted_indices], y_train[sorted_indices],
                    pre_mue_train[sorted_indices], pre_sigma_train[sorted_indices])

    # task f log-marginal
    def log_marginal_likelihood(X, y):
        """
        compute log-marginal likelihood of our Bayesian linear model
        :param X: [array], N*n input data with bias
        :param y: [array], N*1 target data
        :return:
        """
        # assign intermediate variables
        n = X_train.shape[0]

        # calculate intermediate variables S_N
        S_N = np.linalg.pinv(coe_ * np.eye(X.shape[1]) + 1 / sigma**2 * X.T @ X)

        log_marg_likeli = X_train.shape[1]/2 * np.log(coe_) - n/2*np.log(sigma**2) - \
                          1/2*np.linalg.norm(y - X @ w)**2 / sigma**2 + \
                          alpha/2 * w.T @ w - 1/2 * np.log(np.linalg.norm(S_N)) - n/2 * np.log(2*np.pi)
        return log_marg_likeli.flatten()[0]

    log_marg_likeli_train = log_marginal_likelihood(X_train, y_train)
    print(f"log-marginal likelihood of training data: {log_marg_likeli_train}")
    log_marg_likeli_test = log_marginal_likelihood(X_test, y_test)
    print(f"log-marginal likelihood of test data: {log_marg_likeli_test}")

    result = {"RMSE in Training": rmse_train,
              "log-likelihood in Training": log_likeli_train,
              "log-marginal likelihood in Training": log_marg_likeli_train,
              "RMSE in Test": rmse_test,
              "log-likelihood in Test": log_likeli_test,
              "log-marginal likelihood in Test": log_marg_likeli_test
              }
    print("------------------------------------")

    return result


# task f
def random_search(data_train, data_test):
    """
    implement a random search about SE in Bayesian linear ridge regression
    :param data_train: [array], N*2 train data
    :param data_test: [array], N*2 test data
    :return:
    """
    # define random search space
    beta_space = [1, 10, 100]

    # initial result dict
    res_dict = {"RMSE in Training": [],
                "log-likelihood in Training": [],
                "log-marginal likelihood in Training": [],
                "RMSE in Test": [],
                "log-likelihood in Test": [],
                "log-marginal likelihood in Test": []
                }
    # search and compare result
    for beta in beta_space:
        result = Bayesian_linear_ridge_regression(data_train, data_test, beta=beta, SE=True)

        # append result in dict
        res_dict["RMSE in Test"].append(result["RMSE in Training"])
        res_dict["log-likelihood in Test"].append(result["log-likelihood in Test"])
        res_dict["log-marginal likelihood in Test"].append(result["log-marginal likelihood in Test"])

    best_beta_list = []
    # find best beta
    for metric in ["RMSE in Test", "log-likelihood in Test", "log-marginal likelihood in Test"]:
        index = np.flatnonzero(np.abs(np.array(res_dict[metric])) == sorted(np.abs(np.array(res_dict[metric])))[0])
        best_beta = beta_space[int(index)]
        print(f"according to {metric}, best beta is {best_beta}")
        best_beta_list.append(best_beta)

    return best_beta_list


if __name__ == "__main__":
    data_train = load_data("lin_reg_train")
    data_test = load_data("lin_reg_test")

    # # task a
    # rmse_train, rmse_test = ridge_regression(data_train, data_test)
    # print(f"root mean squared error of the training data: {rmse_train}")
    # print(f"root mean squared error of the test data: {rmse_test}")
    #
    # # task b
    # for i in range(2, 5):
    #     rmse_train, rmse_test = ridge_regression(data_train, data_test, n=i)
    #     print(f"polynomials of degrees={i}, root mean squared error of the training data: {rmse_train}")
    #     print(f"polynomials of degrees={i}, root mean squared error of the test data: {rmse_test}")

    # task c
    # result = []
    # for i in range(2, 5):
    #     rmse_train_list, rmse_val_list, rmse_test_list = fold_cross_validation(data_train, data_test, n=i)
    #     print(f"polynomials of degrees={i}, root mean squared error of training: {np.mean(rmse_train_list)}")
    #     print(f"polynomials of degrees={i}, root mean squared error of validation: {np.mean(rmse_val_list)}")
    #     print(f"polynomials of degrees={i}, root mean squared error of test: {np.mean(rmse_test_list)}")
    #
    #     result.append([np.mean(rmse_train_list), np.mean(rmse_val_list), np.mean(rmse_test_list),
    #               np.std(rmse_train_list), np.std(rmse_val_list), np.std(rmse_test_list)])

    # # task d
    # Bayesian_linear_ridge_regression(data_train, data_test)

    # # task e
    # Bayesian_linear_ridge_regression(data_train, data_test, SE=True)

    # task f
    beta_list = random_search(data_train, data_test)

