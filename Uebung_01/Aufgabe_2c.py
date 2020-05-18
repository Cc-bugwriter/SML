import numpy as np
import matplotlib.pyplot as plt


def markov_chain(initial=np.array([1e-5, 1-1e-5]), epoch=20):
    """
    markov_chain to compute bacterium with specific mutation
    :param initial: [array], initial bacterium (default value
    :param epoch:[int], iteration times
    :return:
    mutation [list], the record of evaluation history
    """
    # assign probability matrix
    T = np.array([[.55, .023], [.45, .977]])

    # initial mutation hist list
    mutation = []

    mutation.append(initial[0])

    for i in range(epoch):
        initial = np.dot(T,initial)
        mutation.append(initial[0])

    return mutation


def plot_mutation(mutation):
    """
    plot evaluation history of bacterium with mutation
    :param mutation: , the record of evaluation history
    :return:
    """
    # assign plot data in dict form
    data = {"mutation_proportion": mutation,
            "iteration": np.linspace(0, len(mutation), len(mutation), dtype=np.int16)}

    # initial figure
    plt.figure()
    plt.plot("iteration", "mutation_proportion", data=data)
    plt.grid()
    plt.xlabel("iteration")
    plt.ylabel("mutation_proportion")
    plt.title('evaluation in iteration')

    plt.show()


if __name__ == '__main__':
    m = markov_chain(epoch=20)
    plot_mutation(m)