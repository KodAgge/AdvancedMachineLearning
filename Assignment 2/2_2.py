""" This file is created as a suggested solution template for question 2.2 in DD2434 - Assignment 2.

    We encourage you to keep the function templates as is.
    However this is not a "must" and you can code however you like.
    You can write helper functions however you want.

    If you want, you can use the class structures provided to you (Node, Tree and TreeMixture classes in Tree.py
    file), and modify them as needed. In addition to the sample files given to you, it is very important for you to
    test your algorithm with your own simulated data for various cases and analyse the results.

    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format. Let us know if you face any problems.

    Also, we are aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). We wanted to keep the template codes as simple as possible.
    You can change the file names however you want (i.e tmm_1_samples.txt).

    For this assignment, we gave you three different trees (q_2_2_small_tree, q_2_2_medium_tree, q_2_2_large_tree).
    Each tree has 5 samples (whose inner nodes are masked with np.nan values).
    We want you to calculate the likelihoods of each given sample and report it.
"""

import numpy as np
from Tree import Tree
from Tree import Node


def calculate_likelihood(root, beta, k):
    """
    This function calculates the likelihood of a sample of leaves.
    :param: tree_topology: A tree topology. Type: numpy array. Dimensions: (num_nodes, )
    :param: theta: CPD of the tree. Type: numpy array. Dimensions: (num_nodes, K)
    :param: beta: A list of node assignments. Type: numpy array. Dimensions: (num_nodes, )
                Note: Inner nodes are assigned to np.nan. The leaves have values in [K]
    :return: likelihood: The likelihood of beta. Type: float.

    This is a suggested template. You don't have to use it.
    """

    print("Calculating the likelihood...")

    likelihood = recursive(root, beta, k)

    return likelihood


def recursive(node, beta, k):
    if len(node.descendants) == 0: # If the vertex is a leaf
        obs_index = int(beta[int(node.name)])
        theta = np.array(node.cat)
        theta = theta[:,obs_index] # Extract probabilities of the observed vertex given all values of parent
        return theta
    else:
        theta = np.array(node.cat)
        descendant_theta = np.ones(k)
        for descendant in node.descendants:
            descendant_theta *= recursive(descendant, beta, k) # Multiply probabilities of descendants for different values of node
        theta = np.dot(theta, descendant_theta) 
        return theta


def main(treeSize):    
    print("\nRunning algorithm for " + treeSize + " tree...")
    if treeSize == "small":
        filename = "data/q2_2/q2_2_small_tree.pkl"
    elif treeSize == "medium":
        filename = "data/q2_2/q2_2_medium_tree.pkl"
    else:
        filename = "data/q2_2/q2_2_large_tree.pkl"

    t = Tree()
    t.load_tree(filename)
    # t.print()

    for sample_idx in range(t.num_samples):
        beta = t.filtered_samples[sample_idx]
        sample_likelihood = calculate_likelihood(t.root, beta, t.k)
        print("\tLikelihood for sample " + str(sample_idx) + ": ", sample_likelihood)


def ownTree():
    topology_array = np.array([float('nan'), 0., 0., 1., 1.])
    theta_array = [
        np.array([0.2, 0.8]),
        np.array([[0.9, 0.1], [0.9, 0.1]]),
        np.array([[0.05, 0.95], [0.1, 0.9]]),
        np.array([[0.9, 0.1], [0.9, 0.1]]),
        np.array([[0.1, 0.9], [0.1, 0.9]])
    ]
    t = Tree()
    t.load_tree_from_direct_arrays(topology_array, theta_array)
    t.print()

    t.sample_tree(1)
    

    beta = t.filtered_samples[0]
    print(beta)
    sample_likelihood = calculate_likelihood(t.root, beta, t.k)
    print("\tLikelihood: ", sample_likelihood)

# ownTree()

if __name__ == "__main__":
    main("small")
    main("medium")
    main("large")
