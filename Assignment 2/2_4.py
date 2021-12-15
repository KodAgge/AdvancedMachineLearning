import numpy as np
import matplotlib.pyplot as plt
import Kruskal_v1 as kr
import Kruskal_v2 as kr2
import queue
from Tree import TreeMixture
from Tree import Tree
import sys
import random
from Phylogeny import tree_to_newick_rec
from tabulate import tabulate
import dendropy


def save_results(loglikelihood, topology_array, theta_array, filename):
    """ This function saves the log-likelihood vs iteration values,
        the final tree structure and theta array to corresponding numpy arrays. """

    likelihood_filename = filename + "_em_loglikelihood.npy"
    topology_array_filename = filename + "_em_topology.npy"
    theta_array_filename = filename + "_em_theta.npy"
    print("Saving log-likelihood to ", likelihood_filename, ", topology_array to: ", topology_array_filename,
          ", theta_array to: ", theta_array_filename, "...")
    np.save(likelihood_filename, loglikelihood)
    np.save(topology_array_filename, topology_array)
    np.save(theta_array_filename, theta_array)


def em_algorithm(seed_val, samples, num_clusters, max_num_iter=100, debugging = False):
    """
    This function is for the EM algorithm.
    :param seed_val: Seed value for reproducibility. Type: int
    :param samples: Observed x values. Type: numpy array. Dimensions: (num_samples, num_nodes)
    :param num_clusters: Number of clusters. Type: int
    :param max_num_iter: Maximum number of EM iterations. Type: int
    :return: loglikelihood: Array of log-likelihood of each EM iteration. Type: numpy array.
                Dimensions: (num_iterations, ) Note: num_iterations does not have to be equal to max_num_iter.
    :return: topology_list: A list of tree topologies. Type: numpy array. Dimensions: (num_clusters, num_nodes)
    :return: theta_list: A list of tree CPDs. Type: numpy array. Dimensions: (num_clusters, num_nodes, 2)

    This is a suggested template. Feel free to code however you want.
    """

    # Set the seeds
    np.random.seed(seed_val)
    num_start_tms = 100
    seeds = np.array(np.random.ranf((num_start_tms,)) * 1e5).astype(int)


    # Run the algorithm
    print("Running EM algorithm...")
    

    # Creating intital tms
    print("\nCreating initial", num_start_tms, "tree mixtures...")
    tree_mixtures = create_initial_tms(seeds, num_clusters, samples)
    
    if debugging:
        em_one_dimension(tree_mixtures[0], samples, num_clusters, 10)

        return 
        
    # First step of sieving
    print("\nRunning 10 iterations for the first", num_start_tms, "tree mixtures...\n")
    loglikelihoods = []
    j = 0
    for tm in tree_mixtures:
        loglikelihoods.append(em_one_dimension(tm, samples, num_clusters, 10))
        j += 1
        if j % 10 == 0:
            print(str(j) + " of the first", num_start_tms, "tree mixtures done")


    # Choosing 10 best tree mixtures
    num_best_trees = 10
    print("\nChoosing the", num_best_trees, "best tree mixtures")
    indices = np.argsort(-np.array(loglikelihoods))
    sieved_indices = indices[:num_best_trees]
    best_seeds = seeds[sieved_indices]


    # Recreating best tms
    print("\nRecreating the", num_best_trees, "best tree mixtures and running them for", max_num_iter, "iterations...\n")
    tree_mixtures = create_initial_tms(best_seeds, num_clusters, samples)


    # Running 100 iterations for the best trees
    loglikelihoods = []
    j = 0
    for tm in tree_mixtures:
        loglikelihoods.append(em_one_dimension(tm, samples, num_clusters, 100))
        j += 1
        print(str(j) + " of the first", num_best_trees, "tree mixtures done")

    # Choosing the best tree mixture
    print("\nChoosing the best tree mixtures and running it until convergence\n")
    indices = np.argsort(-np.array(loglikelihoods))
    best_seed = best_seeds[indices[0]]
    tree_mixtures = create_initial_tms([best_seed], num_clusters, samples)
    threshold = 1e-6

    tm, loglikelihoods = em_one_dimension_final(tree_mixtures[0], samples, num_clusters, max_num_iter, threshold)

    print("\n The final loglikelihood was", loglikelihoods[-1])

    topology_list, theta_list = get_arrays(tm, num_clusters)

    return loglikelihoods, topology_list, theta_list, tm.pi


def get_arrays(tm, num_clusters):
    topology_list = []
    theta_list = []
    for i in range(num_clusters):
        topology_list.append(tm.clusters[i].get_topology_array())
        theta_list.append(tm.clusters[i].get_theta_array())

    topology_list = np.array(topology_list)
    theta_list = np.array(theta_list)

    return topology_list, theta_list


def create_initial_tms(seeds, num_clusters, samples):
    tree_mixtures = []
    for seed in seeds:
        tm = TreeMixture(num_clusters=num_clusters, num_nodes=samples.shape[1])
        tm.simulate_pi(seed_val=seed)
        tm.simulate_trees(seed_val=seed)
        tree_mixtures.append(tm)
    
    return tree_mixtures


def em_one_dimension_final(tm, samples, num_clusters, max_num_iterations, threshold):
    d = tm.clusters[0].k
    loglikelihoods = []
    iteration = 0

    for _ in range(max_num_iterations):
        # Step 0 - Create new tree mixture
        tm_new = TreeMixture(num_clusters=num_clusters, num_nodes=samples.shape[1])

        # Step 1 - Calculate responsibilities
        r_new = responsibilities(tm, samples)

        # Step 2 - Calculate pi'
        tm_new.pi = new_pi(r_new)

        # Step 3 - Create G_k's
        tm_new.clusters = create_new_tms2(r_new, samples, num_clusters, d)

        iteration += 1
        loglikelihoods.append(log_likelihood(tm_new, samples)[0])  # Calculating loglikelihood

        tm = tm_new
        
        if iteration > 1:
            if loglikelihoods[-2] > loglikelihoods[-1]:
                print("!! The likelihood decreased !!")
            if abs(loglikelihoods[-1] / loglikelihoods[-2] - 1) < threshold:
                break

    if iteration < max_num_iterations:
        print("The algorithm converged after", iteration, "iterations")    
    else:
        print("The algorithm did not converge after", iteration, "iterations")

    return tm, loglikelihoods


def em_one_dimension(tm, samples, num_clusters, max_num_iterations):
    d = tm.clusters[0].k

    for _ in range(max_num_iterations):
        # Step 0 - Create new tree mixture
        tm_new = TreeMixture(num_clusters=num_clusters, num_nodes=samples.shape[1])

        # Step 1 - Calculate responsibilities
        r_new = responsibilities(tm, samples)

        # Step 2 - Calculate pi'
        tm_new.pi = new_pi(r_new)

        # Step 3 - Create G_k's
        tm_new.clusters = create_new_tms2(r_new, samples, num_clusters, d) # Adding the trees to the tree mixture

        tm = tm_new
    
    loglikelihood = log_likelihood(tm_new, samples) # Calculating loglikelihood

    return loglikelihood[0]


def create_new_tms(r_new, samples, num_clusters, d):
    clusters_new = []
    num_vertices = samples.shape[1]
    G_ks = [kr.Graph(num_vertices) for i in range(num_clusters)] # Creating a graph for every cluster
    for (k, G_k) in enumerate(G_ks):
        root_cdf = np.array([q_root(k, a, r_new, samples) for a in range(d)]) # Calculating the cdf for the root
        
        for s in range(num_vertices):
            for t in range(s):
                G_k.addEdge(t, s, mutual_information(k, t, s, r_new, samples, d)) # Assign value to edges

        # print(G_k.graph)
        max_G = G_k.maximum_spanning_tree() # Calculating the maximum spanning tree

        clusters_new.append(create_tree(max_G, k, r_new, samples, d, root_cdf)) # Adding the new tree 

    return clusters_new


def create_new_tms2(r_new, samples, num_clusters, d):
    clusters_new = []
    num_vertices = samples.shape[1]
    G_ks = [set() for i in range(num_clusters)] # Creating a graph for every cluster
    for (k, G_k) in enumerate(G_ks):
        root_cdf = np.array([q_root(k, a, r_new, samples) for a in range(d)]) # Calculating the cdf for the root
        
        for s in range(num_vertices):
            for t in range(s):
                G_k.add((t, s, mutual_information(k, t, s, r_new, samples, d))) # Assign value to edges

        vertices = list(range(num_vertices))
        graph = {
            'vertices': vertices,
            'edges': G_k
        }
        
        result = kr2.maximum_spanning_tree(graph) # Calculating the maximum spanning tree
        max_G = np.array([np.array([edge[0], edge[1]]) for edge in result])

        clusters_new.append(create_tree(max_G, k, r_new, samples, d, root_cdf)) # Adding the new tree 

    return clusters_new


def q_root(k, a, r, samples):
    return q_single(k, 0, a, r, samples)


def log_likelihood(tm, samples):
    return sum(np.log(responsibilities(tm, samples, True)))


def nan_function(x):
    if x == 0:
        return float('nan')
    else:
        return x


def create_tree(G, k, r, samples, d, root_cdf):
    topology_array, index_array = topology_index(G, d)
    # print(topology_array)
    # print(index_array)

    theta_array = t_array(k, topology_array, index_array, r, samples, d)

    theta_array.insert(0, root_cdf)

    t = Tree()
    t.load_tree_from_direct_arrays(np.array(topology_array), theta_array)
    # t.print()
    return t


def t_array(k, topology_array, index_array, r, samples, d):
    theta_array = []
    for s in range(1,len(topology_array)):
        t = topology_array[s] # Index of parent
        theta = np.array([[q_conditional(k, s, t, a, b, r, samples) for a in range(d)] for b in range(d)])
        theta_array.append(theta)

    return theta_array


def topology_index(G, d):
    topology_array = [float('nan')]
    index_array = []
    visited_rows = []
    # print(G)

    q = queue.Queue()
    q.put(0)
    i = 0

    # Possible bug: index not on same side of G

    while not q.empty():
        x = q.get()

        index_1, index_2 = np.where(G == x)
        index_array.append(x)
        for j in range(len(index_1)):      
            row = index_1[j]
            descendant = int(G[row, (index_2[j] + 1) % 2])
            if row not in visited_rows:
                visited_rows.append(row)
                topology_array.append(i)
                q.put(descendant)
        i += 1

    return topology_array, index_array
    

def mutual_information(k, s, t, r, samples, d):
    information = 0

    for a in range(d):
        q_a = q_single(k, s, a, r, samples)
        for b in range(d):
            q_b = q_single(k, t, b, r, samples)
            q_ab = q_joint(k, s, t, a, b, r, samples)
            if q_ab != 0:
                information += (q_ab + sys.float_info.epsilon) * np.log(q_ab / (q_a * q_b ) + sys.float_info.epsilon)

    return information


def q_conditional(k, s, t, a, b, r, samples):
    q = q_joint(k, s, t, a, b, r, samples) / q_single(k, t, b, r, samples)

    return q


def q_single(k, s, a, r, samples):
    denominator = sum(r[:,k]) # Sum of r_n,k over N
    # print(s, "=", a)
    column = samples[:,s]
    # print(column)
    # print(samples)
    indices = np.where(column == a)[0]
    # print(indices)
    # print(r)
    # print(r[indices,k])
    numerator = sum(r[indices,k])

    q = numerator / denominator
    # print(q)
    return q


def q_joint(k, s, t, a, b, r, samples):
    denominator = sum(r[:,k]) # Sum of r_n,k over N

    # print(s, "=", a)
    # print(t, "=", b)
    columns = samples[:,[s,t]]
    # print(columns)
    # print(samples)
    indices = np.unique(np.where((columns == [a,b]).all(axis=1))[0])
    # print(indices)

    numerator = sum(r[indices,k])
    # print(numerator)
    q = numerator / denominator
    # print(q)
    return q

    # return q
    

def responsibilities(tm, samples, log_prob = False):
    n_clusters = len(tm.clusters)
    n_samples = len(samples)

    p_matrix = np.zeros((n_samples, n_clusters))

    for i in range(n_samples): # Calculate probabilities of every sample for every cluster
        p_matrix[i,:] = [responsibility(tree.root, samples[i]) for tree in tm.clusters]
    
    r_unscaled = p_matrix * tm.pi # Multiply by categorical

    probabilities = r_unscaled.sum(axis=1).reshape((n_samples,1)) # Get normalizing factors

    if log_prob:
        return probabilities

    r = r_unscaled * probabilities ** (-1) # Normalize r

    return r


def new_pi(r):
    N = r.shape[0]
    return r.sum(axis=0) / N
    

def responsibility(node, sample):
    if node.ancestor == None:
        p = node.cat[sample[int(node.name)]] # Value of node
    else:
        p =  node.cat[sample[int(node.ancestor.name)]][sample[int(node.name)]] # Value of ancestor, value of node
    if len(node.descendants) > 0:
        for descendant in node.descendants:
            p *= responsibility(descendant, sample)
    return p


def rf_analysis_old(real_values_filename, output_filename):
    print("\n4.1.1 Loading ground truth trees from Newick files:\n")

    # If you want to compare two trees, make sure you specify the same Taxon Namespace!
    tns = dendropy.TaxonNamespace()

    filename = real_values_filename + "_tree_0_newick.txt"
    with open(filename, 'r') as input_file:
        newick_str = input_file.read()
    t0 = dendropy.Tree.get(data=newick_str, schema="newick", taxon_namespace=tns)
    print("\tTree 0: ", t0.as_string("newick"))
    t0.print_plot()

    filename = real_values_filename + "_tree_1_newick.txt"
    with open(filename, 'r') as input_file:
        newick_str = input_file.read()
    t1 = dendropy.Tree.get(data=newick_str, schema="newick", taxon_namespace=tns)
    print("\tTree 1: ", t1.as_string("newick"))
    t1.print_plot()

    filename = real_values_filename + "_tree_2_newick.txt"
    with open(filename, 'r') as input_file:
        newick_str = input_file.read()
    t2 = dendropy.Tree.get(data=newick_str, schema="newick", taxon_namespace=tns)
    print("\tTree 2: ", t2.as_string("newick"))
    t2.print_plot()

    filename = real_values_filename + "_tree_3_newick.txt"
    with open(filename, 'r') as input_file:
        newick_str = input_file.read()
    t3 = dendropy.Tree.get(data=newick_str, schema="newick", taxon_namespace=tns)
    print("\tTree 3: ", t3.as_string("newick"))
    t3.print_plot()

    filename = real_values_filename + "_tree_4_newick.txt"
    with open(filename, 'r') as input_file:
        newick_str = input_file.read()
    t4 = dendropy.Tree.get(data=newick_str, schema="newick", taxon_namespace=tns)
    print("\tTree 4: ", t4.as_string("newick"))
    t4.print_plot()


    print("\n4.1.2 Loading inferred trees")
    filename = output_filename + "_em_topology.npy"  # This is the result you have.
    topology_list = np.load(filename)
    # print(topology_list.shape)
    # print(topology_list)

    rt0 = Tree()
    rt0.load_tree_from_direct_arrays(topology_list[0])
    rt0 = dendropy.Tree.get(data=rt0.newick, schema="newick", taxon_namespace=tns)
    print("\tInferred tree 0: ", rt0.as_string("newick"))
    rt0.print_plot()

    rt1 = Tree()
    rt1.load_tree_from_direct_arrays(topology_list[1])
    rt1 = dendropy.Tree.get(data=rt1.newick, schema="newick", taxon_namespace=tns)
    print("\tInferred tree 1: ", rt1.as_string("newick"))
    rt1.print_plot()

    rt2 = Tree()
    rt2.load_tree_from_direct_arrays(topology_list[2])
    rt2 = dendropy.Tree.get(data=rt2.newick, schema="newick", taxon_namespace=tns)
    print("\tInferred tree 2: ", rt2.as_string("newick"))
    rt2.print_plot()

    rt3 = Tree()
    rt3.load_tree_from_direct_arrays(topology_list[3])
    rt3 = dendropy.Tree.get(data=rt3.newick, schema="newick", taxon_namespace=tns)
    print("\tInferred tree 3: ", rt3.as_string("newick"))
    rt3.print_plot()

    rt4 = Tree()
    rt4.load_tree_from_direct_arrays(topology_list[4])
    rt4 = dendropy.Tree.get(data=rt4.newick, schema="newick", taxon_namespace=tns)
    print("\tInferred tree 4: ", rt4.as_string("newick"))
    rt4.print_plot()

    print("\n4.1.3 Compare trees and print Robinson-Foulds (RF) distance:\n")

    print("\tt0 vs inferred trees")
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t0, rt0))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t0, rt1))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t0, rt2))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t0, rt3))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t0, rt4))

    print("\tt1 vs inferred trees")
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t1, rt0))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t1, rt1))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t1, rt2))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t1, rt3))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t1, rt4))

    print("\tt2 vs inferred trees")
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t2, rt0))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t2, rt1))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t2, rt2))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t2, rt3))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t2, rt4))

    print("\tt3 vs inferred trees")
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t3, rt0))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t3, rt1))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t3, rt2))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t3, rt3))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t3, rt4))

    print("\tt4 vs inferred trees")
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t4, rt0))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t4, rt1))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t4, rt2))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t4, rt3))
    print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t4, rt4))

    # print("\nInvestigate")

    # print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t0, rt0))
    # print("\tRF distance: \t", dendropy.calculate.treecompare.find_missing_bipartitions(t0, rt0))
    # print("\tRF distance: \t", dendropy.calculate.treecompare.false_positives_and_negatives(t0, rt0))
    # print("\tRF distance: \t", dendropy.calculate.treecompare.find_missing_bipartitions(t0, rt1))
    # print("\tRF distance: \t", dendropy.calculate.treecompare.false_positives_and_negatives(t0, rt1))


def rf_analysis(real_values_filename, output_filename, num_clusters):
    print("\n4.1.1 Loading ground truth trees from Newick files:\n")

    # If you want to compare two trees, make sure you specify the same Taxon Namespace!
    tns = dendropy.TaxonNamespace()

    realTrees = []
    for i in range(num_clusters):
        filename = real_values_filename + "_tree_" + str(i) + "_newick.txt"
        with open(filename, 'r') as input_file:
            newick_str = input_file.read()
        realTrees.append(dendropy.Tree.get(data=newick_str, schema="newick", taxon_namespace=tns))


    print("\n4.1.2 Loading inferred trees")
    filename = output_filename + "_em_topology.npy"  # This is the result you have.
    topology_list = np.load(filename)

    inferredTrees = []
    for i in range(num_clusters):
        rt = Tree()
        rt.load_tree_from_direct_arrays(topology_list[i])
        inferredTrees.append(dendropy.Tree.get(data=rt.newick, schema="newick", taxon_namespace=tns))

    print("\n4.1.3 Compare trees and print Robinson-Foulds (RF) distance:\n")
    for i in range(num_clusters):
        print("\tt" +str(i) + " vs inferred trees")
        for j in range(num_clusters):
            print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(realTrees[i], inferredTrees[j]))


def true_log_likelihood(real_values_filename, sample_filename, num_clusters):
    samples = np.loadtxt(sample_filename, delimiter="\t", dtype=np.int32)

    tm = TreeMixture(num_clusters=num_clusters, num_nodes=samples.shape[1])

    tm.load_mixture(real_values_filename)

    topology_array, theta_array = get_arrays(tm, num_clusters)

    print("\nStructure of the true trees:")
    for i in range(num_clusters):
        print("\n\tCluster: ", i)
        print("Pi: ", tm.pi[i])
        print("\tTopology: \t", topology_array[i])
        print("\tTheta: \t", theta_array[i])

    return log_likelihood(tm, samples)


def create_real_trees1():
    filename = "data/q2_4/more_samples"

    tm = TreeMixture(num_clusters=3, num_nodes=5)
    tm.load_mixture("data/q2_4/q2_4_tree_mixture.pkl")
    tm.samples = list()
    tm.sample_assignments = list()

    tm.sample_mixtures(1000, 1337)

    tm.save_mixture(filename)


def create_real_trees2():
    filename = "data/q2_4/more_nodes_less_clusters.pkl"

    tm = TreeMixture(num_clusters=2, num_nodes=7)

    tm.simulate_pi(seed_val=1337)
    tm.simulate_trees(seed_val=1337)

    tm.sample_mixtures(100, 1337)

    tm.save_mixture(filename, True)


def create_real_trees3():
    filename = "data/q2_4/less_nodes_more_clusters.pkl"

    tm = TreeMixture(num_clusters=5, num_nodes=4)

    tm.simulate_pi(seed_val=1337)
    tm.simulate_trees(seed_val=1337)

    tm.sample_mixtures(100, 1337)

    tm.save_mixture(filename, True)


def main(scenario = "normal"):
    seed_val = 123412567 #12341256

    if scenario == "normal":
        sample_filename = "data/q2_4/q2_4_tree_mixture.pkl_samples.txt"
        output_filename = "q2_4_results.txt"
        real_values_filename = "data/q2_4/q2_4_tree_mixture.pkl"
        num_clusters = 3
    elif scenario == "more samples":
        sample_filename = "data/q2_4/more_samples_samples.txt"
        output_filename = "q2_4_1_results.txt"
        real_values_filename = "data/q2_4/q2_4_tree_mixture.pkl"
        num_clusters = 3
    elif scenario == "ml":
        sample_filename = "data/q2_4/more_nodes_less_clusters.pkl_samples.txt"
        output_filename = "q2_4_2_results.txt"
        real_values_filename = "data/q2_4/more_nodes_less_clusters.pkl"
        num_clusters = 2
    elif scenario == "lm":
        sample_filename = "data/q2_4/less_nodes_more_clusters.pkl_samples.txt"
        output_filename = "q2_4_3_results.txt"
        real_values_filename = "data/q2_4/less_nodes_more_clusters.pkl"
        num_clusters = 5

    print("\n1. Load samples from txt file.\n")

    samples = np.loadtxt(sample_filename, delimiter="\t", dtype=np.int32)
    num_samples, num_nodes = samples.shape
    print("\tnum_samples: ", num_samples, "\tnum_nodes: ", num_nodes)
    print("\tSamples: \n", samples)

    print("\n2. Run EM Algorithm.\n")

    loglikelihood, topology_array, theta_array, pi = em_algorithm(seed_val, samples, num_clusters=num_clusters, debugging = False)

    print("\n3. Save, print and plot the results.\n")

    save_results(loglikelihood, topology_array, theta_array, output_filename)

    for i in range(num_clusters):
        print("\n\tCluster: ", i)
        print("Pi: ", pi[i])
        print("\tTopology: \t", topology_array[i])
        print("\tTheta: \t", theta_array[i])

    plt.figure(figsize=(8, 3))
    plt.subplot(121)
    plt.plot(np.exp(loglikelihood), label='Estimated')
    plt.ylabel("Likelihood of Mixture")
    plt.xlabel("Iterations")
    plt.subplot(122)
    plt.plot(loglikelihood, label='Estimated')
    plt.ylabel("Log-Likelihood of Mixture")
    plt.xlabel("Iterations")
    plt.legend(loc=(1.04, 0))
    plt.show()

    if real_values_filename != "":
        print("\n4. Retrieve real results and compare.\n")
        print("\tComparing the results with real values...")

        print("\t4.1. Make the Robinson-Foulds distance analysis.\n")
        rf_analysis(real_values_filename, output_filename, num_clusters)

        print("\n\t4.2. Make the likelihood comparison.\n")
        true_likelihood = true_log_likelihood(real_values_filename, sample_filename, num_clusters)

        data = [("EM", loglikelihood[-1], np.exp(loglikelihood[-1])),
                ("True", true_likelihood, np.exp(true_likelihood))]

        headers = ["","Loglikelihood","Likelihood"]

        print(tabulate(data, headers=headers))


# create_real_trees1()


# create_real_trees2()


# create_real_trees3()


if __name__ == "__main__":
    main()
