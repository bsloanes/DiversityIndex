import numpy as np
import pandas as pd

def groupRun(gender_weights, gender_labels, re_weights, re_labels, eth_weights, eth_labels, simulation_size, sample_size):
    # Define an empty list to store the resulting data frames
    dfs = []
    #Store indices for race, gender, ethnicity, generate + append
    gender_indices = []
    re_indices = []
    eth_indices = []

    #matrices for the mean, stdv, quantiles
    # Loop through the weighted population procedure and store the resulting data frames and calculate indices
    for i in range(simulation_size):
        # Generate the sample population
        gender_sample = np.random.choice(np.arange(len(gender_weights)), size=sample_size, p=gender_weights)
        re_sample = np.random.choice(np.arange(len(re_weights)), size=sample_size, p=re_weights)
        eth_sample = np.random.choice(np.arange(len(eth_weights)), size=sample_size, p=eth_weights)
        gender_sample_labels = [gender_labels[i] for i in gender_sample]
        re_sample_labels = [re_labels[i] for i in re_sample]
        eth_sample_labels = [eth_labels[i] for i in eth_sample]
        df = pd.DataFrame({'Gender': gender_sample_labels, 'Race': re_sample_labels, 'Ethnicity': eth_sample_labels})


        # Append the resulting data frame to the list
        # perhaps necessary to run gini_simpson. can also try df['Gender'] w/o index
        dfs.append(df)

        gini_simpson_index_gender = 1 - np.sum(np.square(dfs[i]['Gender'].value_counts() / sample_size))
        gini_simpson_index_re = 1 - np.sum(np.square(dfs[i]['Race'].value_counts() / sample_size))
        gini_simpson_index_eth = 1 - np.sum(np.square(dfs[i]['Ethnicity'].value_counts() / sample_size))

        gender_indices.append(gini_simpson_index_gender)
        re_indices.append(gini_simpson_index_re)
        eth_indices.append(gini_simpson_index_eth)
    
    means = [np.mean(gender_indices), np.mean(re_indices), np.mean(eth_indices)]
    std_devs = [np.std(gender_indices), np.std(re_indices), np.std(eth_indices)]
    quantiles = [np.quantile(gender_indices, [0.025, 0.25, 0.5, 0.75, 0.975]),np.quantile(re_indices, [0.025, 0.25, 0.5, 0.75, 0.975]), np.quantile(eth_indices, [0.025, 0.25, 0.5, 0.75, 0.975])]
    
    return quantiles[0], quantiles[1], quantiles[2], std_devs[0], std_devs[1], std_devs[2], means[0], means[1], means[2], gender_indices, re_indices, eth_indices
    #g_quantiles = quantiles[1]
    #re_quantiles = quantiles[2]
    #print(g_quantiles[1])

    #print("Means g, r/e of G-S indices given sample size " +str(sample_size) + " and " + str(simulation_size) + " simulations : ", means)
    #print("Standard deviations g, r/e of G-S indicies given sample size " +str(sample_size) + " and " + str(simulation_size) + " simulations: ", std_devs)
    #print("Quantiles g, r/e of G-S indicies given sample size " +str(sample_size) + " and " + str(simulation_size) + " simulations: ", quantiles)
    
def makeArray(g_quantiles, re_quantiles, eth_quantiles, samplesIndex, simulation_sizesIndex, samples, simulation_sizes):
    for k in range(len(g_quantiles)):
        G_q = np.zeros(shape=(len(samples), len(simulation_sizes)), dtype=int)
        RE_q = np.zeros(shape=(len(samples), len(simulation_sizes)), dtype=int)
        ETH_q = np.zeros(shape=(len(samples), len(simulation_sizes)), dtype=int)
                   
        
        G_q[samplesIndex,simulation_sizesIndex] == g_quantiles[k]
        RE_q[samplesIndex,simulation_sizesIndex] == re_quantiles[k]
        ETH_q[samplesIndex,simulation_sizesIndex] == eth_quantiles[k]
        print(G_q)
        print(RE_q)
        print(ETH_q)

            
def fill_quantile_arrays(quantiles, simulation_sizes, population_sizes):
    num_quantiles = len(quantiles)
    num_simulations = len(simulation_sizes)
    num_populations = len(population_sizes)
    g_q = []
    for i in range(num_quantiles):
        g_q.append(np.zeros((num_simulations, num_populations)))
    for i in range(num_simulations):
        for j in range(num_populations):
            for k in range(num_quantiles):
                g_q[k][i, j] = quantiles[k]
    return g_q

        
        
        
        