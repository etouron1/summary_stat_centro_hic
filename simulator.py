import numpy as np
import random 
import torch
import pyro.distributions as pdist
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from itertools import combinations, product
import matplotlib.ticker as ticker

random.seed(1)

torch.manual_seed(0)

chr_seq_3_chr = {"chr01": 230209, "chr02": 813179, "chr03": 316619}
chr_cen_3_chr = {'chr01': 151584, 'chr02': 238325, 'chr03': 114499}

chr_seq_16_chr = {"chr01": 230209, "chr02": 813179, "chr03": 316619, "chr04": 1531918, 
           "chr05": 576869, "chr06": 270148, "chr07": 1090947, "chr08": 562644,
            "chr09": 439885, "chr10": 745746, "chr11": 666455, "chr12": 1078176,
            "chr13": 924430, "chr14": 784334, "chr15": 1091290, "chr16": 948063}

chr_cen_16_chr = {'chr01': 151584, 'chr02': 238325, 'chr03': 114499, 'chr04': 449819, 
                 'chr05': 152103, 'chr06': 148622, 'chr07': 497042, 'chr08': 105698, 
                 'chr09': 355742, 'chr10': 436418, 'chr11': 439889, 'chr12': 150946, 
                 'chr13': 268149, 'chr14': 628877, 'chr15': 326703, 'chr16': 556070}

def get_num_beads_and_start(chr_seq, sep):

    """ return a dict {chr : nb of beads}, a dict {chr : number of the start bead}, the total number of beads for all chr"""

    chr_bead = {}    # number of beads for each chromosome
    nbead = 0		# total number of beads for all chromosomes
    bead_start = {}  # bead label starts of a chr

    for i in chr_seq.keys(): # attention sorted()
        n = int(chr_seq[i] // sep) + int(chr_seq[i] % sep!=0)
        chr_bead[i] = n # number of beads for chromosome i
        
        nbead = nbead + n # total number of beads for all chromosmes
        bead_start[i] = nbead - n # the start bead for chr i
    return chr_bead, bead_start, nbead

def simulator(n_row, n_col, c_i, c_j, sig_2, intensity, noisy):
    """
    Simulate a noisy matrix C_{n_row x n_col} with a gaussian spot at (c_i, c_j) of size sig_2 
    """
    C = np.zeros((n_row, n_col))
    
    distr = pdist.MultivariateNormal(torch.tensor([c_i, c_j]), sig_2*torch.eye(2))
         
    # for i in range(len(C)):
    #     for j in range(len(C[0])):
    #         C[i,j] = intensity*torch.exp(distr.log_prob(torch.tensor([i,j])))
    indices = torch.tensor([[(i, j) for j in range(len(C[0]))] for i in range(len(C))])
    C = intensity*torch.exp(distr.log_prob(torch.tensor(indices)))
    
    if noisy:
        #noise = np.random.rand(n_row, n_col)*C.max()*0.1
        #mean = C.max()*0.1/2 
        mean = intensity*torch.exp(distr.log_prob(torch.tensor([c_i,c_j])))*0.1/2 
        sigma = intensity*torch.exp(distr.log_prob(torch.tensor([c_i,c_j])))*0.1/2
        #sigma = (C.max()*0.1/2)
        noise = np.random.normal(mean, sigma, size=(n_row, n_col))
        
        sig = np.sqrt(sig_2)
        # noise[c_i-int(sig) : c_i+int(sig)+1, :] = intensity*torch.exp(distr.log_prob(torch.tensor([c_i,c_j])))*0.05
        # noise[:, c_j-int(sig) : c_j+int(sig)+1] = intensity*torch.exp(distr.log_prob(torch.tensor([c_i,c_j])))*0.05
        i0 = max(0, c_i - int(sig))
        i1 = min(n_row, c_i + int(sig) + 1)

        j0 = max(0, c_j - int(sig))
        j1 = min(n_col, c_j + int(sig) + 1)

        noise[i0:i1, :] = 0
        noise[:, j0:j1] = 0 
        # noise[c_i-int(sig) : c_i+int(sig)+1, :] = 0
        # noise[:, c_j-int(sig) : c_j+int(sig)+1] = 0
        

    else:
        noise = np.zeros_like(C)
    
    return C+noise

def plot_C(C, c_i, c_j, sig_2, intensity, c_i_ref, c_j_ref, corr):
    """
    plot the noisy matrix C indicating the true VS simulated parameters
    """
    n_row = len(C)
    n_col = len(C[0])
    fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)
    draw = ax.matshow(
            C,cmap="YlOrRd",origin="upper")
    ax.tick_params(left=True, bottom=False, top=True, right=False, labelbottom=False, labeltop=True)
    fig.colorbar(draw, location='right')
    ax.scatter(c_j, c_i, color="green", marker = '+', s=5, label =rf"$\theta = {(c_i, c_j)}$" )
    ax.scatter(c_j_ref, c_i_ref, color="black", marker = '+', s=5, label =rf"$\theta_0 = {(c_i_ref, c_j_ref)}$" )
    plt.legend()
    plt.title(rf"scale $\alpha = {intensity}$, $\mathcal{{N}}({(c_i, c_j)}, {sig_2:.3f})$"+"\n"+ f"corr = {corr}")
    plt.axis([0, n_col, n_row, 0])
    plt.show()
    #plt.savefig(f"{c_i}_{corr:.4f}_accepted_0.05.svg")


def Pearson_correlation_vector(C_ref, C_simu):
    """
    Compute the vector-based Pearson correlation between 2 contact matrices
    -> the matrix is vectorized
    """
    
    C_ref_upper_vector = C_ref.flat

    
    C_simu_upper_vector = C_simu.flat
   
    
    n = len(C_ref_upper_vector)
    
    sum_exp_ref = 0
    sum_exp_simu = 0
    sum_exp_exp = 0
    sum_exp_carre_simu = 0
    sum_exp_carre_ref = 0
    for i in range(len(C_ref_upper_vector)):
        sum_exp_exp += C_ref_upper_vector[i]*C_simu_upper_vector[i]
        sum_exp_ref += C_ref_upper_vector[i]
        sum_exp_simu += C_simu_upper_vector[i]
        sum_exp_carre_ref += C_ref_upper_vector[i]**2
        sum_exp_carre_simu += C_simu_upper_vector[i]**2
    num = n*sum_exp_exp - sum_exp_ref*sum_exp_simu
    denom = np.sqrt(n*sum_exp_carre_ref - sum_exp_ref**2) * np.sqrt(n*sum_exp_carre_simu - sum_exp_simu**2)
    
    if denom == 0 or np.isnan(denom):
        corr = 0
    else:
        corr = num/denom   
        # print("vector", corr)
        # print("vector", pearsonr(C_ref_upper_vector, C_simu_upper_vector)[0])     
    return corr


def Pearson_correlation_row(C_ref, C_simu):
    """
    Compute the Average row-based Pearson correlation between 2 contact matrices
    """
    
    n = len(C_ref[0])
    corr = 0
    for i in range(len(C_ref)):
        sum_exp_ref = 0
        sum_exp_simu = 0
        sum_exp_exp = 0
        sum_exp_carre_simu = 0
        sum_exp_carre_ref = 0
        for j in range(len(C_ref[0])):
            #if j!=i:
                sum_exp_exp += C_ref[i,j]*C_simu[i,j]
                sum_exp_ref += C_ref[i,j]
                sum_exp_simu += C_simu[i,j]
                sum_exp_carre_ref += C_ref[i,j]**2
                sum_exp_carre_simu += C_simu[i,j]**2
        num = n*sum_exp_exp - sum_exp_ref*sum_exp_simu
        denom = np.sqrt(n*sum_exp_carre_ref - sum_exp_ref**2) * np.sqrt(n*sum_exp_carre_simu - sum_exp_simu**2)
        
        if denom == 0 or np.isnan(denom):
            
            corr_tmp = 0
        else:
            corr_tmp = num/denom
            # print("row", corr_tmp)
            # print("row", pearsonr(C_ref[i], C_simu[i])[0])
        corr += corr_tmp
    return corr/len(C_ref)

def Pearson_correlation_col(C_ref, C_simu):
    """
    Compute the Average column-based Pearson correlation between 2 contact matrices
    """
    
    n = len(C_ref)
    corr = 0
    for j in range(len(C_ref[0])):
        sum_exp_ref = 0
        sum_exp_simu = 0
        sum_exp_exp = 0
        sum_exp_carre_simu = 0
        sum_exp_carre_ref = 0
        for i in range(len(C_ref)):
            if i!=j:
                sum_exp_exp += C_ref[i,j]*C_simu[i,j]
                sum_exp_ref += C_ref[i,j]
                sum_exp_simu += C_simu[i,j]
                sum_exp_carre_ref += C_ref[i,j]**2
                sum_exp_carre_simu += C_simu[i,j]**2
        num = n*sum_exp_exp - sum_exp_ref*sum_exp_simu
        denom = np.sqrt(n*sum_exp_carre_ref - sum_exp_ref**2) * np.sqrt(n*sum_exp_carre_simu - sum_exp_simu**2)
        
        if denom == 0 or np.isnan(denom):
            
            corr_tmp = 0
        else:
            corr_tmp = num/denom
            # print("col", corr_tmp)
            # print("col", pearsonr(C_ref[j], C_simu[j])[0])
        corr += corr_tmp
    return corr/len(C_ref[0])

def Correlation_inter_row_average(chr_seq, C_simu, C_ref, resolution, corr_function):
    """ 
    Compute the correlation for each inter-chr contact matrix
    Return : the average of all inter-chr correlations on all the blocks
    """
    
    num_chr_num_bead, chr_bead_start, nb_bead = get_num_beads_and_start(chr_seq, resolution)

    corr_inter = 0
        
    for chr in chr_seq.keys():
            
        start, end = chr_bead_start[chr], chr_bead_start[chr] + num_chr_num_bead[chr] #get the start bead id and the end bead id for the chr
        inter_simu = C_simu[:, start:end] 
        inter_ref = C_ref[:, start:end] 
        corr_inter += corr_function(inter_ref, inter_simu)
    return corr_inter/len(chr_seq)

from  scipy.special import binom
def Correlation_inter_upper_average(chr_seq, C_simu, C_ref, resolution, corr_function):
    """ 
    Compute the correlation for each inter-chr contact matrix
    Return : the average of all inter-chr correlations on all the blocks
    """
    
    num_chr_num_bead, chr_bead_start, nb_bead = get_num_beads_and_start(chr_seq, resolution)
    
    corr_inter = 0
        
    for (chr_1, chr_2) in combinations(chr_seq.keys(), r=2):
        
        start_1, end_1 = chr_bead_start[chr_1], chr_bead_start[chr_1] + num_chr_num_bead[chr_1] #get the start bead id and the end bead id for the chr
        start_2, end_2 = chr_bead_start[chr_2], chr_bead_start[chr_2] + num_chr_num_bead[chr_2] #get the start bead id and the end bead id for the chr
        inter_simu = C_simu[start_1:end_1, start_2:end_2] 
        inter_ref = C_ref[start_1:end_1, start_2:end_2] 

        corr_inter += corr_function(inter_ref, inter_simu)
    return corr_inter/binom(len(chr_seq), 2)


def Correlation_inter_upper_average_wavelets(chr_seq, approx_coeffs_simu, approx_coeffs_ref, resolution, corr_function):
    """
    Given lists of approximation wavelets coeff, return the average correlation over each bloc average over all the coeffs.
    """

    num_chr_num_bead, chr_bead_start, nb_bead = get_num_beads_and_start(chr_seq, resolution)
    nb_levels = len(approx_coeffs_ref)

    corr_inter = 0

    for level in range(nb_levels):
        # fig, axes = plt.subplots(2, 3, figsize=[14, 8])


        corr_coeff = 0
        # a = 0
        for (chr_1, chr_2) in combinations(chr_seq.keys(), r=2):
                
            start_1, end_1 = chr_bead_start[chr_1], chr_bead_start[chr_1] + num_chr_num_bead[chr_1] #get the start bead id and the end bead id for the chr
            start_2, end_2 = chr_bead_start[chr_2], chr_bead_start[chr_2] + num_chr_num_bead[chr_2] #get the start bead id and the end bead id for the chr
            
            inter_simu = approx_coeffs_simu[level][start_1//2**level+1:end_1//2**level+1, start_2//2**level+1:end_2//2**level+1] 
            inter_ref = approx_coeffs_ref[level][start_1//2**level+1:end_1//2**level+1, start_2//2**level+1:end_2//2**level+1] 
            # axes[1, a].imshow(inter_simu, cmap="YlOrRd")
            # axes[0, a].imshow(inter_ref, cmap="YlOrRd")
            corr_coeff += corr_function(inter_ref, inter_simu)
          
        corr_inter += corr_coeff/binom(len(chr_seq), 2)
        # plt.suptitle(f"level {level}")
        # plt.show()
    return corr_inter/nb_levels
     
def Correlation_inter_average_row(chr_seq, C_simu, C_ref, resolution, corr_function, theta=None, theta_corr_dict=None):
    """
    Input : 
    - C_simu, C_ref : upper strict triangular matrix per bloc
    - theta : {chr : centro_simu}
    - theta_corr_dict : {chr : {centro_simu : corr}}
    """
    num_chr_num_bead, chr_bead_start, nb_bead = get_num_beads_and_start(chr_seq, resolution)
    C_simu = C_simu + np.transpose(np.triu(C_simu, k=1))
    C_ref = C_ref + np.transpose(np.triu(C_ref, k=1))
    corr = 0
    for chr_1 in chr_seq.keys():
        # corr_chr = 0
        for chr_2 in chr_seq.keys():
            if chr_2 != chr_1:
                
                start_1, end_1 = chr_bead_start[chr_1], chr_bead_start[chr_1] + num_chr_num_bead[chr_1] #get the start bead id and the end bead id for the chr
                start_2, end_2 = chr_bead_start[chr_2], chr_bead_start[chr_2] + num_chr_num_bead[chr_2] #get the start bead id and the end bead id for the chr
                inter_simu = C_simu[start_1:end_1, start_2:end_2] 
                inter_ref = C_ref[start_1:end_1, start_2:end_2] 
                
                # corr_chr += corr_function(inter_ref, inter_simu)
                corr += corr_function(inter_ref, inter_simu)
        
        # theta_corr_dict[chr_1][theta[chr_1]] = corr_chr/(len(list(chr_seq.keys()))-1)
    dim = len(list(chr_seq.keys()))
    print(dim)
    print(corr / (dim**2 - dim))
    
        
def Correlation_inter_vector(C_simu, C_ref, resolution, corr_function):
    """
    Vectorize all the upper inter-chr contact matrices and compute the Pearson correlation
    """
     
    num_chr_num_bead, chr_bead_start, nb_bead = get_num_beads_and_start(chr_seq, resolution)

    corr_inter = 0
    inter_simu = np.array([])  
    inter_ref = np.array([])  
    for (chr_1, chr_2) in combinations(chr_seq.keys(), r=2):
            
        start_1, end_1 = chr_bead_start[chr_1], chr_bead_start[chr_1] + num_chr_num_bead[chr_1] #get the start bead id and the end bead id for the chr
        start_2, end_2 = chr_bead_start[chr_2], chr_bead_start[chr_2] + num_chr_num_bead[chr_2] #get the start bead id and the end bead id for the chr
        
        if len(inter_simu) == 0:
            inter_simu = C_simu[start_1:end_1, start_2:end_2].flatten()
            
            inter_ref = C_ref[start_1:end_1, start_2:end_2].flatten()
        else:
            inter_simu = np.concatenate((inter_simu, C_simu[start_1:end_1, start_2:end_2].flatten()), axis=0)
            inter_ref = np.concatenate((inter_ref, C_ref[start_1:end_1, start_2:end_2].flatten()), axis=0)

    
    corr_inter = corr_function(inter_ref, inter_simu)
    return corr_inter


import scipy.stats as ss
def Spearman_correlation_vector(C_ref, C_simu):
    """
    Compute the vector-based Spearman correlation between 2 contact matrices
    -> the matrix is vectorized
    """
    C_ref_vector = C_ref.flat
    C_simu_vector = C_simu.flat
    rank_C_ref = ss.rankdata(C_ref_vector)
    rank_C_simu = ss.rankdata(C_simu_vector)
    return Pearson_correlation_vector(rank_C_ref, rank_C_simu)

def Spearman_correlation_row(C_ref, C_simu):
    """
    Compute the Average row-based Pearson correlation between 2 contact matrices
    """

    corr = 0
    for i in range(len(C_ref)):
        rank_C_ref = ss.rankdata(C_ref[i])
        rank_C_simu = ss.rankdata(C_simu[i])
        corr += Pearson_correlation_vector(rank_C_ref, rank_C_simu)
    return corr/len(C_ref)

def Spearman_correlation_col(C_ref, C_simu):
    """
    Compute the Average column-based Spearman correlation between 2 contact matrices
    """
    corr = 0
    for j in range(len(C_ref[0])):
        rank_C_ref = ss.rankdata(C_ref[:,j])
        rank_C_simu = ss.rankdata(C_simu[:,j])
        corr += Pearson_correlation_vector(rank_C_ref, rank_C_simu)
    return corr/len(C_ref[0])

def simulator_C(chr_seq, resolution, theta, sig_2, intensity, noisy):
     
    nb_bead, start_bead, nb_tot_bead = get_num_beads_and_start(chr_seq, resolution)

    C_simu = np.zeros((nb_tot_bead,nb_tot_bead))

    for (chr_row, chr_col) in combinations(chr_seq.keys(),r=2):
        n_row = chr_seq[chr_row]//resolution
        n_col = chr_seq[chr_col]//resolution
   
        c_i_simu = theta[chr_row]//resolution
        c_j_simu = theta[chr_col]//resolution
        sig_2_simu, intensity_simu = sig_2, intensity
        
        C_simu[start_bead[chr_row]:start_bead[chr_row]+nb_bead[chr_row]-1, start_bead[chr_col]:start_bead[chr_col]+nb_bead[chr_col]-1] = simulator(n_row,n_col, c_i_simu, c_j_simu, sig_2_simu, intensity_simu, noisy=noisy)
    return C_simu

def simulator_genome():
    
    resolution = 10000
    nb_bead, start_bead, nb_tot_bead = get_num_beads_and_start(chr_seq, resolution)
    ################# simulate C_ref #########################
    sig_2_ref = 2
    intensity_ref = 100000
    # C_ref = np.zeros((nb_tot_bead,nb_tot_bead))
    # for (chr_row, chr_col) in combinations(chr_seq.keys(), r=2):
    #     n_row = chr_seq[chr_row]//resolution
    #     n_col = chr_seq[chr_col]//resolution
    #     c_i_ref = chr_cen[chr_row]//resolution
    #     c_j_ref = chr_cen[chr_col]//resolution
    #     C_ref[start_bead[chr_row]:start_bead[chr_row]+nb_bead[chr_row]-1, start_bead[chr_col]:start_bead[chr_col]+nb_bead[chr_col]-1] = simulator(n_row,n_col, c_i_ref, c_j_ref, sig_2_ref, intensity_ref, noisy=1)
    # # # plot_C_genome(C_ref, resolution, sig_2_ref, intensity_ref, chr_cen)
    # with open('simulation_little_genome/noisy/C_ref', 'wb') as f:
    #         pickle.dump(C_ref, f)
    ###########################################################
    with open('simulation_little_genome/noisy/C_ref', 'rb') as f:
            C_ref = pickle.load(f)
    #plot_C_genome(C_ref, resolution, sig_2_ref, intensity_ref, chr_cen)
    # C_1_ref = simulator(100,100, 2, 3, 2, 100, noisy=0)
    # C_1 = simulator(100,100, 2, 3, 2, 100, noisy=1)
    
    # C_2_ref = simulator(100,300, 2, 3, 2, 100, noisy=0)
    # C_2 = simulator(100,300, 2, 3, 2, 100, noisy=1)
    # #plot_C(C_2, 2, 3, 2, 100, 2, 3, 1)
    # print(Pearson_correlation_vector(C_1,C_1_ref))
    # print(Pearson_correlation_vector(C_2,C_2_ref))
    # s_corr_row = []
    # s_corr_col = []
    # s_corr_vector = []
    #p_corr_row = []
    p_corr_row_col = []
    p_corr_vector = []
    with open('simulation_little_genome/noisy/theta', 'rb') as f:
            theta = pickle.load(f)
    with open('simulation_little_genome/noisy/param', 'rb') as f:
            param = pickle.load( f)
    
    # theta = []
    # param = []
    for k in range(0):
        print(k)
        ############# simulate theta, sig_2, intensity ##########
        # centro = {}
        # sig_2 = random.uniform(0.1, 10)
        # intensity = random.choice(range(1, 1001))
        # for chr in chr_seq.keys():
        #     c = pdist.Uniform(low=1, high=chr_seq[chr]-1).sample()
        #     centro[chr]=int(c.detach().item())

        # theta.append(centro)
        # param.append((sig_2, intensity))
        #########################################################
        C_simu = np.zeros((nb_tot_bead,nb_tot_bead))

        for (chr_row, chr_col) in combinations(chr_seq.keys(),r=2):
            n_row = chr_seq[chr_row]//resolution
            n_col = chr_seq[chr_col]//resolution
            # c_i_simu = centro[chr_row]//resolution
            # c_j_simu = centro[chr_col]//resolution
            # c_i_simu = chr_cen[chr_row]//resolution
            # c_j_simu = chr_cen[chr_col]//resolution
            # sig_2_simu = sig_2
            # intensity_simu = intensity
            c_i_simu = theta[k][chr_row]//resolution
            c_j_simu = theta[k][chr_col]//resolution
            sig_2_simu, intensity_simu = param[k]
            
            C_simu[start_bead[chr_row]:start_bead[chr_row]+nb_bead[chr_row]-1, start_bead[chr_col]:start_bead[chr_col]+nb_bead[chr_col]-1] = simulator(n_row,n_col, c_i_simu, c_j_simu, sig_2_simu, intensity_simu, noisy=1)
        
        #plot_C_genome(C_simu, resolution, sig_2_simu, intensity_simu, theta[k])
        
        # s_corr_row.append(Correlation_inter(C_simu, C_ref, resolution, Spearman_correlation_row))
        # s_corr_col.append(Correlation_inter(C_simu, C_ref, resolution, Spearman_correlation_col))
        # s_corr_vector.append(Correlation_inter(C_simu, C_ref, resolution, Spearman_correlation_vector))
        p_corr_row_col.append(0.5*(Correlation_inter_upper_average(C_simu, C_ref, resolution, Pearson_correlation_row)+Correlation_inter_upper_average(C_simu, C_ref, resolution, Pearson_correlation_col)))
        p_corr_vector.append(Correlation_inter_upper_average(C_simu, C_ref, resolution, Pearson_correlation_vector))
        
        # p_corr_vector.append(Correlation_inter_vector(C_simu, C_ref, resolution, Pearson_correlation_vector))

    # with open('simulation_little_genome/clear/theta', 'wb') as f:
    #         pickle.dump(theta, f)
    # with open('simulation_little_genome/clear/param', 'wb') as f:
    #         pickle.dump(param, f)
    # with open('simulation_little_genome/clear/S_corr_inter_vector', 'wb') as f:
    #         pickle.dump(s_corr_vector, f)
    # with open('simulation_little_genome/clear/S_corr_inter_col', 'wb') as f:
    #         pickle.dump(s_corr_col, f)
    # with open('simulation_little_genome/clear/S_corr_inter_row', 'wb') as f:
    #         pickle.dump(s_corr_row, f)
    # with open('simulation_little_genome/clear/P_corr_inter_vector', 'wb') as f:
    #         pickle.dump(p_corr_vector, f)
    # with open('simulation_little_genome/clear/P_corr_inter_col', 'wb') as f:
    #         pickle.dump(p_corr_col, f)
    with open('simulation_little_genome/noisy/P_corr_inter_row_col', 'wb') as f:
            pickle.dump(p_corr_row_col, f)
    with open('simulation_little_genome/noisy/P_corr_inter_vector', 'wb') as f:
            pickle.dump(p_corr_vector, f)

    # with open("simulation_little_genome/clear/theta", 'rb') as f:
    #         theta = pickle.load(f)
    # with open('simulation_little_genome/clear/corr_inter_vector', 'rb') as f:
    #         corr_vector = pickle.load( f)
    # with open('simulation_little_genome/clear/corr_inter_row', 'rb') as f:
    #         corr_row = pickle.load( f)
    # with open('simulation_little_genome/clear/corr_inter_col', 'rb') as f:
    #         corr_col = pickle.load( f)
    
    # corr = {}
    # corr["row"] = corr_row
    # corr["col"] = corr_col
    # corr["vector"] = corr_vector
    #plotly_colormap_corr_theta(theta, corr)
    # plot_colormap_corr_theta(theta, corr_row, "row")
    # plot_colormap_corr_theta(theta, corr_vector, "vector")
    # plot_colormap_corr_theta(theta, corr_col, "col")



def plot_C_genome(chr_seq, chr_cen, C, resolution, sig_2_ref, intensity_ref, centro):
    """ plot the contact matrix for all the chr """

    nb_bead, start_bead, nb_tot_bead = get_num_beads_and_start(chr_seq, resolution)
    end_chr = list(np.array(list(chr_seq.values()))+np.array([0]+list(chr_seq.values())[:-1])+np.array([0,0]+list(chr_seq.values())[:-2]))
    centro_chr = list(np.array([0]+end_chr[:-1])+np.array(list(chr_cen.values())))
    
    #theta_centro_chr = list(np.array([0]+end_chr[:-1])+np.array(list(centro.values())))
    theta_centro_chr = list(np.array([0]+end_chr[:-1])+np.array(centro))
    nb_chr =len(chr_seq.keys())
    
    fig, ax = plt.subplots(figsize=(nb_chr*16, nb_chr*16), tight_layout=True)

    
    draw = ax.matshow(C,cmap="YlOrRd",origin="upper")
    
    chr_name = ['I', 'II', 'III']
    # chr_name = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI']
    
    pos_chr_label = np.array(list(start_bead.values()))+ 0.5*np.array(list(nb_bead.values()))
    
    sec_x = ax.secondary_xaxis(location='top') #draw the separation between chr    
    sec_x.set_xticks(list(start_bead.values()), labels=[])
    sec_x.tick_params('x', length=20, width=1.5)

    sec_2_x = ax.secondary_xaxis(location='top') #write the chr name    
    sec_2_x.set_xticks(pos_chr_label, labels = chr_name)
    sec_2_x.tick_params('x', length=0)

    sec_3_x = ax.secondary_xaxis(location='bottom')
    #sec_3_x.set_xticks(list(np.array(list(start_bead.values()))+np.array(list(centro.values())) // resolution), labels = theta_centro_chr)
    sec_3_x.set_xticks(list(np.array(list(start_bead.values()))+np.array(centro) // resolution), labels = theta_centro_chr)
    sec_3_x.tick_params('x', length=10)
    sec_3_x.tick_params(labelsize=7, color="blue", labelcolor="blue")
    
    sec_y = ax.secondary_yaxis(location=0) #draw the separation between chr   
    sec_y.set_yticks(list(start_bead.values()), labels=[])
    sec_y.tick_params('y', length=20, width=1.5)
    
    sec_2_y = ax.secondary_yaxis(location=0) #write the chr name    
    sec_2_y.set_yticks(pos_chr_label, labels=chr_name) #labels=list(start_bead.keys()))
    sec_2_y.tick_params('y', length=0)

    sec_3_y = ax.secondary_yaxis(location='right')
    #sec_3_y.set_yticks(list(np.array(list(start_bead.values()))+np.array(list(centro.values())) // resolution), labels = theta_centro_chr)
    sec_3_y.set_yticks(list(np.array(list(start_bead.values()))+np.array(centro) // resolution), labels = theta_centro_chr)
    sec_3_y.tick_params('y', length=10)
    sec_3_y.tick_params(labelsize=7, color="blue", labelcolor="blue")

    #plot the grid to separate chr
    ax.vlines(np.array(list(start_bead.values())), ymin=0, ymax=len(C)-1, linestyle=(0,(1,1)), color="black", alpha=0.3, lw=0.7)
    ax.hlines(np.array(list(start_bead.values())), xmin=0, xmax=len(C)-1, linestyle=(0,(1,1)), color="black", alpha=0.3, lw=0.7)

    for (chr_row, chr_col) in product(chr_cen.keys(), chr_cen.keys()):
        centro_bead_row = chr_cen[chr_row] // resolution #get the bead id of the centromere
        centro_bead_col = chr_cen[chr_col] // resolution #get the bead id of the centromere
        plt.scatter(start_bead[chr_col] + centro_bead_col, start_bead[chr_row]+centro_bead_row, color="black", s = 0.5)
        i = int(chr_row[chr_row.find('chr')+3:])-1
        j = int(chr_col[chr_col.find('chr')+3:])-1
        theta_centro_bead_row = centro[i] // resolution #get the bead id of the centromere
        theta_centro_bead_col = centro[j] // resolution #get the bead id of the centromere
        
        # theta_centro_bead_row = centro[chr_row] // resolution #get the bead id of the centromere
        # theta_centro_bead_col = centro[chr_col] // resolution #get the bead id of the centromere
        plt.scatter(start_bead[chr_col] + theta_centro_bead_col, start_bead[chr_row]+theta_centro_bead_row, color="blue", s = 0.5)
 
   
    ax.set_xticks(list(np.array(list(start_bead.values()))+np.array(list(nb_bead.values()))-1)+list(np.array(list(start_bead.values()))+np.array(list(chr_cen.values())) // resolution), labels = end_chr + centro_chr)
    ax.set_yticks(list(np.array(list(start_bead.values()))+np.array(list(nb_bead.values()))-1)+list(np.array(list(start_bead.values()))+np.array(list(chr_cen.values())) // resolution), labels = end_chr + centro_chr)
    ax.tick_params(top = False, labeltop = False, bottom=True, labelbottom=True, left = False, labelleft = False, right = True, labelright=True, labelsize=7)

    markers = [plt.Line2D([0],[0],color="black", marker="o", linestyle=''), plt.Line2D([0],[0],color="blue", marker="o", linestyle='')]
    labels = [r"$\theta_0$", r"$\theta$"]
    ax.legend(markers, labels, bbox_to_anchor=(-0.1, 0.5))
    
    fig.colorbar(draw, location='right')
    title=rf"$\sigma^2 = {sig_2_ref:.3f}$, intensity={intensity_ref}, resolution = {resolution}"
    #title=rf"experimental map, resolution = {resolution}"
    plt.suptitle(title, fontweight="bold", fontsize="medium")

    ax.set_xlim(0, nb_tot_bead)
    ax.set_ylim(0, nb_tot_bead)
    ax.set_box_aspect(1)
    plt.axis([0, nb_tot_bead, nb_tot_bead, 0])
    plt.show() 
     
def plot_colormap_corr_theta(theta, corr):
    """ create 3 sub3D scatterplots with thetas (c_1,c_2,c_3) and the correlation in color """
    fig = plt.figure(figsize=(20, 12))
    for j in range(len(corr)):
        ax = fig.add_subplot(1,len(corr), j+1, projection='3d')

        c_x = []
        c_y = []
        c_z = []
        for i in range(len(theta)):
            
            c_1, c_2, c_3= list(theta[i].values())
            c_x.append(c_1)
            c_y.append(c_2)
            c_z.append(c_3)
        
        surf=ax.scatter(c_x, c_y, c_z, c=list(corr.values())[j], cmap="coolwarm", s=20)
        c_1_ref, c_2_ref, c_3_ref = list(chr_cen.values())

        ax.plot([c_1_ref, c_1_ref], [ax.get_ylim()[0], ax.get_ylim()[1]], [c_3_ref, c_3_ref], color="black", linestyle='--', alpha=0.5) 

        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [c_2_ref, c_2_ref], [c_3_ref, c_3_ref], color="black", linestyle='--', alpha=0.5) 

        ax.plot([c_1_ref, c_1_ref], [c_2_ref, c_2_ref], [ax.get_zlim()[0], ax.get_zlim()[1]], color="black", linestyle='--', alpha=0.5)
    

        ax.scatter(c_1_ref, c_2_ref, c_3_ref, color="black", label = r"$\theta_0$")
        ax.set_xlabel('chr01', fontsize=12, fontweight='bold')
        ax.set_ylabel('chr02',fontsize=12, fontweight='bold')
        ax.set_zlabel('chr03',fontsize=12, fontweight='bold')
        
        ax.set_xticks(list(ax.get_xticks())[:-2] + [chr_seq["chr01"]])
        ax.set_yticks(list(ax.get_yticks())[:-2] + [chr_seq["chr02"]])
        ax.set_zticks(list(ax.get_zticks())[:-3] + [chr_seq["chr03"]])
        ax.set_xlim(0, chr_seq["chr01"])
        ax.set_ylim(0, chr_seq["chr02"])
        ax.set_zlim(0, chr_seq["chr03"])
        ax.legend()
        ax.set_title(rf"{len(theta)} $\theta$")
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))  # Adjust exponent limits
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.zaxis.set_major_formatter(formatter)
        fig.colorbar(surf,  label=f"{list(corr.keys())[j]}-based inter P. corr.", shrink = 0.5, orientation = 'horizontal')
    plt.show()

# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

def plotly_colormap_corr_theta(theta, corr):
    subplot_titles = [f"{name}-based inter " for name in corr.keys()]
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=subplot_titles
    )
    
    #fig = go.Figure()
    colorbar_positions = [0.17, 0.5, 0.83]
    for j, (corr_name, corr_values) in enumerate(corr.items()):
        c_x, c_y, c_z = [], [], []
        
        for i in range(len(theta)):
            c_1, c_2, c_3 = list(theta[i].values())
            c_x.append(c_1)
            c_y.append(c_2)
            c_z.append(c_3)

        # Scatter plot (colored by correlation values)
        fig.add_trace(go.Scatter3d(
            x=c_x, y=c_y, z=c_z,
            mode='markers',
            marker=dict(
                size=5,
                color=corr_values,  # Color mapped to correlation values
                colorscale='RdBu_r',
                colorbar=dict(
                        len=0.3,  # Reduce colorbar size
                        #thickness=15,  # Adjust width
                        y=0,  # Move colorbar lower
                        x=colorbar_positions[j],  # Center the colorbar
                        orientation="h"  # Horizontal colorbar
                    )
            ),
            showlegend=False
            #name=f"{corr_name[j]}"
        ), row = 1, col = j+1)

        # Reference point (chr_cen) and dashed reference lines
        c_1_ref, c_2_ref, c_3_ref = list(chr_cen.values())

        # Add dashed lines for reference
        fig.add_trace(go.Scatter3d(
            x=[c_1_ref, c_1_ref], y=[0, chr_seq["chr02"]], z=[c_3_ref, c_3_ref],
            mode='lines', line=dict(color='black', dash='dash'),
            showlegend=False
        ), row = 1, col = j+1)
        fig.add_trace(go.Scatter3d(
            x=[0, chr_seq["chr01"]], y=[c_2_ref, c_2_ref], z=[c_3_ref, c_3_ref],
            mode='lines', line=dict(color='black', dash='dash'),
            showlegend=False
        ), row = 1, col = j+1)
        fig.add_trace(go.Scatter3d(
            x=[c_1_ref, c_1_ref], y=[c_2_ref, c_2_ref], z=[0, chr_seq["chr03"]],
            mode='lines', line=dict(color='black', dash='dash'),
            showlegend=False
        ), row = 1, col = j+1)

        # Highlight the reference point
        fig.add_trace(go.Scatter3d(
            x=[c_1_ref], y=[c_2_ref], z=[c_3_ref],
            mode='markers',
            marker=dict(size=8, color='black', symbol='diamond'),
            name=r"$\theta_0$", 
            showlegend=bool(j==0)
        ), row = 1, col = j+1)

        # Formatting axes and labels
        fig.update_layout(
            title=r"$\mathbf{" + str(len(theta)) + r"\ \theta\ \text{values}}$",
            title_y = 0.99,
            scene1=dict(
                xaxis=dict(title="chr01", tickmode="array",
                        tickvals=list(range(0, chr_seq["chr01"]+1, int(chr_seq["chr01"] / 5))),
                        tickformat=".1e"),  # Scientific notation
                yaxis=dict(title="chr02", tickmode="array",
                        tickvals=list(range(0, chr_seq["chr02"]+1, int(chr_seq["chr02"] / 5))),
                        tickformat=".1e"),
                zaxis=dict(title="chr03", tickmode="array",
                        tickvals=list(range(0, chr_seq["chr03"]+1, int(chr_seq["chr03"] / 5))),
                        tickformat=".1e"),
            ),
            scene2=dict(
                xaxis=dict(title="chr01", tickmode="array",
                        tickvals=list(range(0, chr_seq["chr01"]+1, int(chr_seq["chr01"] / 5))),
                        tickformat=".1e"),  # Scientific notation
                yaxis=dict(title="chr02", tickmode="array",
                        tickvals=list(range(0, chr_seq["chr02"]+1, int(chr_seq["chr02"] / 5))),
                        tickformat=".1e"),
                zaxis=dict(title="chr03", tickmode="array",
                        tickvals=list(range(0, chr_seq["chr03"]+1, int(chr_seq["chr03"] / 5))),
                        tickformat=".1e"),
            ),
            scene3=dict(
                xaxis=dict(title="chr01", tickmode="array",
                        tickvals=list(range(0, chr_seq["chr01"]+1, int(chr_seq["chr01"] / 5))),
                        tickformat=".1e"),  # Scientific notation
                yaxis=dict(title="chr02", tickmode="array",
                        tickvals=list(range(0, chr_seq["chr02"]+1, int(chr_seq["chr02"] / 5))),
                        tickformat=".1e"),
                zaxis=dict(title="chr03", tickmode="array",
                        tickvals=list(range(0, chr_seq["chr03"]+1, int(chr_seq["chr03"] / 5))),
                        tickformat=".1e"),
            ),
            
            margin=dict(l=0, r=0, b=0, t=40),  # Adjust layout margins
            annotations=[  # Adjust subplot title positions
                dict(
                    text=subplot_titles[i], y=0.95,  # Adjust x/y positions
                    xanchor="center", yanchor="top",
                    font=dict(size=14, family="Arial", color="black"),
                    showarrow=False
                ) for i in range(len(subplot_titles))
            ]
            
        )
    
    # Save as an interactive HTML file
    fig.write_html("interactive_3d_plot.html",include_mathjax="cdn" )

    # Show the interactive plot
    fig.show()

def plot_perf_metric(corr):
    
    fig, ax1 = plt.subplots(figsize=(8, 6))

    facecolor = ["salmon", "cornflowerblue", "yellowgreen"]
    color = ["red", "royalblue", "olivedrab"]
    x_start = [min(corr_values) for corr_values in corr.values()]
    x_stop = [max(corr_values) for corr_values in corr.values()]
    
    for i, (corr_keys, corr_values) in enumerate(corr.items()):
        corr_values = np.array(sorted(corr_values))
        corr_accepted = []
        
        for corr_ref in np.linspace(min(x_start), max(x_stop), 100):
            
            corr_accepted.append(len(corr_values[corr_values>=corr_ref])/len(corr_values))
        
        ax1.plot(np.linspace(min(x_start), max(x_stop), 100), corr_accepted, linestyle = '-', linewidth=2, markersize=7, color=color[i], markerfacecolor=facecolor[i], alpha=0.7, markeredgecolor=color[i], marker='o', label=f"{corr_keys}-based S. corr")
        ax1.set_xlabel(r"Correlation threshold $\epsilon$")
        ax1.set_ylabel(r"Proportion of $\theta$ with $corr>\epsilon$")
        #ax1.set_xticks(np.linspace(min(x_start), max(x_stop), int((max(x_stop) - min(x_start)) / 0.05) ))  
        ax1.set_xticks([max(x_stop)] + list(np.arange(0, 0.225, 0.025)))  
        #ax1.set_xticks(np.linspace(0.6, max(x_stop), int((max(x_stop) - 0.6) / 0.05) ))  
        #ax1.set_xlim(min(x_start), max(x_stop))
        ax1.set_xlim(0.125, max(x_stop))
        ax1.set_ylim(0, 0.6)
        
        ax1.set_title("Clear simulation, performance of the metrics")
        ax1.legend()
    plt.show()

##############################################################################
##############################################################################     
# with open("simulation_little_genome/noisy/theta", 'rb') as f:
#         theta = pickle.load(f)
# with open('simulation_little_genome/noisy/P_corr_inter_vector', 'rb') as f:
#         corr_vector = pickle.load( f)
# with open('simulation_little_genome/noisy/P_corr_inter_row', 'rb') as f:
#         corr_row = pickle.load( f)
# with open('simulation_little_genome/noisy/P_corr_inter_col', 'rb') as f:
#         corr_col = pickle.load( f)
    
# corr = {}
# corr["P_row"] = corr_row
# corr["P_col"] = corr_col
# corr["P_vector"] = corr_vector

#plotly_colormap_corr_theta(theta, corr)

# plot_perf_metric(corr)

#simulator_genome()

#simulate C_ref
if 0:
    ref = {(27,27):(8,17), (27,105): (8,29), (105,27):(8,20)}
    n_row = 105
    n_col = 27
    #n_col = 105
    resolution = 32000
    c_i_ref = ref[(n_row, n_col)][0]*resolution
    #c_j_ref = 29*resolution
    c_j_ref = ref[(n_row, n_col)][1]*resolution
    #c_j_ref = 17*resolution
    
    sig_2_ref = 2
    intensity_ref = 100000
    C_ref = simulator(n_row,n_col, c_i_ref//resolution, c_j_ref//resolution, sig_2_ref, intensity_ref, noisy=1)
    #plot_C(C_ref, c_i_ref//resolution, c_j_ref//resolution, sig_2_ref, intensity_ref, c_i_ref//resolution, c_j_ref//resolution, corr=1)

#simulate 1 bloc with c_i/c_j only
if 0:
    theta_corr_row = {}
    theta_corr_col = {}
    theta_corr_vector = {}
    theta_param = {}
    for k in range(1000):
        print(k)
        c_i = pdist.Uniform(low=1, high=(n_row-1)*resolution).sample()
        c_j = pdist.Uniform(low=1, high=(n_col-1)*resolution).sample()
        # sig_2 = random.uniform(0.1, 10)
        # intensity = random.choice(range(1, 1001))
        sig_2 = sig_2_ref
        intensity = intensity_ref
        #c_j=random.choice(range(1,n_col-1))
        C_simu =simulator(n_row,n_col, c_i//resolution, c_j//resolution, sig_2, intensity)
        corr_row = Pearson_correlation_row(C_ref, C_simu)
        corr_vector = Pearson_correlation_vector(C_ref, C_simu)
        corr_col = Pearson_correlation_col(C_ref, C_simu)
        theta_corr_row[(int(c_i.detach().item()), int(c_j.detach().item()))] = corr_row
        theta_corr_col[(int(c_i.detach().item()), int(c_j.detach().item()))] = corr_col
        theta_corr_vector[(int(c_i.detach().item()), int(c_j.detach().item()))] = corr_vector
        #theta_param[int(c_i.detach().item())] = (sig_2, intensity)
    with open('theta_corr_row', 'wb') as f:
            pickle.dump(theta_corr_row, f)
    with open('theta_corr_col', 'wb') as f:
            pickle.dump(theta_corr_col, f)
    with open('theta_corr_vector', 'wb') as f:
            pickle.dump(theta_corr_vector, f)
    # with open('theta_param', 'wb') as f:
    #         pickle.dump(theta_param, f)

#simulate 1 bloc with sigma^2 only
if 0:
    sig_2_corr_row_col = {}
    sig_2_corr_vector = {}
    for i in range(500):
        print(i)
        sig_2 = random.uniform(0.1, 10)
        C_simu =simulator(n_row,n_col, c_i_ref//resolution, c_j_ref//resolution, sig_2, intensity_ref, noisy=1)
        corr_row = Pearson_correlation_row(C_ref, C_simu)
        corr_col = Pearson_correlation_col(C_ref, C_simu)
        corr_vector = Pearson_correlation_vector(C_ref, C_simu)
            
        sig_2_corr_row_col[sig_2] = 0.5*(corr_row+corr_col)
        sig_2_corr_vector[sig_2] = corr_vector
    with open('matrix_105_27/1000_theta/inference_c_i_c_j/sig_2_corr_row_col', 'wb') as f:
            pickle.dump(sig_2_corr_row_col, f)
    with open('matrix_105_27/1000_theta/inference_c_i_c_j/sig_2_corr_vector', 'wb') as f:
            pickle.dump(sig_2_corr_vector, f)

     
#simulate 1 bloc reprend theta only
if 0:
    # with open("matrix_27_105/inference_col_100_c_j/theta_param", 'rb') as f:
    #         theta_param = pickle.load(f) 
    with open("matrix_105_27/1000_theta/inference_c_i_c_j/theta_P_corr_row", 'rb') as f:
            theta_corr_row = pickle.load(f) 
    with open("matrix_105_27/1000_theta/inference_c_i_c_j/theta_P_corr_col", 'rb') as f:
            theta_corr_col = pickle.load(f) 
    with open("matrix_105_27/1000_theta/inference_c_i_c_j/theta_P_corr_vector", 'rb') as f:
            theta_corr_vector = pickle.load(f) 
    if theta_corr_row.keys()==theta_corr_col.keys()==theta_corr_vector.keys():
        print("ok")

    theta_corr_row_col_new = {}
    theta_corr_vector_new = {}
        
    for (c_i, c_j), corr in theta_corr_row.items():
        print(c_i, c_j)
        #C_simu =simulator(n_row,n_col, c_i//resolution, c_j_ref//resolution, sig_2, intensity)
        C_simu =simulator(n_row,n_col, c_i//resolution, c_j//resolution, sig_2_ref, intensity_ref, noisy=1)
        corr_row = Pearson_correlation_row(C_ref, C_simu)
        corr_col = Pearson_correlation_col(C_ref, C_simu)
        corr_vector = Pearson_correlation_vector(C_ref, C_simu)
        
        theta_corr_row_col_new[(c_i, c_j)] = 0.5*(corr_row+corr_col)
        theta_corr_vector_new[(c_i,c_j)] = corr_vector
        
    # with open('noisy_theta_P_corr_col', 'wb') as f:
    #         pickle.dump(theta_corr_col_new, f)
    with open('noisy_theta_P_corr_row_col', 'wb') as f:
            pickle.dump(theta_corr_row_col_new, f)
    with open('noisy_theta_P_corr_vector', 'wb') as f:
            pickle.dump(theta_corr_vector_new, f)


# with open("matrix_27_105/inference_col_c_j/1000_theta/theta_corr_vector", 'rb') as f:
#             theta_corr = pickle.load(f) 
#plot correlation VS c_i
if 0:
    fig,ax = plt.subplots()
    theta_corr = dict(sorted(theta_corr.items(), key=lambda item: item[0]))

    plt.plot(theta_corr.keys(), theta_corr.values(), marker='o')
    plt.title(f"Pearson correlation \n {len(theta_corr.keys())}"+r"$\theta's$")
    plt.xlabel(r"$\theta = c_j$")
    plt.ylabel(f"upper-matrix-based Pearson correlation")
    ax.set_xlim(1, (n_col-1)*resolution)
    sec_x = ax.secondary_xaxis(location='bottom') #draw the separation between chr    
    right = (n_col-1)*resolution
    sec_x.set_xticks([1, right], labels=["1", f"{right}"])
    sec_x.tick_params('x', length=10, width=1.5)
    ax.axvline(x=c_j_ref, linestyle='--', color="crimson", label=rf"$\theta_0$")
    plt.legend()
    plt.show()


#histo c_i
if 0:
    with open("matrix_27_105/inference_row_c_i/1000_theta/theta_corr", 'rb') as f:
            theta_corr = pickle.load(f) 
    with open("matrix_27_105/inference_row_c_i/1000_theta/theta_param", 'rb') as f:
            theta_param = pickle.load(f)

    for prop in np.linspace(0.05,0.5, 6):
        start = int(len(theta_corr.keys())*(1-prop))-1
        
        theta_corr_sorted= dict(sorted(theta_corr.items(), key=lambda item: item[1])) #sort by values
        thetas_corr_accepted = dict(list(theta_corr_sorted.items())[start:]) #take theta:corr_inter accepted 
        thresh = list(thetas_corr_accepted.values())[0] #take the corr corresponding to the prop
        thetas_accepted = list(thetas_corr_accepted.keys()) #take the thetas accepted

        df = pd.DataFrame(thetas_corr_accepted.items(), columns=["theta", "corr_inter"])
        fig, ax = plt.subplots(figsize=(8,6))

        sns.histplot(df, x="theta", kde=True, alpha=0.3, edgecolor='cornflowerblue')
        
        ax.set_xlabel(r"$\theta$")
        ax.axvline(x=c_i_ref, linestyle='--', color="crimson", label=rf"$\theta_0$")
        ax.scatter(list(theta_corr.keys()), [0]*len(list(theta_corr.keys())), marker="+", color='red', label=r"$\theta$ not accepted")
        ax.scatter(thetas_accepted, [0]*len(thetas_accepted), marker='+', color="navy", label = r"$\theta$ accepted")
        ax.legend([f"({len(thetas_accepted)} / {len(list(theta_corr.keys()))}) \n {int(len(thetas_accepted)/len(list(theta_corr.keys()))*100)}% accepted", rf"$\theta_0$", r"$\theta$ not accepted", r"$\theta$ accepted"])#, bbox_to_anchor=(0.85,1.01))
        ax.set_title(rf"$p(\theta_{{centro}}|C_{{ref}})$ for $\epsilon = {thresh:.3f}$")
        sec_x = ax.secondary_xaxis(location='bottom') #draw the separation between chr    
        sec_x.set_xticks([1, (n_row-1)*resolution], labels=[])
        sec_x.tick_params('x', length=10, width=1.5)
        extraticks = [1, (n_row-1)*resolution]
        ax.set_xticks(list(ax.get_xticks()) + extraticks)
        ax.set_xlim(1, (n_row-1)*resolution)
        #plt.show()
        plt.savefig(f"{prop}.svg")
        if prop==0.05:
            
            for theta in thetas_accepted[:5]:
                
                # C_simu = simulator(n_row, n_col, c_i_ref//resolution, theta//resolution, theta_param[theta][0], theta_param[theta][1])
                # plot_C(C_simu, c_i_ref//resolution, theta//resolution, theta_param[theta][0], theta_param[theta][1], c_i_ref//resolution, c_j_ref//resolution, thetas_corr_accepted[theta])

                C_simu = simulator(n_row, n_col, theta//resolution, c_j_ref//resolution, theta_param[theta][0], theta_param[theta][1])
                print(theta//resolution, theta_param[theta][0], theta_param[theta][1], thetas_corr_accepted[theta])
                plot_C(C_simu, theta//resolution, c_j_ref//resolution, theta_param[theta][0], theta_param[theta][1], c_i_ref//resolution, c_j_ref//resolution, thetas_corr_accepted[theta])
    
#plot_C(C_simu, c_i, c_j, sig_2, intensity, c_i_ref, c_j_ref)

#colormap correlation VS c_i, sigma_2 
if 0:
     
    with open("matrix_27_105/1000_theta/inference_c_i_c_j/theta_P_corr_row_mean", 'rb') as f:
            theta_corr = pickle.load(f) 
   
    # with open("matrix_105_27/1000_theta/inference_row_c_i/theta_param", 'rb') as f:
    #         theta_param = pickle.load(f) 

    #theta_corr = dict(sorted(theta_corr.items(), key=lambda item: item[0]))
    #theta_param = dict(sorted(theta_param.items(), key=lambda item: item[0]))


    #sig_2 = [i[0] for i in list(theta_param.values())]
    
    #c_i= list(theta_corr.keys())
    c_i= [i[0] for i in list(theta_corr.keys())]
    c_j= [i[1] for i in list(theta_corr.keys())]

    corr = list(theta_corr.values())
    print(len(corr))
    
    # X, Y = np.meshgrid(c_i, sig_2)
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.scatter(c_i, sig_2, corr, c=corr, cmap="coolwarm")

    # ax.set_zlim(0,1)
    # ax.scatter(c_i_ref, sig_2_ref,1, color = "black", label=r"$c_{i, ref}, \sigma^2_{ref}$")
    # ax.set_xlabel(r"$c_i$")
    # ax.set_ylabel(r"$\sigma^2$")
    # ax.set_zlabel("row-based P. corr.")
    # plt.legend()
    # plt.show()

    fig,ax = plt.subplots()

    # surf = ax.scatter(c_i, sig_2, c=corr, cmap="coolwarm", s=20)
    surf = ax.scatter(c_i, c_j, c=corr, cmap="coolwarm", s=20)
    fig.colorbar(surf, location='right', label="row-based P. corr.")
    # fig.colorbar(surf, location='right', label="upper-matrix-based P. corr.")

    #ax.set_zlim(0,1)
    ax.axvline(c_i_ref, linestyle='--', color="grey", alpha = 0.5)
    # ax.axhline(sig_2_ref, linestyle='--', color="grey", alpha=0.5)
    ax.axhline(c_j_ref, linestyle='--', color="grey", alpha=0.5)
   
    # ax.scatter(c_i_ref, sig_2_ref, color = "black", s=20, label=r"$c_{i, ref}, \sigma^2_{ref}$")
    ax.scatter(c_i_ref, c_j_ref, color = "black", s=20, label=r"$c_{i, ref}, c_{j,ref}$")

    ax.set_xlabel(r"$c_i$")
    # ax.set_ylabel(r"$\sigma^2$")
    ax.set_ylabel(r"$c_j$")
    
    ax.set_xticks(list(ax.get_xticks())[2:] + [1, (n_row-1)*resolution])
    # ax.set_yticks(list(ax.get_yticks()) + [0.1])
    ax.set_yticks(list(ax.get_yticks()) + [1, (n_col-1)*resolution])

    ax.set_xlim(1, (n_row-1)*resolution)
    # ax.set_ylim(0.1, 10)
    ax.set_ylim(1, (n_col-1)*resolution)

    # ax.set_title(r"row-based Pearson correlation VS $(c_i, \sigma^2)$")
    ax.set_title(r"row-based Pearson correlation VS $(c_i, c_j)$")
    # ax.set_title(r"upper-matrix-based Pearson correlation VS $(c_i, c_j)$")
    #ax.set_zlabel("row-based P. corr.")
    plt.legend()
    plt.show()

