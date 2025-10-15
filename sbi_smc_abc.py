import torch
from sbi import utils as utils
from torch.distributions import MultivariateNormal
from itertools import combinations
from simulator import plot_C_genome
from sbi.inference import SMCABC
from sbi.neural_nets.embedding_nets import CNNEmbedding
import random
import numpy as np
import pywt
import pickle

import seaborn as sns
import matplotlib.pyplot as plt
torch.manual_seed(0)

chr_seq_3_chr = {"chr01": 230209, "chr02": 813179, "chr03": 316619}
chr_cen_3_chr = {'chr01': 151584, 'chr02': 238325, 'chr03': 114499}

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

def simulator(theta, resolution, sigma_spot, noisy):
        
    if sigma_spot=="variable":
        sig_2_simu = random.uniform(0.1, 10)
    else:
        sig_2_simu = sig_2_ref
               
    intensity_simu = 100

    C_simu = torch.zeros((nb_tot_bead,nb_tot_bead))

    for (chr_row, chr_col) in combinations(chr_seq_3_chr.keys(),r=2):
        
        n_row = chr_seq_3_chr[chr_row]//resolution
        n_col = chr_seq_3_chr[chr_col]//resolution
        index_row = int(chr_row[chr_row.find("chr")+3:])-1
        index_col = int(chr_col[chr_col.find("chr")+3:])-1
        c_i_simu = theta[index_row]//resolution
        c_j_simu = theta[index_col]//resolution

        def simulator_1_bloc(n_row, n_col, c_i, c_j, sig_2, intensity, noisy=noisy):
            
            # Simulate a noisy matrix C_{n_row x n_col} with a gaussian spot at (c_i, c_j) of size sig_2 
            
            C = torch.zeros((n_row, n_col))
            
            distr = MultivariateNormal(torch.tensor([c_i, c_j]), sig_2*torch.eye(2))
                
            indices = torch.tensor([[(i, j) for j in range(len(C[0]))] for i in range(len(C))])
            C = intensity*torch.exp(distr.log_prob(torch.tensor(indices)))
            
            if noisy:
                mean = intensity * torch.exp(distr.log_prob(torch.tensor([c_i,c_j]))) * 0.1 / 2 
                sigma = intensity * torch.exp(distr.log_prob(torch.tensor([c_i,c_j]))) * 0.1 / 2
       
                noise = mean + sigma*torch.randn((n_row, n_col))

            else:
                noise = torch.zeros_like(C)
            
            return C+noise
        
        C_simu[start_bead[chr_row]:start_bead[chr_row]+nb_bead[chr_row]-1, start_bead[chr_col]:start_bead[chr_col]+nb_bead[chr_col]-1] = simulator_1_bloc(n_row,n_col, c_i_simu, c_j_simu, sig_2_simu, intensity_simu, noisy=noisy)
    
    return C_simu


resolution= 32000
origin = "true"
dim = 3
sig_2_ref = 1.0
sigma_spot = "variable"
noisy = 1
wavelets=''

nb_bead, start_bead, nb_tot_bead = get_num_beads_and_start(chr_seq_3_chr, resolution)

nb_train = 1000
nb_posterior_samples = 50

prior_range = torch.tensor([230209, 813179, 316619])
prior = utils.BoxUniform(torch.ones(dim), prior_range-1)
prior_post = utils.BoxUniform(torch.ones(dim)/prior_range, (prior_range-1)/prior_range)

if origin == "true":
    C_ref =  torch.from_numpy(np.load(f"ref/3_chr_ref_{resolution}_norm_HiC_duan_intra_all.npy")).float()
    if wavelets=="wavelets":
        C_ref, (LH, HL, HH) = pywt.dwt2(C_ref, 'bior1.3')
        C_ref = torch.from_numpy(C_ref) 
else:
    C_ref = simulator(list(chr_cen_3_chr.values()), resolution, sigma_spot, noisy)
    if wavelets=="wavelets":
        C_ref, (LH, HL, HH) = pywt.dwt2(C_ref, 'bior1.3')
        C_ref = torch.from_numpy(C_ref)

def get_simulations_sbi(proposal, round, nb_train, DNN):

    if round == 0:
        theta = prior.sample((nb_train,))
    else:
        theta=proposal.sample((nb_train,))
        #theta = theta*prior_range
        # plt.scatter(theta[:,0], torch.zeros_like(theta[:,0]))
        # plt.show()
    # C = torch.zeros(nb_train,C_ref.size(0))
    # C = torch.zeros(nb_train,C_ref.size(0), C_ref.size(1))
    theta_reg = torch.zeros((nb_train, dim))
    for k in range(nb_train):
     
        C_tmp = simulator(theta[k], resolution, sigma_spot, noisy)
        # C[k] = vectorise_upper_matrix(C_tmp, resolution)
        if wavelets=="wavelets":
            C_tmp, (LH, HL, HH) = pywt.dwt2(C_tmp, 'bior1.3')
            C_tmp = torch.from_numpy(C_tmp)
        
        #print(DNN(C[k]).size())
        #C[k] = C_tmp
        theta_reg[k] = DNN(C_tmp.unsqueeze(0))
    
    #theta = theta/prior_range #norm thetas
    print("training set done")
    return theta, theta_reg

theta_ref = torch.tensor(list(chr_cen_3_chr.values()))

# input_dim = (C_ref.size(0), C_ref.size(1))
# output_dim = theta_ref.size(0)
# decay_linear= 4
# nb_hidden_layers= 4
# kernel_size=3
# pool_kernel_size = 2
# first_hidden_dim = 512
# DNN = CNNEmbedding(input_shape=input_dim, in_channels=1, output_dim=output_dim, num_linear_layers=nb_hidden_layers, num_linear_units=first_hidden_dim, kernel_size=kernel_size, pool_kernel_size=pool_kernel_size)

path_dnn = f"simulation_little_genome/{origin}/res_{resolution}/noisy/sigma_{sigma_spot}/summary_stat/CNN/sigmoid/sequential/"
path_sbi = f"simulation_little_genome/{origin}/res_{resolution}/noisy/sigma_{sigma_spot}/sbi/"
nb_seq = 10
DNN = torch.load(path_dnn+"dnn.pkl")
inference_method = SMCABC
density_estimator = "maf"
proposal = prior

if 0:
    for k in range(nb_seq+1):

        theta, theta_reg = get_simulations_sbi(proposal, k, nb_train, DNN)

        theta_reg = theta_reg.detach()
        print(theta)
        print(theta_reg)
        
        #inference = inference_method(prior_post, density_estimator=density_estimator)
        inference = inference_method(simulator,prior,
        distance: Union[str, Callable] = "l2",
        requires_iid_data: Optional[None] = None,
        distance_kwargs: Optional[Dict] = None,
        num_workers: int = 1,
        simulation_batch_size: int = 1,
        distance_batch_size: int = -1,
        show_progress_bars: bool = True,
        kernel: Optional[str] = "gaussian",
        algorithm_variant: str = "C",)
        posterior_estimator = inference.append_simulations(theta=theta, x=theta_reg, proposal=proposal).train(
                training_batch_size=100
            )
        
            
        #posterior.set_default_x(C_ref.reshape(1,C_ref.size(0)*C_ref.size(1)))
        proposal = posterior.set_default_x(DNN(C_ref.unsqueeze(0)))

        samples = posterior.sample((nb_posterior_samples,))

        with open(f'{path_sbi}{k}_thetas_accepted', 'wb') as f:
                    #pickle.dump(samples*prior_range, f)
                    pickle.dump(samples, f)

if 1:
    fig, axes = plt.subplots(2,2, figsize=(12, 10))
    color = plt.cm.RdYlBu_r(np.linspace(0, 1, nb_seq+1))

    for i in range(nb_seq+1):
    
        with open(f'{path_sbi}{i}_thetas_accepted', 'rb') as f:
                    samples = pickle.load(f)
                    
        for k, chr in enumerate (list(chr_seq_3_chr.keys())):
            #axes[k//2, k%2].scatter(samples[:,k], torch.zeros_like(samples[:,k]))
            sns.kdeplot(ax = axes[k//2, k%2], data=samples[:, k], color = color[i], label = f"round {i}")
            
            axes[k//2, k%2].set_xlabel(r"$\theta_{centro}$")
            axes[k//2, k%2].axvline(x=chr_cen_3_chr[chr], linestyle='--', color="goldenrod", label=rf"$\theta_0$")
            
            axes[k//2, k%2].set_title(rf"$\theta_{{centro}}$ for chr ${k+1}$")
            sec_x = axes[k//2, k%2].secondary_xaxis(location='bottom') #draw the separation between chr    
            sec_x.set_xticks([1, chr_seq_3_chr[chr]], labels=[])
            sec_x.tick_params('x', length=10, width=1.5)
            extraticks = [1, chr_seq_3_chr[chr]]
            
            axes[k//2, k%2].set_xticks(list(axes[k//2, k%2].get_xticks()) + extraticks)
            axes[k//2, k%2].set_xlim(1, chr_seq_3_chr[chr])
            axes[k//2, k%2].legend()

    plt.suptitle(rf"$p(\theta_{{centro}}|C_{{ref}})$"+f"\n {origin} data - noisy : {noisy}"+fr" - $\sigma^2$ {sigma_spot} - res {resolution}" + f"\n {inference_method.__name__} - {density_estimator} - wavelets")
    plt.show()
                

    
     
    