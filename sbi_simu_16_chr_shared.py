import torch
from sbi import utils as utils
from torch.distributions import MultivariateNormal
from itertools import combinations
from simulator import plot_C_genome
from performance import mean_distance_best_thetas
from sbi.inference import NPE, NLE, DirectPosterior
from sbi.neural_nets.embedding_nets import CNNEmbedding
from sbi.neural_nets import posterior_nn
import random
import numpy as np
import pywt
import pickle
import torch.nn as nn


import seaborn as sns
import matplotlib.pyplot as plt
torch.manual_seed(0)

chr_seq_3_chr = {"chr01": 230209, "chr02": 813179, "chr03": 316619}
chr_cen_3_chr = {'chr01': 151584, 'chr02': 238325, 'chr03': 114499}

chr_seq_16_chr = {"chr1": 230209, "chr2": 813179, "chr3": 316619, "chr4": 1531918, 
           "chr5": 576869, "chr6": 270148, "chr7": 1090947, "chr8": 562644,
            "chr9": 439885, "chr10": 745746, "chr11": 666455, "chr12": 1078176,
            "chr13": 924430, "chr14": 784334, "chr15": 1091290, "chr16": 948063}

chr_cen_16_chr = {'chr1': 151584, 'chr2': 238325, 'chr3': 114499, 'chr4': 449819, 
                 'chr5': 152103, 'chr6': 148622, 'chr7': 497042, 'chr8': 105698, 
                 'chr9': 355742, 'chr10': 436418, 'chr11': 439889, 'chr12': 150946, 
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

def simulator_row(chr_id, theta, resolution, sigma_spot, noisy):
    """
    Simulate 1 row of blocs of the contact matrix
    """

    chr_num = "chr" + str(chr_id)

    if sigma_spot=="variable":
        sig_2_simu = random.uniform(0.1, 10)
    else:
        sig_2_simu = sig_2_ref
               
    intensity_simu = 100

    C_simu = np.zeros((nb_bead[chr_num],nb_tot_bead))

    n_row = nb_bead[chr_num]
    index_row = chr_id-1
    
    for chr_col in chr_seq.keys():
        if chr_col != chr_num:
        
            n_col = nb_bead[chr_col]
            index_col = int(chr_col[chr_col.find("chr")+3:])-1

            c_i_simu = theta[index_row]//resolution
            c_j_simu = theta[index_col]//resolution
            
            def simulator_1_bloc(n_row, n_col, c_i, c_j, sig_2, intensity, noisy=noisy):
                
                # Simulate a noisy matrix C_{n_row x n_col} with a gaussian spot at (c_i, c_j) of size sig_2 
                
                C = torch.zeros((n_row, n_col))
       
                distr = MultivariateNormal(torch.tensor([c_i, c_j]), sig_2*torch.eye(2))
                    
                indices = torch.tensor([[(i, j) for j in range(len(C[0]))] for i in range(len(C))])
                
                C = intensity*torch.exp(distr.log_prob(indices))
                
                if noisy:
                    mean = intensity * torch.exp(distr.log_prob(torch.tensor([c_i,c_j]))) * 0.1 / 2 
                    sigma = intensity * torch.exp(distr.log_prob(torch.tensor([c_i,c_j]))) * 0.1 / 2
        
                    noise = mean + sigma*torch.randn((n_row, n_col))

                else:
                    noise = torch.zeros_like(C)
                
                return C+noise
          
            C_simu[:start_bead[chr_num]+nb_bead[chr_num], start_bead[chr_col]:start_bead[chr_col]+nb_bead[chr_col]] = simulator_1_bloc(n_row,n_col, c_i_simu, c_j_simu, sig_2_simu, intensity_simu, noisy=noisy)


    return C_simu


resolution= 32000
origin = "true"
sig_2_ref = 1.0
sigma_spot = "variable"
noisy = 1
wavelets=''

dim=16

if dim==3:
    chr_seq = chr_seq_3_chr
    chr_cen = chr_cen_3_chr
    prior_range = torch.tensor([230209, 813179, 316619])
     
if dim==16:
    chr_seq = chr_seq_16_chr
    chr_cen = chr_cen_16_chr
    prior_range = torch.tensor([230209, 813179, 316619, 1531918, 
                                576869, 270148, 1090947, 562644,
                                439885, 745746, 666455, 1078176,
                                924430, 784334, 1091290, 948063])

nb_bead, start_bead, nb_tot_bead = get_num_beads_and_start(chr_seq, resolution)

nb_train = 1000
nb_posterior_samples = 50

path_sbi = f"simulation_little_genome/{dim}_chr/{origin}/res_{resolution}/noisy/sigma_{sigma_spot}/row/sbi/shared/"

prior_simu = utils.BoxUniform(torch.ones(dim), prior_range-1)
# prior_post = utils.BoxUniform(torch.ones(dim)/prior_range, (prior_range-1)/prior_range)

theta_ref = torch.tensor(list(chr_cen.values()))



def get_simulations_sbi_pretrained(chr_id, proposal, round, nb_train):

    if round == 0:
        theta = prior.sample((nb_train,))
    else:
        theta=proposal.sample((nb_train,))
        #theta = theta*prior_range
        # plt.scatter(theta[:,0], torch.zeros_like(theta[:,0]))
        # plt.show()
    # C = torch.zeros(nb_train,C_ref.size(0))
    # C = torch.zeros(nb_train,C_ref.size(0), C_ref.size(1))
    theta_reg = torch.zeros((nb_train, 1))
    for k in range(nb_train):
        theta_simu = prior_simu.sample()
        theta_simu[chr_id-1] = theta[k]
        C_tmp_row = simulator_row(chr_id, theta_simu, resolution, sigma_spot, noisy)
        
        
        # C[k] = vectorise_upper_matrix(C_tmp, resolution)
        if wavelets=="wavelets":
            C_tmp_row, (LH, HL, HH) = pywt.dwt2(C_tmp_row, 'bior1.3')
        C_tmp_row = torch.from_numpy(C_tmp_row).float()
        
        
        #print(DNN(C[k]).size())
        #C[k] = C_tmp
        C_tmp_row = C_tmp_row.unsqueeze(0)
        features = cnn_subnet(C_tmp_row.unsqueeze(0))          # → (B, F)
        
        features = features.view(C_tmp_row.size(0), -1)  # flatten
                    
        
        theta_reg[k] = linear_subnets[chr_id-1](features).squeeze()  # → (B,)
        
    
    #theta = theta/prior_range #norm thetas
    print("training set done")
    
    return theta, theta_reg

def get_simulations_sbi(proposal, round, nb_train):

    if round == 0:
        theta = prior.sample((nb_train,))
    else:
        theta=proposal.sample((nb_train,))
        #theta = theta*prior_range

    C = torch.zeros(nb_train,C_ref.size(0), C_ref.size(1))
    
    for k in range(nb_train):
     
        C_tmp = simulator_row(chr_id, theta[k], resolution, sigma_spot, noisy)

        # C[k] = vectorise_upper_matrix(C_tmp, resolution)
        if wavelets=="wavelets":
            C_tmp, (LH, HL, HH) = pywt.dwt2(C_tmp, 'bior1.3')
        C_tmp = torch.from_numpy(C_tmp).float()
        
        C[k] = C_tmp
    
    #theta = theta/prior_range #norm thetas
    print("training set done")
    return theta, C

def get_embedding_net(input_dim, 
                      output_dim,
                      nb_hidden_layers= 4,
                      kernel_size=3,
                      pool_kernel_size = 2,
                      first_hidden_dim = 512,
                      out_channels_per_layer = [6,12]):
    DNN = CNNEmbedding(input_shape=input_dim, in_channels=1, output_dim=output_dim, num_linear_layers=nb_hidden_layers, num_linear_units=first_hidden_dim, kernel_size=kernel_size, pool_kernel_size=pool_kernel_size, out_channels_per_layer=out_channels_per_layer)
    print(DNN)
    return DNN
     
                    
def get_cnn_subnet(in_channels, kernel_size, pool_kernel_size, out_channels_per_layer):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels_per_layer[0], kernel_size=kernel_size, padding=kernel_size//2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=pool_kernel_size),
        nn.Conv2d(in_channels=out_channels_per_layer[0], out_channels=out_channels_per_layer[1], kernel_size=kernel_size, padding=kernel_size//2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=pool_kernel_size),
        )
     
def get_linear_subnet(input_mlp, output_mlp, decay):

        power_fisrt_hidden = 0
        while decay**power_fisrt_hidden < input_mlp:
            power_fisrt_hidden += 1

        first_hidden_dim = decay**(power_fisrt_hidden-1)

        nb_hidden_layers= power_fisrt_hidden-2

        layers = [nn.Linear(input_mlp, first_hidden_dim), nn.ReLU()]
        
        for k in range(nb_hidden_layers - 2):
            layers.append(nn.Linear(first_hidden_dim//decay**k, first_hidden_dim//4**(k+1))) #//4**k //4**(k+1)
            layers.append(nn.ReLU())
        layers.append(nn.Linear(first_hidden_dim//4**(k+1), output_mlp)) #//4**(k+1)
        
        layers.append(nn.Sigmoid())
        linear_subnet = nn.Sequential(*layers)
        return linear_subnet
 

# input_dim = (C_ref.size(0), C_ref.size(1))
# output_dim = theta_ref.size(0)
# decay_linear= 4
# nb_hidden_layers= 4
# kernel_size=3
# pool_kernel_size = 2
# first_hidden_dim = 512
# DNN = CNNEmbedding(input_shape=input_dim, in_channels=1, output_dim=output_dim, num_linear_layers=nb_hidden_layers, num_linear_units=first_hidden_dim, kernel_size=kernel_size, pool_kernel_size=pool_kernel_size)

#embedding net pre tained
if 1:
    
    path_dnn = f"simulation_little_genome/{dim}_chr/{origin}/res_{resolution}/noisy/sigma_{sigma_spot}/row/summary_stat/1_theta/shared/"
    nb_seq = 10
    inference_method = NPE
    density_estimator = "maf"

    ########## load the dnn ########
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    kernel_size=3
    pool_kernel_size = 2
    out_channels_per_layer = [6,12]
    decay = 4
    cnn_subnet = get_cnn_subnet(in_channels=1, kernel_size=kernel_size, pool_kernel_size=pool_kernel_size, out_channels_per_layer=out_channels_per_layer)
    ######################

    ##### indep MLP ######

    linear_subnets = []
    for chr_id in range(1,17):
    
        C_ref_row =  np.load(f"ref/{dim}_chr_{chr_id}_ref_{resolution}_norm_HiC_duan_intra_all.npy")
    
        #C_ref_row=np.delete(C_ref_row, range(start_bead['chr'+str(chr_id)], start_bead['chr'+str(chr_id)] + nb_bead['chr'+str(chr_id)]), axis =1) #remove intra
        C_ref_row = torch.from_numpy(C_ref_row).float() 

    
        input_dim = (C_ref_row.size(0), C_ref_row.size(1))
        # print("chr", chr_id, input_dim)
        output_mlp = 1
        # output_mlp = 16
        input_mlp = (input_dim[0]//decay) * (input_dim[1]//decay) * out_channels_per_layer[1]
        linear_subnets.append(get_linear_subnet(input_mlp=input_mlp, output_mlp=output_mlp, decay=decay))
    checkpoint = torch.load(path_dnn + "dnn_500_epochs.pth", map_location=device)

    # Load CNN weights
    cnn_subnet.load_state_dict(checkpoint['cnn'])

    # Load MLP weights
    for i, mlp in enumerate(linear_subnets):
        mlp.load_state_dict(checkpoint['mlps'][i])
    ##############################

    for chr_id in range(1,17):
        prior = utils.BoxUniform(torch.ones(1), prior_range[chr_id-1]-1)
        proposal = prior    
        
    
        #inference = inference_method(prior_post, density_estimator=density_estimator)
        inference = inference_method(prior, density_estimator=density_estimator)


        for k in range(nb_seq+1):
            print("round", k)

            theta, theta_reg = get_simulations_sbi_pretrained(chr_id, proposal, k, nb_train)

            theta_reg = theta_reg.detach()
            # print(theta)
            # print(theta_reg)
            
            posterior_estimator = inference.append_simulations(theta=theta, x=theta_reg, proposal=proposal).train(
                    training_batch_size=100
                )
            if inference_method==NPE:
                #posterior = DirectPosterior(prior=prior_post, posterior_estimator=posterior_estimator)
                posterior = DirectPosterior(prior=prior, posterior_estimator=posterior_estimator)
            

            if inference_method==NLE:
                posterior = inference.build_posterior(mcmc_method="slice_np_vectorized",
                                                mcmc_parameters={"num_chains": 20,
                                                "thin": 5})
                
            if origin == "true":
                C_ref_row =  torch.from_numpy(np.load(f"ref/{dim}_chr_{chr_id}_ref_{resolution}_norm_HiC_duan_intra_all.npy")).float()
                if wavelets=="wavelets":
                    C_ref_row, (LH, HL, HH) = pywt.dwt2(C_ref_row, 'bior1.3')
                    C_ref_row = torch.from_numpy(C_ref_row) 

            else:
                C_ref_row = simulator_row(chr_id, list(chr_cen.values()), resolution, sigma_spot, noisy)
                if wavelets=="wavelets":
                    C_ref_row, (LH, HL, HH) = pywt.dwt2(C_ref_row, 'bior1.3')
                    C_ref_row = torch.from_numpy(C_ref_row)

            C_ref_row = C_ref_row.unsqueeze(0)
            features = cnn_subnet(C_ref_row.unsqueeze(0))          # → (B, F)
            features = features.view(C_ref_row.size(0), -1)
            S_phi_C_ref = linear_subnets[chr_id-1](features).squeeze()
            proposal = posterior.set_default_x(S_phi_C_ref)

            samples = posterior.sample((nb_posterior_samples,))

            with open(f'{path_sbi}chr_{chr_id}_{k}_thetas_accepted', 'wb') as f:
                        #pickle.dump(samples*prior_range, f)
                        pickle.dump(samples, f)

#embedding net trained jointly
if 0:
    dim = 3
    nb_train = 10000
  

    nb_seq = 0
    proposal = prior
    input_dim = (C_ref.size(0), C_ref.size(1))
    output_dim = theta_ref.size(0)
    out_channels_per_layer=[6,12]
    input_mlp = (input_dim[0]//4) * (input_dim[1]//4) * out_channels_per_layer[1]
    
    power = 0
    while 4**power < input_mlp:
        power += 1
    first_hidden_dim = 4**(power-1)
    nb_hidden_layers= power
    embedding_net = get_embedding_net(input_dim, output_dim, kernel_size=3,out_channels_per_layer=out_channels_per_layer, first_hidden_dim=first_hidden_dim, nb_hidden_layers=nb_hidden_layers)
    neural_posterior = posterior_nn(model="nsf", embedding_net=embedding_net)
    inference = NPE(prior=prior, density_estimator=neural_posterior)
    for k in range(nb_seq+1):
        print("round", k)

        theta, C = get_simulations_sbi(proposal, k, nb_train)

        posterior_estimator = inference.append_simulations(theta=theta, x=C, proposal=proposal).train(
                training_batch_size=100
            )
        
            
        posterior = DirectPosterior(prior=prior, posterior_estimator=posterior_estimator)
        
        proposal = posterior.set_default_x(C_ref)

        samples = posterior.sample((nb_posterior_samples,))

        with open(f'{path_sbi}{k}_thetas_accepted', 'wb') as f:
                    #pickle.dump(samples*prior_range, f)
                    pickle.dump(samples, f)

#plot 3 chr
if 0:
    inference_method=NPE
    density_estimator="nsf"
    nb_seq = 2
  
    background_color = "lightgray"
    fig, axes = plt.subplots(2,2, figsize=(12, 10))
    color = plt.cm.RdYlBu_r(np.linspace(0, 1, nb_seq+1))

    for i in range(nb_seq+1):
    
        with open(f'{path_sbi}{i}_thetas_accepted', 'rb') as f:
                    samples = pickle.load(f)
        

                    
        for k, chr in enumerate (list(chr_seq.keys())):
            #axes[k//2, k%2].scatter(samples[:,k], torch.zeros_like(samples[:,k]))
            sns.kdeplot(ax = axes[k//2, k%2], data=samples[:, k], color = color[i], label = f"round {i}")
            
            axes[k//2, k%2].set_xlabel(r"$\theta_{centro}$")
            axes[k//2, k%2].axvline(x=chr_cen[chr], linestyle='--', color="goldenrod", label=rf"$\theta_0$")
            
            axes[k//2, k%2].set_title(rf"$\theta_{{centro}}$ for chr ${k+1}$")
            sec_x = axes[k//2, k%2].secondary_xaxis(location='bottom') #draw the separation between chr    
            sec_x.set_xticks([1, chr_seq[chr]], labels=[])
            sec_x.tick_params('x', length=10, width=1.5)
            extraticks = [1, chr_seq[chr]]
            if i==0:
                    axes[k//2, k%2].set_xticks(list(axes[k//2, k%2].get_xticks()) + extraticks)
           
            axes[k//2, k%2].set_xlim(1, chr_seq[chr])
      
            axes[k//2,k%2].set_facecolor(background_color)


    d_mean_list_sbi = []
    for i in range(nb_seq+1):
        with open(path_sbi+f"{i}_thetas_accepted", 'rb') as f:
                thetas_accepted_sbi = pickle.load(f)

        d_mean_list_sbi.append(mean_distance_best_thetas(thetas_accepted_sbi, theta_ref))
    axes[1,1].plot(range(nb_seq+1), d_mean_list_sbi, '-', color = color[0], alpha = 0.5)

    for i in range(nb_seq+1):
        axes[1,1].plot(i, d_mean_list_sbi[i], 'o', color=color[i],label=f"SBI round {nb_seq}")

    #axes[1,1].plot(range(nb_seq+1), d_mean_list_sbi, '-o', color = "green", alpha = 0.5, label=f"SBI round {nb_seq}")

    axes[1,1].legend()
    axes[1,1].axhline(y=resolution, color="black", linestyle='--', linewidth = 1)
 
    axes[1,1].set_xlabel(r"SBI round")
    axes[1,1].set_ylabel(r"$mean(||\theta_i - \theta_{ref}||)$")
    axes[1,1].set_title(r"mean distance of $\theta$ accepted to $\theta_{ref}$ for $50$ sampled $\theta$")
    axes[1,1].set_facecolor(background_color)

    plt.suptitle(rf"$p(\theta_{{centro}}|C_{{ref}})$"+f"\n {origin} data - noisy : {noisy}"+fr" - $\sigma^2$ {sigma_spot} - res {resolution}" + f"\n {inference_method.__name__} - {density_estimator}  {wavelets}")
    plt.tight_layout()
    plt.show()

#plot 16 chr joint
if 0:
    inference_method=NPE
    density_estimator="nsf"
    nb_seq = 10
  
    background_color = "lightgray"
    fig, axes = plt.subplots(4,5, figsize=(12, 10))
    gs = fig.add_gridspec(4,5)

    ax3 = fig.add_subplot(gs[:, 4])
    color = plt.cm.RdYlBu_r(np.linspace(0, 1, nb_seq+1))

    for i in range(nb_seq+1):
    
        with open(f'{path_sbi}{i}_thetas_accepted', 'rb') as f:
                    samples = pickle.load(f)
        

                    
        for k, chr in enumerate (list(chr_seq.keys())):
            #axes[k//2, k%2].scatter(samples[:,k], torch.zeros_like(samples[:,k]))
            sns.kdeplot(ax = axes[k//4, k%4], data=samples[:, k], color = color[i], label = f"round {i}")
            
            axes[k//4, k%4].set_xlabel(r"$\theta_{centro}$")
            axes[k//4, k%4].axvline(x=chr_cen[chr], linestyle='--', color="goldenrod", label=rf"$\theta_0$")
            
            axes[k//4, k%4].set_title(rf"$\theta_{{centro}}$ for chr ${k+1}$")
            sec_x = axes[k//4, k%4].secondary_xaxis(location='bottom') #draw the separation between chr    
            sec_x.set_xticks([1, chr_seq[chr]], labels=[])
            sec_x.tick_params('x', length=10, width=1.5)
            extraticks = [1, chr_seq[chr]]
            if i==0:
                    axes[k//4, k%4].set_xticks(list(axes[k//4, k%4].get_xticks()) + extraticks)
           
            axes[k//4, k%4].set_xlim(1, chr_seq[chr])
      
            axes[k//4,k%4].set_facecolor(background_color)


    d_mean_list_sbi = []
    for i in range(nb_seq+1):
        with open(path_sbi+f"{i}_thetas_accepted", 'rb') as f:
                thetas_accepted_sbi = pickle.load(f)

        d_mean_list_sbi.append(mean_distance_best_thetas(thetas_accepted_sbi, theta_ref))
    ax3.plot(d_mean_list_sbi, range(nb_seq+1), '-', color = color[0], alpha = 0.5)

    for i in range(nb_seq+1):
        ax3.plot(d_mean_list_sbi[i],i, 'o', color=color[i],label=f"SBI round {i}")

    #ax3.plot(range(nb_seq+1), d_mean_list_sbi, '-o', color = "green", alpha = 0.5, label=f"SBI round {nb_seq}")

    ax3.legend()
    # ax3.set_xticklabels([])  # Remove x-tick labels
    # ax3.set_yticklabels([])  # Remove y-tick labels

    ax3.axvline(x=resolution, color="black", linestyle='--', linewidth = 1)
 
    ax3.set_ylabel(r"SBI round")
    ax3.set_xlabel(r"$mean(||\theta_i - \theta_{ref}||)$")
    ax3.set_title(r"mean distance of $\theta$ accepted to $\theta_{ref}$ for $50$ sampled $\theta$")
    ax3.set_facecolor(background_color)

    plt.suptitle(rf"$p(\theta_{{centro}}|C_{{ref}})$"+f"\n {origin} data - noisy : {noisy}"+fr" - $\sigma^2$ {sigma_spot} - res {resolution}" + f"\n {inference_method.__name__} - {density_estimator}  {wavelets}")
    plt.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.07, wspace=0.4, hspace = 0.7)
    #plt.tight_layout()
    plt.show()

#plot 16 chr row
if 0:
   
    inference_method=NPE
    density_estimator="nsf"
    nb_seq = 10
   

    background_color = "lightgray"
    fig, axes = plt.subplots(4,5, figsize=(12, 10))
    gs = fig.add_gridspec(4,5)

    ax3 = fig.add_subplot(gs[:, 4])
    color = plt.cm.RdYlBu_r(np.linspace(0, 1, nb_seq+1))

    d_mean_list_sbi = []

    for i in range(nb_seq+1):
        d = 0
        with open(f'{path_sbi}{i}_thetas_accepted', 'rb') as f:
                    thetas_accepted_per_dim = pickle.load(f)

        for k, chr in enumerate (list(chr_seq.keys())):
       
            # with open(f'{path_sbi}chr_{k+1}_{i}_thetas_accepted', 'rb') as f:
            #         samples = pickle.load(f)
            # d+= 1.0/dim * mean_distance_best_thetas(samples, theta_ref[k])
            # sns.kdeplot(ax = axes[k//4, k%4], data=samples[:,0], color = color[i], label = f"round {i}")

            
            d+= 1.0/dim * mean_distance_best_thetas(thetas_accepted_per_dim[chr], theta_ref[k])
            sns.kdeplot(ax = axes[k//4, k%4], data=thetas_accepted_per_dim[chr][:,0], color = color[i], label = f"round {i}")
           
        
            axes[k//4, k%4].set_xlabel(r"$\theta_{centro}$")
            axes[k//4, k%4].axvline(x=chr_cen[chr], linestyle='--', color="goldenrod", label=rf"$\theta_0$")
            
            axes[k//4, k%4].set_title(rf"$\theta_{{centro}}$ for chr ${k+1}$")
            sec_x = axes[k//4, k%4].secondary_xaxis(location='bottom') #draw the separation between chr    
            sec_x.set_xticks([1, chr_seq[chr]], labels=[])
            sec_x.tick_params('x', length=10, width=1.5)
            extraticks = [1, chr_seq[chr]]
            if i==0:
                    axes[k//4, k%4].set_xticks(list(axes[k//4, k%4].get_xticks()) + extraticks)
           
            axes[k//4, k%4].set_xlim(1, chr_seq[chr])
      
            axes[k//4,k%4].set_facecolor(background_color)

        d_mean_list_sbi.append(d)
        ax3.plot(d_mean_list_sbi[i],i, 'o', color=color[i],label=f"SBI round {i}")

    ax3.plot(d_mean_list_sbi, range(nb_seq+1), '-', color = color[0], alpha = 0.5)
    
        


    # for i in range(nb_seq+1):
    #     with open(path_sbi+f"{i}_thetas_accepted", 'rb') as f:
    #             thetas_accepted_sbi = pickle.load(f)

    #     d_mean_list_sbi.append(mean_distance_best_thetas(thetas_accepted_sbi, theta_ref[chr_id-1]))
    # ax3.plot(d_mean_list_sbi, range(nb_seq+1), '-', color = color[0], alpha = 0.5)

    # for i in range(nb_seq+1):
    #     ax3.plot(d_mean_list_sbi[i],i, 'o', color=color[i],label=f"SBI round {i}")

    #ax3.plot(range(nb_seq+1), d_mean_list_sbi, '-o', color = "green", alpha = 0.5, label=f"SBI round {nb_seq}")

    ax3.legend()
    # ax3.set_xticklabels([])  # Remove x-tick labels
    # ax3.set_yticklabels([])  # Remove y-tick labels

    ax3.axvline(x=resolution, color="black", linestyle='--', linewidth = 1)
 
    ax3.set_ylabel(r"SBI round")
    ax3.set_xlabel(r"$mean(||\theta_i - \theta_{ref}||)$")
    ax3.set_title(r"mean distance of $\theta$ accepted to $\theta_{ref}$ for $50$ sampled $\theta$")
    ax3.set_facecolor(background_color)

    plt.suptitle(rf"$p(\theta_{{centro}}|C_{{ref}})$"+f"\n {origin} data - noisy : {noisy}"+fr" - $\sigma^2$ {sigma_spot} - res {resolution}" + f"\n {inference_method.__name__} - {density_estimator}  {wavelets}")
    plt.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.07, wspace=0.4, hspace = 0.7)
    #plt.tight_layout()
    plt.show()
                

    
     
    