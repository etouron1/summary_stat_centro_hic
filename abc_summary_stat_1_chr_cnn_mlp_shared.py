import sbi.inference
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
import pyro.distributions as pdist

from sbi import utils as utils


import numpy as np
from simulator import get_num_beads_and_start
from simulator import Correlation_inter_row_average,Correlation_inter_upper_average, Pearson_correlation_vector, Pearson_correlation_col

from itertools import combinations, permutations
import random
import pickle
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # For custom legend handle
from matplotlib.colors import to_rgba


import pandas as pd        
import seaborn as sns

from  scipy.special import binom
from torch.utils.tensorboard import SummaryWriter


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

chr_seq_3_chr = {"chr1": 230209, "chr2": 813179, "chr3": 316619}

chr_cen_3_chr = {'chr1': 151584, 'chr2': 238325, 'chr3': 114499}

chr_seq_16_chr = {"chr1": 230209, "chr2": 813179, "chr3": 316619, "chr4": 1531918, 
           "chr5": 576869, "chr6": 270148, "chr7": 1090947, "chr8": 562644,
            "chr9": 439885, "chr10": 745746, "chr11": 666455, "chr12": 1078176,
            "chr13": 924430, "chr14": 784334, "chr15": 1091290, "chr16": 948063}

chr_cen_16_chr = {'chr1': 151584, 'chr2': 238325, 'chr3': 114499, 'chr4': 449819, 
                 'chr5': 152103, 'chr6': 148622, 'chr7': 497042, 'chr8': 105698, 
                 'chr9': 355742, 'chr10': 436418, 'chr11': 439889, 'chr12': 150946, 
                 'chr13': 268149, 'chr14': 628877, 'chr15': 326703, 'chr16': 556070}

origin="true"
sigma_spot = "variable"
noisy_ref = 1

resolution = 32000

dim = 3
sig_2_ref = 1

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

nb_bead_16_chr, start_bead_16_chr, nb_tot_bead_16_chr = get_num_beads_and_start(chr_seq_16_chr, resolution)
prior_range_16_chr = torch.tensor([230209, 813179, 316619, 1531918,
                                576869, 270148, 1090947, 562644,
                                439885, 745746, 666455, 1078176,
                                924430, 784334, 1091290, 948063])
prior_16_chr = utils.BoxUniform(torch.ones(1), prior_range_16_chr-1)

if origin=="true":
    C_ref =  np.load(f"ref/{dim}_chr_ref_{resolution}_norm_HiC_duan_intra_all.npy")
    C_ref = torch.from_numpy(C_ref).float() 
    C_ref = C_ref + torch.transpose(C_ref, 0,1)

    C_ref_16_chr =  np.load(f"ref/16_chr_ref_{resolution}_norm_HiC_duan_intra_all.npy")
    C_ref_16_chr = torch.from_numpy(C_ref_16_chr).float()
    C_ref_16_chr = C_ref_16_chr + torch.transpose(C_ref_16_chr, 0,1)


plt.matshow(C_ref)
plt.show()
# chr_id=1

# prior = utils.BoxUniform(torch.ones(1), prior_range[chr_id-1]-1)
prior = utils.BoxUniform(torch.ones(1), prior_range-1)

#C_ref_row =  np.load(f"ref/{dim}_chr_{chr_id}_ref_{resolution}_norm_HiC_duan_intra_all.npy")

theta_ref = torch.tensor(list(chr_cen.values()))

# C_ref = torch.from_numpy(C_ref).float() 
# print(C_ref.size())

# C_ref_row = C_ref[start_bead["chr0" + str(chr_id)]:start_bead["chr0" + str(chr_id)] + nb_bead["chr0" + str(chr_id)], :]

# C_ref = vectorise_upper_matrix(C_ref, resolution)

#path = f"simulation_little_genome/{origin}/res_{resolution}/noisy/sigma_{sigma_spot}/summary_stat/CNN/"
#path = f"simulation_little_genome/{origin}/res_{resolution}/noisy/sigma_{sigma_spot}/summary_stat/CNN/sigmoid/sequential/"
path = f"simulation_little_genome/{dim}_chr/{origin}/res_{resolution}/noisy/sigma_{sigma_spot}/row/summary_stat/cnn_mlp_shared/attention_pooling/"

       

def get_cnn_subnet(in_channels, kernel_size, pool_kernel_size, out_channels_per_layer):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels_per_layer[0], kernel_size=kernel_size, padding=kernel_size//2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=pool_kernel_size),
        nn.Conv2d(in_channels=out_channels_per_layer[0], out_channels=out_channels_per_layer[1], kernel_size=kernel_size, padding=kernel_size//2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=pool_kernel_size),
        )
     
def get_mlp_subnet(input_mlp, output_mlp, decay):

        power_first_hidden = 0
        while decay**power_first_hidden < input_mlp:
            power_first_hidden += 1

        first_hidden_dim = decay**(power_first_hidden)
        
        nb_hidden_layers= power_first_hidden-1
        

        layers = [nn.Linear(input_mlp, first_hidden_dim), nn.ReLU()]
        
        for k in range(nb_hidden_layers):
            layers.append(nn.Linear(first_hidden_dim//decay**k, first_hidden_dim//4**(k+1))) #//4**k //4**(k+1)
            layers.append(nn.ReLU())
        layers.append(nn.Linear(first_hidden_dim//4**(k+1), output_mlp)) #//4**(k+1)
        
        layers.append(nn.Sigmoid())
        linear_subnet = nn.Sequential(*layers)
        return linear_subnet

# def simulator(theta, resolution, sigma_spot, noisy):
        
#     if sigma_spot=="variable":
#         sig_2_simu = random.uniform(0.1, 10)
#     else:
#         sig_2_simu = sig_2_ref
               
#     intensity_simu = 100

#     C_simu = torch.zeros((nb_tot_bead,nb_tot_bead))

#     for (chr_row, chr_col) in combinations(chr_seq.keys(),r=2):
        
#         n_row = chr_seq[chr_row]//resolution
#         n_col = chr_seq[chr_col]//resolution
#         index_row = int(chr_row[chr_row.find("chr")+3:])-1
#         index_col = int(chr_col[chr_col.find("chr")+3:])-1
#         c_i_simu = theta[index_row]//resolution
#         c_j_simu = theta[index_col]//resolution

#         def simulator_1_bloc(n_row, n_col, c_i, c_j, sig_2, intensity, noisy=noisy):
            
#             # Simulate a noisy matrix C_{n_row x n_col} with a gaussian spot at (c_i, c_j) of size sig_2 
            
#             C = torch.zeros((n_row, n_col))
            
#             distr = MultivariateNormal(torch.tensor([c_i, c_j]), sig_2*torch.eye(2))
                
#             indices = torch.tensor([[(i, j) for j in range(len(C[0]))] for i in range(len(C))])
            
#             C = intensity*torch.exp(distr.log_prob(indices))
            
#             if noisy:
#                 mean = intensity * torch.exp(distr.log_prob(torch.tensor([c_i,c_j]))) * 0.1 / 2 
#                 sigma = intensity * torch.exp(distr.log_prob(torch.tensor([c_i,c_j]))) * 0.1 / 2
       
#                 noise = mean + sigma*torch.randn((n_row, n_col))

#             else:
#                 noise = torch.zeros_like(C)
            
#             return C+noise
        
#         C_simu[start_bead[chr_row]:start_bead[chr_row]+nb_bead[chr_row]-1, start_bead[chr_col]:start_bead[chr_col]+nb_bead[chr_col]-1] = simulator_1_bloc(n_row,n_col, c_i_simu, c_j_simu, sig_2_simu, intensity_simu, noisy=noisy)
    
#     return C_simu + torch.transpose(C_simu, 0,1)


def simulator(theta, resolution, sigma_spot, noisy):
        
    if sigma_spot=="variable":
        sig_2_simu = random.uniform(0.1, 10)
    else:
        sig_2_simu = sig_2_ref
               
    intensity_simu = 100

    C_simu = torch.zeros((nb_tot_bead,nb_tot_bead))

    for (chr_row, chr_col) in combinations(chr_seq.keys(),r=2):
        
        n_row = chr_seq[chr_row]//resolution
        n_col = chr_seq[chr_col]//resolution
        index_row = int(chr_row[chr_row.find("chr")+3:])-1
        index_col = int(chr_col[chr_col.find("chr")+3:])-1
        c_i_simu = theta[index_row]//resolution
        c_j_simu = theta[index_col]//resolution

        def simulator_1_bloc(n_row, n_col, c_i, c_j, sig_2, intensity, noisy=noisy, device=device):
         
            y, x = torch.meshgrid(torch.arange(n_row, device=device), torch.arange(n_col, device=device), indexing='ij')
            # print('x', 'y', x.size(),x, y.size(),  y)
            exponent = -((x - c_j)**2 + (y - c_i)**2) / (2 * sig_2)
            norm_const = 1.0/(2*torch.pi*sig_2)
            # print(exponent.size())
            C = intensity * norm_const*torch.exp(exponent)

            if noisy:
                level_noise = torch.tensor([0.05])
                mean = intensity * norm_const * level_noise 
                sigma = intensity * norm_const * level_noise
                
                noise = mean + sigma*torch.randn((n_row, n_col))
                # print('manual', noise)
                
                sig = np.sqrt(sig_2)
                
                i0 = max(0, int(c_i) - int(sig))
                i1 = min(n_row, int(c_i) + int(sig) + 1)

                j0 = max(0, int(c_j) - int(sig))
                j1 = min(n_col, int(c_j) + int(sig) + 1)

                noise[i0:i1, :] = 0
                noise[:, j0:j1] = 0 
                C += noise

            return C
        
        C_simu[start_bead[chr_row]:start_bead[chr_row]+nb_bead[chr_row]-1, start_bead[chr_col]:start_bead[chr_col]+nb_bead[chr_col]-1] = simulator_1_bloc(n_row,n_col, c_i_simu, c_j_simu, sig_2_simu, intensity_simu, noisy=noisy)
    
    return C_simu + torch.transpose(C_simu, 0,1)

def simulator_16_chr(theta, resolution, sigma_spot, noisy):

        if sigma_spot=="variable":
            sig_2_simu = random.uniform(0.1, 10)
        else:
            sig_2_simu = sig_2_ref

        intensity_simu = 100

        C_simu = torch.zeros((nb_tot_bead_16_chr,nb_tot_bead_16_chr), device=device)

        for (chr_row, chr_col) in combinations(chr_seq_16_chr.keys(),r=2):

            n_row = chr_seq_16_chr[chr_row]//resolution
            n_col = chr_seq_16_chr[chr_col]//resolution
            index_row = int(chr_row[chr_row.find("chr")+3:])-1
            index_col = int(chr_col[chr_col.find("chr")+3:])-1
            c_i_simu = theta[index_row]//resolution
            c_j_simu = theta[index_col]//resolution

            def simulator_1_bloc(n_row, n_col, c_i, c_j, sig_2, intensity, noisy=noisy, device=device):

                y, x = torch.meshgrid(torch.arange(n_row, device=device), torch.arange(n_col, device=device), indexing='ij')
                # print('x', 'y', x.size(),x, y.size(),  y)
                exponent = -((x - c_j)**2 + (y - c_i)**2) / (2 * sig_2)
                norm_const = 1.0/(2*torch.pi*sig_2)
                # print(exponent.size())
                C = intensity * norm_const*torch.exp(exponent)

                if noisy:
                    level_noise = torch.tensor([0.05], device=device)
                    mean = intensity * norm_const * level_noise
                    sigma = intensity * norm_const * level_noise

                    noise = mean + sigma*torch.randn((n_row, n_col), device=device)
                    # print('manual', noise)

                    sig = np.sqrt(sig_2)

                    i0 = max(0, int(c_i) - int(sig))
                    i1 = min(n_row, int(c_i) + int(sig) + 1)

                    j0 = max(0, int(c_j) - int(sig))
                    j1 = min(n_col, int(c_j) + int(sig) + 1)

                    noise[i0:i1, :] = 0
                    noise[:, j0:j1] = 0
                    C += noise

                return C

            C_simu[start_bead_16_chr[chr_row]:start_bead_16_chr[chr_row]+nb_bead_16_chr[chr_row]-1, start_bead_16_chr[chr_col]:start_bead_16_chr[chr_col]+nb_bead_16_chr[chr_col]-1] = simulator_1_bloc(n_row,n_col, c_i_simu, c_j_simu, sig_2_simu, intensity_simu, noisy=noisy)

        return C_simu + torch.transpose(C_simu, 0,1)

# def simulator_row(chr_id, theta, resolution, sigma_spot, noisy):
#     """
#     Simulate 1 row of blocs of the contact matrix
#     """

#     chr_num = "chr" + str(chr_id)

#     if sigma_spot=="variable":
#         sig_2_simu = random.uniform(0.1, 10)
#     else:
#         sig_2_simu = sig_2_ref
               
#     intensity_simu = 100

#     C_simu = np.zeros((nb_bead[chr_num],nb_tot_bead))

#     n_row = nb_bead[chr_num]
#     index_row = chr_id-1
    
#     for chr_col in chr_seq.keys():
#         if chr_col != chr_num:
        
#             n_col = nb_bead[chr_col]
#             index_col = int(chr_col[chr_col.find("chr")+3:])-1

#             c_i_simu = theta[index_row]//resolution
#             c_j_simu = theta[index_col]//resolution
            
#             def simulator_1_bloc(n_row, n_col, c_i, c_j, sig_2, intensity, noisy=noisy):
                
#                 # Simulate a noisy matrix C_{n_row x n_col} with a gaussian spot at (c_i, c_j) of size sig_2 
                
#                 C = torch.zeros((n_row, n_col))
       
#                 distr = MultivariateNormal(torch.tensor([c_i, c_j]), sig_2*torch.eye(2))
                    
#                 indices = torch.tensor([[(i, j) for j in range(len(C[0]))] for i in range(len(C))])
                
#                 C = intensity*torch.exp(distr.log_prob(indices))
                
#                 if noisy:
#                     mean = intensity * torch.exp(distr.log_prob(torch.tensor([c_i,c_j]))) * 0.1 / 2 
#                     sigma = intensity * torch.exp(distr.log_prob(torch.tensor([c_i,c_j]))) * 0.1 / 2
        
#                     noise = mean + sigma*torch.randn((n_row, n_col))

#                 else:
#                     noise = torch.zeros_like(C)
                
#                 return C+noise
          
#             C_simu[:start_bead[chr_num]+nb_bead[chr_num], start_bead[chr_col]:start_bead[chr_col]+nb_bead[chr_col]] = simulator_1_bloc(n_row,n_col, c_i_simu, c_j_simu, sig_2_simu, intensity_simu, noisy=noisy)


#     return C_simu

# def vectorise_upper_matrix(C, resolution):
    
#     first_bloc = 1
#     for (chr_1, chr_2) in combinations(chr_seq.keys(), r=2):
            
#         start_1, end_1 = start_bead[chr_1], start_bead[chr_1] + nb_bead[chr_1] #get the start bead id and the end bead id for the chr
#         start_2, end_2 = start_bead[chr_2], start_bead[chr_2] + nb_bead[chr_2] #get the start bead id and the end bead id for the chr
#         inter_bloc = C[start_1:end_1, start_2:end_2] 
#         if first_bloc:
#             C_vector = torch.flatten(inter_bloc)
#             first_bloc = 0
#         else: 
#             C_vector = torch.hstack((C_vector, torch.flatten(inter_bloc)))
   
#     return C_vector


def get_simulations_C(nb_train):
    # theta = theta_ref.repeat(nb_train,1)

    # theta[:, chr_id-1] = prior.sample((nb_train,)).squeeze()
    theta = prior.sample((nb_train,))

    # C = torch.zeros(nb_train,C_ref.size(0))
    
    C = torch.zeros(nb_train, C_ref.size(0), C_ref.size(1))
  
    for k in range(nb_train):

        C_tmp = simulator(theta[k], resolution, sigma_spot, noisy_ref)       
        # plt.matshow(C_tmp)
        # plt.show()
        #C_tmp=np.delete(C_tmp, range(start_bead['chr'+str(chr_id)], start_bead['chr'+str(chr_id)] + nb_bead['chr'+str(chr_id)]), axis =1)

        # C_tmp = torch.from_numpy(C_tmp).float() 


        # C[k] = vectorise_upper_matrix(C_tmp, resolution)

        #C[k] = C_tmp[start_bead["chr0" + str(chr_id)]:start_bead["chr0" + str(chr_id)] + nb_bead["chr0" + str(chr_id)], :]
        C[k] = C_tmp

    #theta = theta[:,chr_id-1]/prior_range[chr_id-1] #norm thetas
    theta = theta/prior_range #norm thetas
    
    return theta, C
     

     
def get_dataloaders(nb_train,
        training_batch_size: int = 200,
        validation_fraction: float = 0.1,
    ):

        theta, C  = get_simulations_C(nb_train)
        train_loader = {}
        val_loader = {}
        for chr_id in range(1,dim+1):
            
            C_row = C[:, start_bead['chr'+str(chr_id)]:start_bead['chr'+str(chr_id)] + nb_bead['chr'+str(chr_id)], :]
            
            # ###### cut intra #######
            # C_left = C[:,start_bead['chr'+str(chr_id)]:start_bead['chr'+str(chr_id)] + nb_bead['chr'+str(chr_id)],  : start_bead['chr'+str(chr_id)]]
            # C_right = C[:, start_bead['chr'+str(chr_id)]:start_bead['chr'+str(chr_id)] + nb_bead['chr'+str(chr_id)], start_bead['chr'+str(chr_id)] + nb_bead['chr'+str(chr_id)]:]
            
            # try:
                
            #     C_row = torch.cat((C_left, C_right), dim=-1)
            # except :
                
            #     if C_left.size(-1) > 0:
            #         C_row = C_left
            #     elif C_right.size(-1) > 0:
            #         C_row = C_right
            # #######################      
                 
            dataset = data.TensorDataset(theta[:,chr_id-1], C_row) #out 1 chr
            # dataset = data.TensorDataset(theta, C_row) #out 16 chr
            # Get total number of training examples.
            num_examples = theta.size(0)
            # Select random train and validation splits from (theta, x) pairs.
            num_training_examples = int((1 - validation_fraction) * num_examples)
            num_validation_examples = num_examples - num_training_examples

        
            # Seperate indicies for training and validation
            permuted_indices = torch.randperm(num_examples)
            train_indices, val_indices = (
                permuted_indices[:num_training_examples],
                permuted_indices[num_training_examples:],
            )

            # Create training and validation loaders using a subset sampler.
            # Intentionally use dicts to define the default dataloader args
            # Then, use dataloader_kwargs to override (or add to) any of these defaults
            # https://stackoverflow.com/questions/44784577/in-method-call-args-how-to-override-keyword-argument-of-unpacked-dict
            train_loader_kwargs = {
                "batch_size": min(training_batch_size, num_training_examples),
                "drop_last": True,
                "sampler": SubsetRandomSampler(train_indices.tolist()),
                # "shuffle": True
            }
            val_loader_kwargs = {
                "batch_size": min(training_batch_size, num_validation_examples),
                "shuffle": False,
                "drop_last": True,
                "sampler": SubsetRandomSampler(val_indices.tolist()),
            }
            
            train_loader['chr'+str(chr_id)] = data.DataLoader(dataset, **train_loader_kwargs)
            val_loader['chr'+str(chr_id)] = data.DataLoader(dataset, **val_loader_kwargs)

        return train_loader, val_loader



def converged(epochs_since_last_improvement, stop_after_epochs):

        converged = False

        # If no validation improvement over many epochs, stop training.
        if epochs_since_last_improvement > stop_after_epochs - 1:
            converged = True

        return converged

# def loss(theta, C, DNN):
    
#     return torch.mean((DNN(C)-theta)**2, dim=1)

def loss(theta, theta_hat):

    #return torch.mean((theta_hat-theta)**2, dim=1) #out 16 chr
    return torch.mean((theta_hat-theta)**2) #out 1 chr

def mean_distance_best_thetas(thetas_accepted, theta_ref):

    d_mean = 0

    for theta in thetas_accepted:
        try:
            d = 0
            for i in range(len(theta)):
                d+= (theta[i] - theta_ref[i])**2
            d_mean += np.sqrt(d/len(theta))
        except:
            d_mean += np.abs(theta-theta_ref)



    return d_mean/len(thetas_accepted)

def DNN_eval(C_row, chr_id, cnn_subnet, linear_subnets):

    theta_hat = torch.zeros(1)
    
    # print("chr", chr_id)

    C_row = C_row.unsqueeze(0)
    features = cnn_subnet(C_row.unsqueeze(0))          # → (B, F)
    
    features = features.view(C_row.size(0), -1)  # flatten
                
    # theta_hat = linear_subnets[chr_id-1](features).squeeze()[chr_id-1]  # → (B,) out 16 chr
    theta_hat = linear_subnets[chr_id-1](features).squeeze()  # → (B,) out 1 chr
    return theta_hat

def slice_to_key(d, end_key, include_end=True):
        result = {}
        for k, v in d.items():
            result[k] = v
            if k == end_key:
                if not include_end:
                    result.pop(k)
                break
        return result

class AttentionPooling(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        # Small MLP to compute attention scores
        self.attn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)   # outputs scalar attention score
        )

    def forward(self, x):
        """
        x: (B, C, H, W) feature maps
        """
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H*W).permute(0, 2, 1)   # (B, N, C)

        # Compute attention scores
        attn_scores = self.attn(x_flat)              # (B, N, 1)
        attn_weights = F.softmax(attn_scores, dim=1) # (B, N, 1)

        # Weighted sum mult attn_weights for all x in the same C
        
        pooled = torch.sum(attn_weights * x_flat, dim=1)  # (B, N, C) -> (B, C)
        return pooled

############## convolution ############################

# m = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5)
# input = torch.randn(100, 1, 10,10)
# #input of size [N,C,H, W]
# # N:batch size,
# # C: number of channels,
# # H: height of input planes in pixels,
# # W: width in pixels.
# print(input.size())
# output = m(input)
# print(output.size())
# # output = F.conv2d(input, 5)
# # print(output.size())

# output = F.max_pool2d(input=output, kernel_size=2)
# print(output.size())

##################### ABC PEARSON ##############################

##################### TRAINING #################################
if 1:
    # writer = SummaryWriter(log_dir=path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ##### shared CNN 3/16 chr #####
    kernel_size=3
    pool_kernel_size = 2
    out_channels_per_layer = [6,12]
    decay = 4
    cnn_subnet = get_cnn_subnet(in_channels=1, kernel_size=kernel_size, pool_kernel_size=pool_kernel_size, out_channels_per_layer=out_channels_per_layer)
    ######################

    ##### indep MLP ######

    
    output_mlp = 1 #out 1 chr
    # output_mlp = 16 #out 16 chr
    input_mlp = out_channels_per_layer[1]
    
    mlp_subnet = get_mlp_subnet(input_mlp=input_mlp, output_mlp=output_mlp, decay=decay)
    
    # pool = nn.AdaptiveAvgPool2d((1, 1)) #average pooling (B, C, H, W) -> (B, C) average over H, W
    pool = AttentionPooling(in_dim=out_channels_per_layer[1])
    ###################### 
    # print("cnn")
    # for name, param in cnn_subnet.named_parameters():
    #     print(f"{name}: {param.shape}")
    # print("mlp")
    # for i, mlp in enumerate(linear_subnets):
    #     print(i)
    #     for name, param in mlp.named_parameters():
    #         print(f"{name}: {param.shape}")
    print(cnn_subnet)
    print(pool)
    print(mlp_subnet)
    # with open(path+f"dnn_structure.txt", "w") as f:
    #      f.write(str(cnn_subnet))
    #      f.write(str(pool))
    #      f.write(str(mlp_subnet))
    ##### data loader####
    nb_train= 5000
    print('nb train', nb_train)



    
    print("data loader")

    train_loader, val_loader = get_dataloaders(nb_train)

    print("train loader and val loader done") 
    
         
    # with open(path+"train_loader", "wb") as f:
    #      pickle.dump(train_loader, f)
    # with open(path+"val_loader", "wb") as f:
    #      pickle.dump(val_loader, f)
    # print("loading data loader")
    # with open(path+"train_loader", "rb") as f:
    #      train_loader = pickle.load(f)
    # with open(path+"val_loader", "rb") as f:
    #      val_loader = pickle.load(f)   
    ###################### 

    stop_after_epochs = 20
    learning_rate = 5e-4
    max_num_epochs=  200 #2**31 - 1

    #optimizer = torch.optim.Adam(list(cnn_subnet.parameters()) + list(mlp_subnet.parameters()), lr=learning_rate)
    optimizer = torch.optim.Adam(list(cnn_subnet.parameters()) + list(pool.parameters()) + list(mlp_subnet.parameters()), lr=learning_rate)
    epoch, epoch_since_last_improvement, val_loss, best_val_loss = 0, 0, float("Inf"), float("Inf")
    train_loss_list = []
    val_loss_list = []
    loss_theta_ref = []
    distance_to_theta_train = []
    distance_to_theta_val = []
    distance_to_theta = {key : [] for key in chr_seq.keys()}
    distance_simu_to_theta = {key : [] for key in chr_seq.keys()}
    
    taille_list = ['chr3', 'chr6', 'chr9', 'chr12', 'chr15']
    croissant_theta_ref = {key : { k : [] for k in taille_list} for key in chr_seq_16_chr.keys()}
    croissant_theta_simu = {key : { k : [] for k in taille_list} for key in chr_seq_16_chr.keys()}

    train_chr_start = 1
    train_chr_end = 2
    ################# 10 C synthetics ####################
    nb_C_synth = 10
    theta_synth = []
    C_synth = []
    for k in range(nb_C_synth):

        theta = prior.sample()
        theta_synth.append(theta)

        C_tmp = simulator(theta, resolution, sigma_spot, noisy_ref)
        C_tmp = C_tmp.reshape(1,1,C_tmp.size(0), C_tmp.size(1))
        C_synth.append(C_tmp)
    #####################################################

    ################# 10 C synthetics croissant ####################
        
    theta_synth_croissant = []
    C_synth_croissant = [] #10 C de taille 16 chr

    for k in range(nb_C_synth):

        theta = prior_16_chr.sample()
        theta_synth_croissant.append(theta)

        C_tmp = simulator_16_chr(theta, resolution, sigma_spot, noisy_ref)
        C_tmp = C_tmp.reshape(1,1,C_tmp.size(0), C_tmp.size(1))
        C_synth_croissant.append(C_tmp)

    #######################################################################

    #while epoch <= max_num_epochs and not converged(epoch_since_last_improvement, stop_after_epochs):
    while epoch <= max_num_epochs:
        iter_loader = {name: iter(loader) for name, loader in train_loader.items()}

        
        print("epoch", epoch)
        
        #for chr_id in range(1,dim+1):
            # print('chr', chr_id)
            
           
            # Train for a single epoch.
            # DNN.train()
            # print('train')
        train_loss_sum = 0
        estimation_train = 0
        
        for k in range(sum(len(loader) for loader in train_loader.values())):
            chr_id = torch.randint(train_chr_start, train_chr_end+1, (1,)).item()
            loader_name = 'chr'+str(chr_id)
            try:
                batch = next(iter_loader[loader_name])
            except StopIteration:
                # Iterator exhausted, recreate it
                iter_loader[loader_name] = iter(train_loader[loader_name])
                batch = next(iter_loader[loader_name])
            
            optimizer.zero_grad()
            # Get batches on current device.
            theta_batch, x_batch= (
                batch[0].to(device),
                batch[1].to(device),
            )
            
            # print('train')
            # print(chr_id)
            # plt.matshow(x_batch[0,:,:])
            # plt.show()
            features = cnn_subnet(x_batch.unsqueeze(1))          #(B, C, H, W)
            features = pool(features)                           #(B, C, 1, 1)
            features = features.view(theta_batch.size(0), -1)  # flatten (B, C)

            theta_hat = mlp_subnet(features).squeeze()  # → (B,1) 
            
            estimation_train += torch.mean(torch.abs((theta_hat-theta_batch)*prior_range[chr_id-1]), dim=0).item()
            
        
            
            train_losses = loss(theta_batch,theta_hat) #compute loss over batch i
            
            train_loss = torch.mean(train_losses) #mean loss over batch i
            train_loss_sum += train_losses.sum().item() #sum loss over batchs for chr k

            train_loss.backward() #differentiate wrt batch i
            
            optimizer.step()
        
            
            
        nb_item_train = sum(len(train_loader['chr'+str(chr)]) * train_loader['chr'+str(chr)].batch_size for chr in range(train_chr_start,train_chr_end+1))
        
        train_loss_average = train_loss_sum / (
            nb_item_train
        ) #average loss over all items of all batchs for all chr 
        
        nb_batchs_train = sum(len(train_loader['chr'+str(chr)]) for chr in range(train_chr_start,train_chr_end+1)) #nb_batchs for all chr
        
        distance_to_theta_train.append(estimation_train/nb_batchs_train)

        val_loss_sum = 0
        estimation_val = 0
        with torch.no_grad():
            
            for chr in range (train_chr_start, train_chr_end+1):
            
                for batch in val_loader['chr'+str(chr)]:
                    theta_batch, x_batch = (
                        batch[0].to(device),
                        batch[1].to(device),
                        
                    )
                    # print('val')
                    # print(chr)
                    # plt.matshow(x_batch[0,:,:])
                    # plt.show()
                    features = cnn_subnet(x_batch.unsqueeze(1))          #(B, C, H, W)
                    features = pool(features)                           # (B, C, 1,1)          
                    features = features.view(theta_batch.size(0), -1)  # flatten (B, C)  
                    theta_hat = mlp_subnet(features).squeeze()  # → (B,1) 

                    estimation_val += torch.mean(torch.abs((theta_hat-theta_batch)*prior_range[chr-1]), dim=0).item()

                

                    # Take negative loss here to get validation log_prob.
                    # val_losses = loss(theta_batch,x_batch, DNN)
                    val_losses = loss(theta_batch,theta_hat) #loss over batch i
                    val_loss_sum += val_losses.sum().item() #sum loss over batchs for chr k


            
            nb_item_val = sum(len(val_loader['chr'+str(chr)]) * val_loader['chr'+str(chr)].batch_size for chr in range(train_chr_start, train_chr_end+1))
            
            # Take mean over all validation samples.
            val_loss_average = val_loss_sum / (
                nb_item_val
            ) #average loss over all items of all batches for chr k

            nb_batchs_val = sum(len(val_loader['chr'+str(chr)]) for chr in range(train_chr_start,train_chr_end+1))
            
            distance_to_theta_val.append(estimation_val/nb_batchs_val)

            # for name, param in cnn_subnet.named_parameters():
            #     writer.add_histogram(f"cnn_{name}.weights", param, epoch)
            #     if param.grad is not None:
            #         writer.add_histogram(f"cnn_{name}.grads", param.grad, epoch)
            # for i, mlp in enumerate(linear_subnets):
            #     for name, param in mlp.named_parameters():
            #         writer.add_histogram(f"mlp_chr_{i+1}_{name}.weights", param, epoch)
            #         if param.grad is not None:
            #             writer.add_histogram(f"mlp_chr{i+1}_{name}.grads", param.grad, epoch)

            # writer.add_scalar("Loss/train", train_loss_average, epoch)


        train_loss_list.append(train_loss_average) #loss over all chr
        
        # if len(val_loss_list) > 0 and val_loss > val_loss_list[-1]:#start overfit
        #     epoch_since_last_improvement +=1

        val_loss_list.append(val_loss_average) # loss over all chr

        

        loss_ref_average = 0
        for num_chr in range(train_chr_start, train_chr_end+1):
            
            C_ref_row = C_ref[start_bead['chr'+str(num_chr)]:start_bead['chr'+str(num_chr)] + nb_bead['chr'+str(num_chr)], :]
            C_ref_row = C_ref_row.reshape(1,1,C_ref_row.size(0), C_ref_row.size(1))
            
            # ###### cut intra #######
            # C_ref_left = C_ref[start_bead['chr'+str(num_chr)]:start_bead['chr'+str(num_chr)] + nb_bead['chr'+str(num_chr)],  : start_bead['chr'+str(num_chr)]]
            # C_ref_right = C_ref[start_bead['chr'+str(num_chr)]:start_bead['chr'+str(num_chr)] + nb_bead['chr'+str(num_chr)], start_bead['chr'+str(num_chr)] + nb_bead['chr'+str(num_chr)]:]
            
            # try:
                
            #     C_ref_row = torch.cat((C_ref_left, C_ref_right), dim=-1)
            # except :
                
            #     if C_ref_left.size(-1) > 0:
            #         C_ref_row = C_ref_left
            #     elif C_ref_right.size(-1) > 0:
            #         C_ref_row = C_ref_right
            
            # C_ref_row = C_ref_row.reshape(1,1, C_ref_row.size(0), C_ref_row.size(1))
            
            # #######################  
            # print(num_chr)
            # plt.matshow(C_ref_row[0,0,:,:])
            # plt.show()

            
            theta_ref = chr_cen['chr'+str(num_chr)]
            
            features = cnn_subnet(C_ref_row)          #(B, C, H, W)
            features = pool(features)            
            features = features.view(1, -1)  # flatten (B, C)  
            
            theta_hat = mlp_subnet(features).squeeze().item()*prior_range[num_chr-1]  # (B, 1) 
            loss_ref_average += (theta_hat/prior_range[num_chr-1]-theta_ref/prior_range[num_chr-1])**2
            

            distance_to_theta['chr'+str(num_chr)].append(torch.abs(theta_hat-theta_ref).item())
            

            distance_simu = 0
            for k in range(nb_C_synth):
                
                # theta = prior.sample()

                # C_tmp = simulator(theta, resolution, sigma_spot, noisy_ref)  
                theta = theta_synth[k]
                C_tmp = C_synth[k]
                C_tmp_row = C_tmp[:,:,start_bead['chr'+str(num_chr)]:start_bead['chr'+str(num_chr)] + nb_bead['chr'+str(num_chr)], :]
                
                # ###### cut intra #######
                # C_tmp_left = C_tmp[start_bead['chr'+str(num_chr)]:start_bead['chr'+str(num_chr)] + nb_bead['chr'+str(num_chr)],  : start_bead['chr'+str(num_chr)]]
                # C_tmp_right = C_tmp[start_bead['chr'+str(num_chr)]:start_bead['chr'+str(num_chr)] + nb_bead['chr'+str(num_chr)], start_bead['chr'+str(num_chr)] + nb_bead['chr'+str(num_chr)]:]
                
                # try:
                    
                #     C_tmp_row = torch.cat((C_tmp_left, C_tmp_right), dim=-1)
                # except :
                    
                #     if C_tmp_left.size(-1) > 0:
                #         C_tmp_row = C_tmp_left
                #     elif C_tmp_right.size(-1) > 0:
                #         C_tmp_row = C_tmp_right
                # C_tmp_row = C_tmp_row.reshape(1,1, C_tmp_row.size(0), C_tmp_row.size(1))
                
                # #######################
                # print(num_chr)
                # plt.matshow(C_tmp_row[0,0,:,:])
                # plt.show()
                features = cnn_subnet(C_tmp_row)          #(B, C, H, W)
                features = pool(features)               #(B,C,1,1)
                features = features.view(1, -1)  # flatten (B, C)  

                theta_hat = mlp_subnet(features).squeeze().item() * prior_range[num_chr-1]  # (B, 16) 
                
                distance_simu+=torch.abs(theta_hat-theta[num_chr-1]).item()*0.1
            distance_simu_to_theta['chr'+str(num_chr)].append(distance_simu)

        loss_theta_ref.append(loss_ref_average/(train_chr_end-train_chr_start+1))

        ################ test C croissant ######################
        for num_chr in range(1, 16+1):
            chr_id = 'chr'+str(num_chr)
            for taille in taille_list:
                chr_seq_croissant = slice_to_key(chr_seq_16_chr, taille) #cut dict until taille
                # print(chr_seq_croissant)
                nb_bead_croissant, start_bead_croissant, nb_tot_bead_croissant = get_num_beads_and_start(chr_seq_croissant, resolution)

                C_ref_row = C_ref_16_chr[start_bead_16_chr[chr_id]:start_bead_16_chr[chr_id] + nb_bead_16_chr[chr_id], :start_bead_croissant[taille] + nb_bead_croissant[taille]]
                C_ref_row = C_ref_row.reshape(1,1,C_ref_row.size(0), C_ref_row.size(1))
                # print(chr_id)
                # print(taille)
                # print(C_ref_row.size())
                # plt.matshow(C_ref_row[0,0,:,:])  
                # plt.show()
                theta_ref = chr_cen_16_chr[chr_id]
                features = cnn_subnet(C_ref_row)          #(B, C, H, W)
                features = pool(features)            
                features = features.view(1, -1)  # flatten (B, C)  
                
                theta_hat = mlp_subnet(features).squeeze().item()*prior_range_16_chr[num_chr-1]  # (B, 1) 
                
                croissant_theta_ref[chr_id][taille].append(torch.abs(theta_hat-theta_ref))


                croissant_simu = 0
                for k in range(nb_C_synth):
     
                    
                    theta = theta_synth_croissant[k]
                    C_tmp_row = C_synth_croissant[k][:,:,start_bead_16_chr[chr_id]:start_bead_16_chr[chr_id] + nb_bead_16_chr[chr_id], :start_bead_croissant[taille] + nb_bead_croissant[taille]]
                    # print(theta)
                    # print(chr_id)
                    # plt.matshow(C_tmp_row[0,0,:,:])
                    # plt.show()
                    features = cnn_subnet(C_tmp_row)          #(B, C, H, W)
                    features = pool(features)               #(B,C,1,1)
                    features = features.view(1, -1)  # flatten (B, C)  
                    theta_hat = mlp_subnet(features).squeeze().item() * prior_range_16_chr[num_chr-1]  # (B, 16) 

                    croissant_simu += torch.abs(theta_hat-theta[num_chr-1])*1.0/nb_C_synth
                croissant_theta_simu[chr_id][taille].append(croissant_simu)



        #######################################################


        epoch += 1

        # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        # DNN.zero_grad(set_to_none=True)
    # writer.close()
    # print(distance_to_theta)
    # print(distance_simu_to_theta)
    print(f"converged after {epoch} epochs with a validation loss {val_loss_list[-1]}")
    plt.figure()
    plt.title('loss')
    plt.plot(range(epoch), torch.sqrt(torch.tensor(train_loss_list)), label="train")
    plt.plot(range(epoch), torch.sqrt(torch.tensor(val_loss_list)), label="val")
    plt.legend()
    plt.tight_layout()
    plt.show()
    # plt.savefig(path+f"{nb_train}_loss_layers_kernel_{kernel_size}_conv_{out_channels_per_layer[0]}_{out_channels_per_layer[1]}_theta_16.png")
    #plt.savefig(path+f"{nb_train}_loss_linear_{decay_linear}_kernel_{kernel_size}.png")
        #plt.savefig(path+f"{nb_train}_loss_linear_{decay_linear}.png")
    plt.figure()
    plt.title('loss C_ref')
    plt.plot(range(epoch), torch.sqrt(torch.tensor(loss_theta_ref)))
    plt.legend()
    plt.tight_layout()
    plt.show()

    for num_chr in range(train_chr_start,train_chr_end+1):
        plt.figure()
        plt.plot(range(epoch), distance_to_theta['chr'+str(num_chr)])
        plt.title('chr'+str(num_chr)+' ref')
        plt.ylabel('|theta_ref - DNN(C_ref)|')
        plt.axhline(y=resolution)
        plt.tight_layout()
        # plt.savefig(path+f"{nb_train}_distance_to_theta_chr_{num_chr}.png")
        plt.show()

    for num_chr in range(train_chr_start,train_chr_end+1):
        plt.figure()
        plt.plot(range(epoch), distance_simu_to_theta['chr'+str(num_chr)])
        plt.title('chr'+str(num_chr) + ' simu')
        plt.ylabel('|theta_ref - DNN(C_ref)|')
        plt.axhline(y=resolution)
        plt.tight_layout()
        # plt.savefig(path+f"{nb_train}_distance_simu_to_theta_chr_{num_chr}.png")
        plt.show()
    
    
    plt.figure()
    plt.plot(range(epoch), distance_to_theta_train, label='train')
    plt.plot(range(epoch), distance_to_theta_val, label='val')
    plt.title(' train/val')
    plt.ylabel('|theta_ref - DNN(C_ref)|')
    plt.axhline(y=resolution)
    plt.legend()
    plt.tight_layout()
    plt.show()
        # plt.savefig(path+f"{nb_train_dnn}_distance_train_val_to_theta_chr_{num_chr}.png")
    ############### croissant test #################
    df_min = []
    df_moy = []
    df_max = []
    for num_chr in range(1, 16+1):
        # print('chr'+str(num_chr))
        for taille in taille_list:
            values = croissant_theta_ref['chr'+str(num_chr)][taille][-50:]
            min_croissant_theta_ref = min(values)
            max_croissant_theta_ref = max(values)
            moy_croissant_theta_ref = sum(values)/len(values)
            df_min.append({'chr': f'chr{num_chr}',
                'taille': taille,
                'stat': 'min',
                'value': min_croissant_theta_ref.item()})
            df_moy.append({'chr': f'chr{num_chr}',
                'taille': taille,
                'stat': 'mean',
                'value': moy_croissant_theta_ref.item()})
            df_max.append({'chr': f'chr{num_chr}',
                'taille': taille,
                'stat': 'max',
                'value': max_croissant_theta_ref.item()})
    df_min = pd.DataFrame(df_min)
    df_moy = pd.DataFrame(df_moy)
    df_max = pd.DataFrame(df_max)
    df = [df_min, df_moy, df_max]
    
    color_min = [to_rgba('green') for _ in range(len(taille_list))]
    color_moy = [to_rgba('orange') for _ in range(len(taille_list))]
    color_max = [to_rgba('red') for _ in range(len(taille_list))]
    colors = [color_min, color_moy, color_max]
    plt.figure()
    for k in range(3):
        sns.stripplot(data=df[k],x='chr',y='value',hue='taille',dodge=True,jitter=False,size=6, palette=colors[k])
    plt.title(f'ref - Min-mean-max over last 50 epochs')
    plt.ylabel('|theta_ref - DNN(C_ref)|')
    handles = [
    Line2D([0], [0], marker='o', color='red', label='max',
        markerfacecolor='red', markersize=6),
    Line2D([0], [0], marker='o', color='orange', label='mean',
        markerfacecolor='orange', markersize=6),
    Line2D([0], [0], marker='o', color='green', label='min',
        markerfacecolor='green', markersize=6),
    ]
    plt.axhline(y = resolution, linestyle="--", color="black")

    plt.legend(handles=handles)
    plt.tight_layout()
    plt.show()

    for num_chr in range(1, 16+1):
        # print('chr'+str(num_chr))
        plt.figure()
        for taille in taille_list:
            plt.plot(range(epoch), croissant_theta_ref['chr'+str(num_chr)][taille], label=taille)
        plt.title('chr'+str(num_chr) + ' ref - croissant')
        plt.ylabel('|theta_ref - DNN(C_ref)|')
        plt.axhline(y=resolution)
        plt.legend()
        plt.tight_layout()
        plt.show()
        # plt.savefig(path+f"chr_{ctx.config['chr_id']}_{nb_train_dnn}_decroissant_ref_patch_size_{patch_size}_embed_dim_{embed_dim}_nb_blocks_{depth}_nb_heads_{num_heads}.png")
    
    df_min = []
    df_moy = []
    df_max = []
    for num_chr in range(1, 16+1):
        # print('chr'+str(num_chr))
        for taille in taille_list:
            values = croissant_theta_simu['chr'+str(num_chr)][taille][-50:]
            min_croissant_theta_simu = min(values)
            max_croissant_theta_simu = max(values)
            moy_croissant_theta_simu = sum(values)/len(values)
            df_min.append({'chr': f'chr{num_chr}',
                'taille': taille,
                'stat': 'min',
                'value': min_croissant_theta_simu.item()})
            df_moy.append({'chr': f'chr{num_chr}',
                'taille': taille,
                'stat': 'mean',
                'value': moy_croissant_theta_simu.item()})
            df_max.append({'chr': f'chr{num_chr}',
                'taille': taille,
                'stat': 'max',
                'value': max_croissant_theta_simu.item()})
    df_min = pd.DataFrame(df_min)
    df_moy = pd.DataFrame(df_moy)
    df_max = pd.DataFrame(df_max)
    df = [df_min, df_moy, df_max]
    
    color_min = [to_rgba('green') for _ in range(len(taille_list))]
    color_moy = [to_rgba('orange') for _ in range(len(taille_list))]
    color_max = [to_rgba('red') for _ in range(len(taille_list))]
    colors = [color_min, color_moy, color_max]
    plt.figure()
    for k in range(3):
        sns.stripplot(data=df[k],x='chr',y='value',hue='taille',dodge=True,jitter=False,size=6, palette=colors[k])
    plt.title(f'simu - Min-mean-max over 50 last epochs')
    plt.ylabel('|theta - DNN(C)|')
    handles = [
    Line2D([0], [0], marker='o', color='red', label='max',
        markerfacecolor='red', markersize=6),
    Line2D([0], [0], marker='o', color='orange', label='mean',
        markerfacecolor='orange', markersize=6),
    Line2D([0], [0], marker='o', color='green', label='min',
        markerfacecolor='green', markersize=6),
    ]
    plt.axhline(y = resolution, linestyle="--", color="black")

    plt.legend(handles=handles)
    plt.tight_layout()
    plt.show()

    for num_chr in range(1,16+1):
        # print('chr'+str(num_chr))
        plt.figure()
        for taille in taille_list:
            plt.plot(range(epoch), croissant_theta_simu['chr'+str(num_chr)][taille], label=taille)
        plt.title('chr'+str(num_chr) + ' simu - croissant')
        plt.ylabel('|theta - DNN(C)|')
        plt.axhline(y=resolution)
        plt.legend()
        plt.tight_layout()
        plt.show()
        # plt.savefig(path+f"chr_{ctx.config['chr_id']}_{nb_train_dnn}_decroissant_simu_patch_size_{patch_size}_embed_dim_{embed_dim}_nb_blocks_{depth}_nb_heads_{num_heads}.png")

    ##############################################

    # torch.save(DNN, path+f"chr_{chr_id}_dnn.pkl")
    # torch.save({'cnn': cnn_subnet.state_dict(),'mlps': mlp_subnet.state_dict()}, path + "dnn.pth")
            
    with open(path+f"{nb_train}_train_kernel_{kernel_size}_conv_{out_channels_per_layer[0]}_{out_channels_per_layer[1]}_theta_16.txt", "w") as f:
                f.write(f"converged after {epoch} epochs with a validation loss {val_loss_list[-1]} \n")
                theta_hat = torch.zeros(dim)
                for chr_id in range(1,17):
                    # print("chr", chr_id)
                    C_ref_row =  np.load(f"ref/{dim}_chr_{chr_id}_ref_{resolution}_norm_HiC_duan_intra_all.npy")
                        
                    #C_ref_row=np.delete(C_ref_row, range(start_bead['chr'+str(chr_id)], start_bead['chr'+str(chr_id)] + nb_bead['chr'+str(chr_id)]), axis =1) #remove intra
                    C_ref_row = torch.from_numpy(C_ref_row).float() 
                    C_ref_row = C_ref_row.unsqueeze(0)
                    features = cnn_subnet(C_ref_row.unsqueeze(0))          # → (B, F)
                    features = pool(features)
                    
                    # features = features.view(C_ref_row.size(0), -1)  # flatten
                                
                    #theta_hat[chr_id-1] = linear_subnets[chr_id-1](features).squeeze()[chr_id-1]  # → (B,)
                    theta_hat[chr_id-1] = mlp_subnet(features).squeeze()  # → (B,)

                f.write(f"theta_ref : {theta_ref}, DNN(C_ref) : {theta_hat*prior_range}\n")
                f.write(f"||theta_ref - DNN(C_ref)|| = {torch.sqrt(torch.mean((theta_hat*prior_range-theta_ref)**2))}")


if 0:
    nb_seq = 0
    nb_train_ABC= 1000

    sum_stat = 1

    if sum_stat:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        title = "theta_dnn"
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
        checkpoint = torch.load(path + "dnn_500_epochs.pth", map_location=device)

        # Load CNN weights
        cnn_subnet.load_state_dict(checkpoint['cnn'])

        # Load MLP weights
        for i, mlp in enumerate(linear_subnets):
            mlp.load_state_dict(checkpoint['mlps'][i])
    else:
        title = "theta_P_corr_inter"

    theta_hat = torch.zeros(dim)
    for chr_id in range(1,17):
        # print("chr", chr_id)
        C_ref_row =  np.load(f"ref/{dim}_chr_{chr_id}_ref_{resolution}_norm_HiC_duan_intra_all.npy")
            
        #C_ref_row=np.delete(C_ref_row, range(start_bead['chr'+str(chr_id)], start_bead['chr'+str(chr_id)] + nb_bead['chr'+str(chr_id)]), axis =1) #remove intra
        C_ref_row = torch.from_numpy(C_ref_row).float() 
        C_ref_row = C_ref_row.unsqueeze(0)
        features = cnn_subnet(C_ref_row.unsqueeze(0))          # → (B, F)
        
        features = features.view(C_ref_row.size(0), -1)  # flatten
                    
        #theta_hat[chr_id-1] = linear_subnets[chr_id-1](features).squeeze()[chr_id-1]  # → (B,)
        theta_hat[chr_id-1] = linear_subnets[chr_id-1](features).squeeze()  # → (B,)

        print(f"theta_ref : {theta_ref[chr_id-1]}, DNN(C_ref) : {theta_hat[chr_id-1].detach()*prior_range[chr_id-1]}")
        print(f"|theta_ref - DNN(C_ref)| = {torch.abs(theta_hat[chr_id-1]*prior_range[chr_id-1]-theta_ref[chr_id-1]).detach()}")
        print()
    print(f"||theta_ref - DNN(C_ref)||_2 = {torch.sqrt(torch.sum((theta_hat.detach()*prior_range-theta_ref)**2))}")
    ############################## SMCABC ####################################
    # print("ABC")

    #plot_C_genome(C_ref, resolution, 1, 100, chr_cen)

    if 0:
        ######## ABC round 0 ###############################
        
        theta_per_dim = dict.fromkeys(chr_seq_16_chr.keys())
        for chr_id in range(1,dim+1):
            theta_per_dim['chr'+str(chr_id)] = {}

        param = []
        for k in range(nb_train_ABC):
                print( k)
                ############# simulate theta ##########
                
                # theta_simu = theta_ref.clone()
                # theta_simu = prior_range/2
                # theta_simu[chr_id-1] = prior.sample()
                
                theta_simu = prior.sample()
                
                ####################################### 
                if sigma_spot=="variable":
                    sig_2_simu = random.uniform(0.1, 10)
                else:
                    sig_2_simu = 1.0
                intensity_simu, noisy = 100,noisy_ref
                param.append((sig_2_simu, intensity_simu))
                C_simu = simulator(theta_simu, resolution, sigma_spot, noisy)
                
                for chr_id in range(1,dim+1):
                    C_ref_row =  np.load(f"ref/{dim}_chr_{chr_id}_ref_{resolution}_norm_HiC_duan_intra_all.npy")
                    C_simu_row = C_simu[start_bead['chr'+str(chr_id)]:start_bead['chr'+str(chr_id)] + nb_bead['chr'+str(chr_id)], :]
                    
                    
                    if sum_stat:
                        
                        #print(DNN)
                        C_ref_row = torch.from_numpy(C_ref_row).float() 
                        theta_hat_C = DNN_eval(C_simu_row, chr_id, cnn_subnet, linear_subnets)
                        theta_hat_C_ref = DNN_eval(C_ref_row, chr_id, cnn_subnet, linear_subnets)
                
                        
                        theta_per_dim['chr'+str(chr_id)][theta_simu[chr_id-1]] = torch.mean((theta_hat_C-theta_hat_C_ref)**2).detach()
                    else:
                        theta_per_dim['chr'+str(chr_id)][theta_simu[chr_id-1]] = Correlation_inter_row_average(chr_seq, C_simu_row, C_ref_row, resolution, Pearson_correlation_vector)  
     
                #plot_C_genome(C_simu, resolution, sig_2_simu, intensity_simu, theta_simu)
        # if sum_stat:
        #     print(theta_ref[chr_id-1])       
        #     print(DNN(C_ref_row.unsqueeze(0))*prior_range[chr_id-1]) 

        # with open(f'{path}0_param', 'wb') as f:
        #     pickle.dump(param, f)
        # with open(f'{path}0_{title}', 'wb') as f:
        #     pickle.dump(theta_per_dim, f)

        ###### compute mean distance #########
        d_mean_list = np.zeros(100)
        thetas_accepted_per_dim = dict.fromkeys(chr_seq_16_chr.keys(), {})

        for j,prop in enumerate(np.linspace(25,1,100)):
            for chr_id in range(1,dim+1):
                if sum_stat==0:
                    theta_1_dim_sorted= dict(sorted(theta_per_dim['chr'+str(chr_id)].items(), key=lambda item: item[1], reverse=False)) #sort by values  
                else:  
                    theta_1_dim_sorted= dict(sorted(theta_per_dim['chr'+str(chr_id)].items(), key=lambda item: item[1], reverse=True)) #sort by values    
               
                start = int(len(theta_1_dim_sorted)*(1-prop/100))

                thetas_accepted_per_dim['chr'+str(chr_id)] = torch.hstack(list(dict(list(theta_1_dim_sorted.items())[start:]).keys())) #take theta:dist_dnn accepted 

                d_mean_list[j] += 1.0/dim*mean_distance_best_thetas(thetas_accepted_per_dim['chr'+str(chr_id)], theta_ref[chr_id-1])
        
        with open(path+f"0_mean_distance", 'wb') as f:
                pickle.dump(d_mean_list, f)
        ########################################

        ################## select good thetas ######################
        prop = 0.05
        start = int(nb_train_ABC*(1-prop))
        nb_accepted = int(nb_train_ABC*prop)

        if sum_stat:
            print("dnn")
            thetas_accepted_per_dim = dict.fromkeys(chr_seq_16_chr.keys(), {})
            for chr_id in range(1,dim+1):
                theta_1_dim_sorted= dict(sorted(theta_per_dim['chr'+str(chr_id)].items(), key=lambda item: item[1], reverse=True)) #sort by values
                thetas_accepted_per_dim['chr'+str(chr_id)] = torch.hstack(list(dict(list(theta_1_dim_sorted.items())[start:]).keys())) #take theta:corr_inter accepted 

        else:
            print("corr")
            thetas_accepted_per_dim = dict.fromkeys(chr_seq_16_chr.keys(), {})
            for chr_id in range(1,dim+1):
                theta_1_dim_sorted= dict(sorted(theta_per_dim['chr'+str(chr_id)].items(), key=lambda item: item[1], reverse=False)) #sort by values
                thetas_accepted_per_dim['chr'+str(chr_id)] = torch.hstack(list(dict(list(theta_1_dim_sorted.items())[start:]).keys())) #take theta:corr_inter accepted 
                # thetas_accepted_per_dim['chr'+str(chr_id)] = torch.stack(thetas_accepted_per_dim['chr'+str(chr_id)], dim=0)
        
        with open(f'{path}0_thetas_accepted', 'wb') as f:
                    pickle.dump(thetas_accepted_per_dim, f)
        ################################################################


        
    for nb_seq in range(nb_seq+1):
        print("sequential", nb_seq)
        # ############# load train set at time 0 ######################
        # with open(f'{path}{nb_seq}_{title}', 'rb') as f:
        #     theta_per_dim=pickle.load(f)
        # #########################################################

        ############# load accepted theta at time 0 ######################
        with open(f'{path}{nb_seq}_thetas_accepted', 'rb') as f:
            thetas_accepted_per_dim=pickle.load(f)
        prop = 0.05
        nb_accepted = int(nb_train_ABC*prop)
        # #########################################################

        
        ############## weights #######################
        weights = dict.fromkeys(chr_seq_16_chr.keys())
        for chr_id in range(1,dim+1):
            weights['chr'+str(chr_id)] = torch.ones(nb_accepted,dtype=torch.float64 )

        if nb_seq == 0:
            for chr_id in range(1,dim+1):
                weights['chr'+str(chr_id)] /= nb_accepted#uniform weights
            
        else:
            sigma = resolution
            with open(f'{path}{nb_seq-1}_thetas_accepted', 'rb') as f:
                thetas_t_1 = pickle.load(f)
            with open(f'{path}{nb_seq-1}_weights', 'rb') as f:
                weights_t_1 = pickle.load(f)

            #weights = torch.ones(nb_accepted)
            for chr_id in range(1,dim+1):
                #
                #log_weights = torch.ones(nb_accepted, dtype=torch.float64)
                #
                for i, theta_t in enumerate(thetas_accepted_per_dim['chr'+str(chr_id)]):
                    #
                    #log_distr = torch.zeros(len(thetas_t_1['chr'+str(chr_id)]))
                    #
                    denom = 0
                    for j, theta_t_1 in enumerate(thetas_t_1['chr'+str(chr_id)]):
                        #
                        #log_distr[j] = pdist.Normal(theta_t_1, sigma**2).log_prob(theta_t) #log(N(theta_{t-1}, sigma^2 Id))
                        #
                        #distr = pdist.MultivariateNormal(theta_t_1, sigma**2*torch.eye(dim)) #N(theta_{t-1}, sigma^2 Id)
                        distr = pdist.Normal(theta_t_1, sigma**2) #N(theta_{t-1}, sigma^2 Id)
                        perturb_kernel = torch.exp(distr.log_prob(theta_t)) 
                        weight_t_1 = weights_t_1['chr'+str(chr_id)][j] #w_{t-1} (theta_{t-1})
                        
                        denom += weight_t_1*perturb_kernel
                    #
                    #log_weight_t_1 = torch.log(weights_t_1['chr'+str(chr_id)]) #log_w_t_1
                    #log_denom = torch.logsumexp(log_weight_t_1 + log_distr, dim=0) # log sum exp (log_w_t_1 + log K)
                    #
                    num = 1.0
                    weights['chr'+str(chr_id)][i] = num/denom
                    #print(weights['chr'+str(chr_id)][i])
                    #
                    #log_weights[i] = -log_denom # w = 1/d, log(w) = - log(d)
                    #
                
                norm = weights['chr'+str(chr_id)].sum()
                weights['chr'+str(chr_id)] /= norm

                #
                #log_norm = torch.logsumexp(log_weights, dim=0) #log sum exp log(w)
                #log_weights -= log_norm #w = w / sum(w), log(w) = log(w) - log sum (w)
                #weights['chr'+str(chr_id)] = torch.exp(log_weights) #w = exp(log(w))
                #

        with open(f'{path}{nb_seq}_weights', 'wb') as f:
                pickle.dump(weights, f)
            
        ################################################

        ########### sample new thetas ##################
        theta = torch.zeros((nb_train_ABC,dim), dtype=int) #(1000, 16)

        for chr_id in range(1,dim+1):

            new_id = torch.multinomial(weights['chr'+str(chr_id)],nb_accepted, replacement=True) #sample from {thetas, weights}
            
            thetas_accepted = thetas_accepted_per_dim['chr'+str(chr_id)][new_id]
            
            sigma = resolution #perturb the thetas
            perturb_dist = pdist.Normal(torch.zeros(1), torch.ones(1)) #N(0,Id)
            nb_run = int(1 / prop)
            for k in range(1, nb_run+1): #create 1000 thetas from thetas accepted with perturbation
                perturb_eps = perturb_dist.sample((nb_accepted,))
              
                theta_proposal = (thetas_accepted+sigma*perturb_eps.squeeze(1)).int() #theta + sigma*N(0, Id)
              
                theta_out_prior = theta_proposal>prior_range[chr_id-1]  #check in prior
               
                theta_proposal[theta_out_prior]=thetas_accepted[theta_out_prior].int() #if out prior : take thetas accepted
                
                theta_out_prior = theta_proposal<1  #check in prior
               
                theta_proposal[theta_out_prior]=thetas_accepted[theta_out_prior].int() #if out prior : take thetas accepted
                
                theta[(k-1)*nb_accepted:k*nb_accepted, chr_id-1] = theta_proposal
            
        

        #################################################
    
        theta_per_dim = dict.fromkeys(chr_seq_16_chr.keys())
        for chr_id in range(1,dim+1):
            theta_per_dim['chr'+str(chr_id)] = {}

        param = []
        for k, theta_simu in enumerate(theta):
            print(k)
            #theta_simu = prior.sample()

            ##### {chr : theta} #####
            if sigma_spot=="variable":
                sig_2_simu = random.uniform(0.1, 10)
            else:
                sig_2_simu = 1.0
            intensity_simu, noisy = 100,noisy_ref
            param.append((sig_2_simu, intensity_simu))
            C_simu = simulator(theta_simu, resolution, sigma_spot, noisy)

            for chr_id in range(1,dim+1):
                C_ref_row =  np.load(f"ref/{dim}_chr_{chr_id}_ref_{resolution}_norm_HiC_duan_intra_all.npy")
                C_simu_row = C_simu[start_bead['chr'+str(chr_id)]:start_bead['chr'+str(chr_id)] + nb_bead['chr'+str(chr_id)], :]

                if sum_stat:

                    #print(DNN)
                    C_ref_row = torch.from_numpy(C_ref_row).float() 
                    theta_hat_C = DNN_eval(C_simu_row, chr_id, cnn_subnet, linear_subnets)
                    theta_hat_C_ref = DNN_eval(C_ref_row, chr_id, cnn_subnet, linear_subnets)
            
                    
                    theta_per_dim['chr'+str(chr_id)][theta_simu[chr_id-1]] = torch.mean((theta_hat_C-theta_hat_C_ref)**2).detach()
                else:
                    theta_per_dim['chr'+str(chr_id)][theta_simu[chr_id-1]] = Correlation_inter_row_average(chr_seq, C_simu_row, C_ref_row, resolution, Pearson_correlation_vector)  

                       
        # with open(f'{path}{nb_seq+1}_param', 'wb') as f:
        #             pickle.dump(param, f)
        # with open(f'{path}{nb_seq+1}_{title}', 'wb') as f:
        #             pickle.dump(theta_per_dim, f)

        ###### compute mean distance #########
        d_mean_list = np.zeros(100)
        thetas_accepted_per_dim = dict.fromkeys(chr_seq_16_chr.keys(), {})

        for j,prop in enumerate(np.linspace(25,1,100)):
            for chr_id in range(1,dim+1):
                if sum_stat==0:
                    theta_1_dim_sorted= dict(sorted(theta_per_dim['chr'+str(chr_id)].items(), key=lambda item: item[1], reverse=False)) #sort by values  
                else:  
                    theta_1_dim_sorted= dict(sorted(theta_per_dim['chr'+str(chr_id)].items(), key=lambda item: item[1], reverse=True)) #sort by values    
               
                start = int(len(theta_1_dim_sorted)*(1-prop/100))

                thetas_accepted_per_dim['chr'+str(chr_id)] = torch.hstack(list(dict(list(theta_1_dim_sorted.items())[start:]).keys())) #take theta:dist_dnn accepted 

                d_mean_list[j] += 1.0/dim*mean_distance_best_thetas(thetas_accepted_per_dim['chr'+str(chr_id)], theta_ref[chr_id-1])
        
        with open(path+f"{nb_seq+1}_mean_distance", 'wb') as f:
                pickle.dump(d_mean_list, f)
        ########################################

        ################## select good thetas ######################
        prop = 0.05
        start = int(nb_train_ABC*(1-prop))
        nb_accepted = int(nb_train_ABC*prop)

        if sum_stat:
            print("dnn")
            thetas_accepted_per_dim = dict.fromkeys(chr_seq_16_chr.keys(), {})
            for chr_id in range(1,dim+1):
                theta_1_dim_sorted= dict(sorted(theta_per_dim['chr'+str(chr_id)].items(), key=lambda item: item[1], reverse=True)) #sort by values
                thetas_accepted_per_dim['chr'+str(chr_id)] = torch.hstack(list(dict(list(theta_1_dim_sorted.items())[start:]).keys())) #take theta:corr_inter accepted 

        else:
            print("corr")
            thetas_accepted_per_dim = dict.fromkeys(chr_seq_16_chr.keys(), {})
            for chr_id in range(1,dim+1):
                theta_1_dim_sorted= dict(sorted(theta_per_dim['chr'+str(chr_id)].items(), key=lambda item: item[1], reverse=False)) #sort by values
                thetas_accepted_per_dim['chr'+str(chr_id)] = torch.hstack(list(dict(list(theta_1_dim_sorted.items())[start:]).keys())) #take theta:corr_inter accepted 
                # thetas_accepted_per_dim['chr'+str(chr_id)] = torch.stack(thetas_accepted_per_dim['chr'+str(chr_id)], dim=0)
        
        with open(f'{path}{nb_seq+1}_thetas_accepted', 'wb') as f:
                    pickle.dump(thetas_accepted_per_dim, f)
        ################################################################

                 