
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils import data
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import pyro.distributions as pdist

from sbi import utils as utils
#from sbi.neural_nets.embedding_nets import CNNEmbedding
from cnn import CNNEmbedding

import numpy as np
# from simulator import get_num_beads_and_start#, chr_seq_3_chr, chr_cen_3_chr, chr_seq_16_chr, chr_cen_16_chr

from itertools import combinations, product
import random
import pickle
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt

from transformer_row_centro import BioVisionTransformer
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from torch.utils.tensorboard import SummaryWriter


torch.manual_seed(0)


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

dim_ref = 3
sig_2_ref = 1

if dim_ref==3:
    chr_seq_ref = chr_seq_3_chr
    chr_cen_ref = chr_cen_3_chr
    prior_range_ref = torch.tensor([230209, 813179, 316619])
     
if dim_ref==16:
    chr_seq_ref = chr_seq_16_chr
    chr_cen_ref = chr_cen_16_chr
    prior_range_ref = torch.tensor([230209, 813179, 316619, 1531918, 
                                576869, 270148, 1090947, 562644,
                                439885, 745746, 666455, 1078176,
                                924430, 784334, 1091290, 948063])

nb_train_dnn = 500 #5000
nb_train_abc = 1000    
nb_bead_ref, start_bead_ref, nb_tot_bead_ref = get_num_beads_and_start(chr_seq_ref, resolution)
print(nb_bead_ref)
print(start_bead_ref)
if origin=="true":
    C_ref =  np.load(f"ref/{dim_ref}_chr_ref_{resolution}_norm_HiC_duan_intra_all.npy")
    C_ref = torch.from_numpy(C_ref).float() 
    


prior = utils.BoxUniform(torch.ones(1), prior_range_ref-1)

theta_ref = torch.tensor(list(chr_cen_ref.values()))


path = f"simulation_little_genome/{dim_ref}_chr/{origin}/res_{resolution}/noisy/sigma_{sigma_spot}/transformer/row/per_chr/3_pos_per_bloc/mlp/"



# def pad_matrices(matrices):
#     """ pad matrices to the biggest of the train set (add 0 at the bottom right)"""
#     max_h = max(m.size(-2) for m in matrices)
#     max_w = max(m.size(-1) for m in matrices)

#     padded = []
#     mask = []
#     for m in matrices:
        
#         h, w = m.size(-2), m.size(-1)
#         pad_h = max_h - h
#         pad_w = max_w - w
#         padded_m = F.pad(m, (0, pad_w, 0, pad_h))  # pad (left, right, top, bottom)
#         padded.append(padded_m.squeeze(0))
        
        
#         m_mask = torch.zeros(1, max_h, max_w, dtype=torch.bool, device=m.device)
#         m_mask[:, :h, :w] = 1
#         mask.append(m_mask)
    
#     return padded #, torch.stack(mask)



# def pad_input(patch_size, chr_seq, start_bead, nb_bead, C, chr_row):
     
#         """ pad a row of C (add 0 after each block of the row)"""

#         if nb_bead['chr'+str(chr_row)]%patch_size !=0:
#             padding_row = patch_size-nb_bead['chr'+str(chr_row)]%patch_size
            
#             row_to_insert = torch.zeros((padding_row, C.size(1)))
#             C = torch.cat((C, row_to_insert), dim=0) #insert bloc of rows


#         padding_cumul = 0
        
#         for i, chr in enumerate(list(chr_seq.keys())):
             
#             pos_bloc = start_bead[chr] + nb_bead[chr] + padding_cumul
          
#             if nb_bead[chr]%patch_size !=0:
#                 padding = patch_size-nb_bead[chr]%patch_size #nb of 0 to add in the row
#                 padding_cumul += padding
     
#                 col_to_insert = torch.zeros((C.size(0), padding))
                
#                 C = torch.cat((C[:, :pos_bloc], col_to_insert, C[:, pos_bloc:]), dim=1) #insert bloc of cols
        
#         return 

def pad_input_row(patch_size, C, chr_row):
        
            """ pad a row of C (B, C, H, W) (add 0 after each block of the row)"""
            
            if nb_bead_ref['chr'+str(chr_row)]%patch_size !=0:
                padding_row = patch_size-nb_bead_ref['chr'+str(chr_row)]%patch_size
                
                row_to_insert = torch.zeros((C.size(0), C.size(1), padding_row, C.size(3)))
                C = torch.cat((C, row_to_insert), dim=-2) #insert bloc of rows

            
            padding_cumul = 0
            
            for i, chr in enumerate(list(chr_seq_ref.keys())):
                
                pos_bloc = start_bead_ref[chr] + nb_bead_ref[chr] + padding_cumul
            
                if nb_bead_ref[chr]%patch_size !=0:
                    padding = patch_size-nb_bead_ref[chr]%patch_size #nb of 0 to add in the row
                    padding_cumul += padding
        
                    col_to_insert = torch.zeros((C.size(0), C.size(1), C.size(2), padding))
                    
                    C = torch.cat((C[:,:,:, :pos_bloc], col_to_insert, C[:,:,:, pos_bloc:]), dim=-1) #insert bloc of cols
            
            return C

def pad_input_C(patch_size, C):
        """ padd each block of C"""
       
        padding_cumul = 0
        
        for i, chr in enumerate(list(chr_seq_ref.keys())):
             
            pos_bloc = start_bead_ref[chr] + nb_bead_ref[chr] + padding_cumul
            
          
            if nb_bead_ref[chr]%patch_size !=0:
                padding = patch_size-nb_bead_ref[chr]%patch_size #nb of 0 to add in the row
                padding_cumul += padding
          
                row_to_insert = torch.zeros((padding, C.size(0)))

                C = torch.cat((C[:pos_bloc], row_to_insert, C[pos_bloc:]), dim=0)
                col_to_insert = torch.zeros((C.size(0), padding))
                
                C = torch.cat((C[:, :pos_bloc], col_to_insert, C[:, pos_bloc:]), dim=1)
    
        return C

# def create_genome(nb_chr):
#     chr_seq = {}
#     chr_cen = {}
#     for i in range(nb_chr):
#          length_chr = random.randint(200000, 2000000)
#          pos_centro = random.randint(100, length_chr-100)
#          chr_seq['chr'+str(i+1)] = length_chr
#          chr_cen['chr'+str(i+1)] = pos_centro
#     return chr_seq, chr_cen
          
########### simulate one row of C given a simulated genome chr_seq ###########

# def simulator_row(chr_seq, start_bead, nb_bead, nb_tot_bead, chr_row, theta, resolution, sigma_spot, noisy):
        
#     if sigma_spot=="variable":
#         sig_2_simu = random.uniform(0.5, 2)
#     else:
#         sig_2_simu = sig_2_ref
               
#     intensity_simu = 100

#     C_simu = torch.zeros((nb_bead['chr'+str(chr_row)],nb_tot_bead))

#     #for (chr_row, chr_col) in combinations(chr_seq.keys(),r=2):
#     n_row = nb_bead['chr'+str(chr_row)]
#     c_i_simu = theta[chr_row-1]//resolution

#     for chr_col in chr_seq.keys():
        
#         n_col = nb_bead[chr_col]
      
#         index_col = int(chr_col[chr_col.find("chr")+3:])-1
        
#         # c_i_simu = theta[index_row-13]//resolution
#         # c_j_simu = theta[index_col-13]//resolution
#         c_j_simu = theta[index_col]//resolution

#         def simulator_1_bloc(n_row, n_col, c_i, c_j, sig_2, intensity, noisy=noisy):
            
#             # Simulate a noisy matrix C_{n_row x n_col} with a gaussian spot at (c_i, c_j) of size sig_2 
            
#             C = torch.zeros((n_row, n_col))
            
#             # distr = MultivariateNormal(torch.tensor([c_i, c_j]), sig_2*torch.eye(2))
                
#             # indices = torch.tensor([[(i, j) for j in range(len(C[0]))] for i in range(len(C))])
#             iy, ix = torch.meshgrid(
#                 torch.arange(n_row, dtype=torch.float32),
#                 torch.arange(n_col, dtype=torch.float32),
#                 indexing="ij"
#             )
            
#             # C = intensity*torch.exp(distr.log_prob(indices))
            
#             C = intensity * torch.exp(-((iy - c_i)**2 + (ix - c_j)**2) / (2*sig_2))

#             if noisy:
#                 #mean = intensity * torch.exp(distr.log_prob(torch.tensor([c_i,c_j]))) * 0.1 / 2 
#                 mean = intensity * 0.1 / 2 
#                 sigma = intensity * 0.1 / 2
       
#                 noise = mean + sigma*torch.randn((n_row, n_col))

#                 sig = np.sqrt(sig_2)
                
#                 i0 = max(0, int(c_i) - int(sig))
#                 i1 = min(n_row, int(c_i) + int(sig) + 1)

#                 j0 = max(0, int(c_j) - int(sig))
#                 j1 = min(n_col, int(c_j) + int(sig) + 1)

#                 noise[i0:i1, :] = 0
#                 noise[:, j0:j1] = 0 

#             else:
#                 noise = torch.zeros_like(C)
            
#             return C+noise
        
#         #C_simu[start_bead[chr_row]:start_bead[chr_row]+nb_bead[chr_row]-1, start_bead[chr_col]:start_bead[chr_col]+nb_bead[chr_col]-1] = simulator_1_bloc(n_row,n_col, c_i_simu, c_j_simu, sig_2_simu, intensity_simu, noisy=noisy)
#         # print(C_simu[:, start_bead[chr_col]:start_bead[chr_col]+nb_bead[chr_col]].size() )
#         # print(simulator_1_bloc(n_row,n_col, c_i_simu, c_j_simu, sig_2_simu, intensity_simu, noisy=noisy).size())
#         C_simu[:, start_bead[chr_col]:start_bead[chr_col]+nb_bead[chr_col]] = simulator_1_bloc(n_row,n_col, c_i_simu, c_j_simu, sig_2_simu, intensity_simu, noisy=noisy)
#         C_simu[:, start_bead['chr'+str(chr_row)]:start_bead['chr'+str(chr_row)]+nb_bead['chr'+str(chr_row)]] = 0
#     return C_simu 
############################################################################

########## simulate variable size of row of C ##########       
# def get_simulations(patch_size):

#     sigma_spot = "variable"
#     noisy = 1

#     # genomes = []
#     thetas = []
#     # thetas_ref = []
#     C = []

#     for k in range(nb_train_dnn):
#         nb_chr = random.randint(2,20)
#         chr_seq, chr_cen = create_genome(nb_chr) #create a genome

#         nb_bead, start_bead, nb_tot_bead = get_num_beads_and_start(chr_seq, resolution) 

#         chr_row = random.randint(1, nb_chr) #choose the chr in row

#         prior_range = torch.tensor(list(chr_seq.values()))
#         prior = utils.BoxUniform(torch.ones(nb_chr), prior_range-1)


#         theta = prior.sample() #sample all centromeres
        
#         # C = torch.zeros(nb_train_dnn,C_ref.size(0))
#         # print(C.size())
       
#         C_tmp = simulator_row(chr_seq, start_bead, nb_bead, nb_tot_bead, chr_row, theta, resolution, sigma_spot, noisy)
   
#         # print(chr_seq, chr_cen)
#         # print(chr_row)
#         # plt.matshow(C_tmp)
#         # plt.show()

#         # print(C_tmp.size())
#         C_tmp = pad_input(patch_size, chr_seq, start_bead, nb_bead, C_tmp, chr_row) #pad to patch size
#         C_tmp = C_tmp.reshape(1,1,C_tmp.size(0), C_tmp.size(1))
#         # genomes.append(chr_seq)
#         # thetas_ref.append(chr_cen["chr"+str(chr_row)])

#         thetas.append(theta[chr_row-1]/prior_range[chr_row-1])

#         C.append(C_tmp)

#     #return genomes, thetas_ref, thetas, C
#     return thetas, C
###########################################################

# class VariableSizeDataset(Dataset):
#     def __init__(self, thetas, Cs):
#         self.thetas = thetas  # e.g. tensor [N, d] or list
#         self.Cs = Cs          # list of matrices of varying shape
    
#     def __len__(self):
#         return len(self.thetas)
    
#     def __getitem__(self, idx):
#         return self.thetas[idx], self.Cs[idx]
    
############### dataloaders for variable size of row ############### 
# def get_dataloaders(patch_size,
#         training_batch_size: int = 200,
#         validation_fraction: float = 0.1,
#     ):

#         #genomes, thetas_ref, theta, C  = get_simulations(patch_size)
#         theta, C  = get_simulations(patch_size) #list of theta (size 1) and row of C (various size)
#         C_padded = pad_matrices(C) #pad to the biggest matrices
#         theta = torch.stack(theta, dim=0)
#         C = torch.stack(C_padded, dim=0)
       
#         dataset = data.TensorDataset(theta, C)
        
#         # Get total number of training examples.
#         #num_examples = theta.size(0)
#         num_examples = len(dataset)
#         # Select random train and validation splits from (theta, x) pairs.
#         num_training_examples = int((1 - validation_fraction) * num_examples)
#         num_validation_examples = num_examples - num_training_examples

       
#         # Seperate indicies for training and validation
#         permuted_indices = torch.randperm(num_examples)
#         train_indices, val_indices = (
#             permuted_indices[:num_training_examples],
#             permuted_indices[num_training_examples:],
#         )

#         # Create training and validation loaders using a subset sampler.
#         # Intentionally use dicts to define the default dataloader args
#         # Then, use dataloader_kwargs to override (or add to) any of these defaults
#         # https://stackoverflow.com/questions/44784577/in-method-call-args-how-to-override-keyword-argument-of-unpacked-dict
#         train_loader_kwargs = {
#             "batch_size": min(training_batch_size, num_training_examples),
#             "drop_last": True,
#             "sampler": SubsetRandomSampler(train_indices.tolist()),
#             #"collate_fn": lambda batch: batch
#         }
#         val_loader_kwargs = {
#             "batch_size": min(training_batch_size, num_validation_examples),
#             "shuffle": False,
#             "drop_last": True,
#             "sampler": SubsetRandomSampler(val_indices.tolist()),
#             #"collate_fn": lambda batch: batch
#         }
        
#         train_loader = data.DataLoader(dataset, **train_loader_kwargs)
#         val_loader = data.DataLoader(dataset, **val_loader_kwargs)

#         return train_loader, val_loader
##############################################################################

def simulator(theta, resolution, sigma_spot, noisy):
        
    if sigma_spot=="variable":
        sig_2_simu = random.uniform(0.1, 10)
    else:
        sig_2_simu = sig_2_ref
               
    intensity_simu = 100

    C_simu = torch.zeros((nb_tot_bead_ref,nb_tot_bead_ref))

    for (chr_row, chr_col) in combinations(chr_seq_ref.keys(),r=2):
        
        n_row = chr_seq_ref[chr_row]//resolution
        n_col = chr_seq_ref[chr_col]//resolution
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
        
        C_simu[start_bead_ref[chr_row]:start_bead_ref[chr_row]+nb_bead_ref[chr_row]-1, start_bead_ref[chr_col]:start_bead_ref[chr_col]+nb_bead_ref[chr_col]-1] = simulator_1_bloc(n_row,n_col, c_i_simu, c_j_simu, sig_2_simu, intensity_simu, noisy=noisy)
    
    return C_simu + torch.transpose(C_simu, 0,1)

def get_simulations_C(nb_train_dnn, patch_size):

    theta = prior.sample((nb_train_dnn,))
    
    C = torch.zeros(nb_train_dnn,1,C_ref.size(0), C_ref.size(1))
  
    for k in range(nb_train_dnn):

        C_tmp = simulator(theta[k], resolution, sigma_spot, noisy_ref)  
        # C_tmp = pad_input_C(patch_size, C_tmp)
        # plt.matshow(C_tmp)
        # plt.show()
        C_tmp = C_tmp.reshape(1,1,C_tmp.size(0), C_tmp.size(1))

        
        #C_tmp=np.delete(C_tmp, range(start_bead['chr'+str(chr_id)], start_bead['chr'+str(chr_id)] + nb_bead['chr'+str(chr_id)]), axis =1)

        # C_tmp = torch.from_numpy(C_tmp).float() 

        C[k] = C_tmp

    #theta = theta[:,chr_id-1]/prior_range[chr_id-1] #norm thetas
    theta = theta/prior_range_ref #norm thetas
    
    return theta, C

def get_dataloaders(nb_train_dnn, patch_size,
        training_batch_size: int = 200,
        validation_fraction: float = 0.1,
    ):

        theta, C  = get_simulations_C(nb_train_dnn, patch_size)
        train_loader = {}
        val_loader = {}
        for chr_id in range(1,dim_ref+1):
            # fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            C_row = C[:,:, start_bead_ref['chr'+str(chr_id)]:start_bead_ref['chr'+str(chr_id)] + nb_bead_ref['chr'+str(chr_id)], :]
            C_row = pad_input_row(patch_size, C_row,chr_id)
            
            # left = C[:,:, start_bead_ref['chr'+str(chr_id)]:start_bead_ref['chr'+str(chr_id)] + nb_bead_ref['chr'+str(chr_id)], :start_bead_ref['chr'+str(chr_id)]]
            # right = C[:,:, start_bead_ref['chr'+str(chr_id)]:start_bead_ref['chr'+str(chr_id)] + nb_bead_ref['chr'+str(chr_id)], start_bead_ref['chr'+str(chr_id)] + nb_bead_ref['chr'+str(chr_id)]:]
            # cut_intra = torch.cat([left, right], dim=3)
            # print(C_row.size())
            # print(left.size())
            # print(right.size())
            # axes[0].matshow(C_row[0,0,:,:])
            # # axes[1].matshow(left[0,0,:,:])
            # # axes[2].matshow(right[0,0,:,:])
            # axes[1].matshow(cut_intra[0,0,:,:])
            # plt.show()
            
            dataset = data.TensorDataset(theta[:,chr_id-1], C_row) #out 1 chr
            # dataset = data.TensorDataset(theta[:,chr_id-1], cut_intra) #out 1 chr
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

def loss(theta, C, DNN):
    
    #print(DNN(C))
    theta_hat = DNN(C, nb_bead_ref).squeeze(-1)
    # print(theta_hat.size())
    
    loss = (theta_hat - theta) ** 2  

    return loss

def shuffled_batches(trainloaders):
    # Build (loader_name, batch_idx) pairs
    batch_refs = []
    for name, loader in trainloaders.items():
        n_batches = len(loader)   # number of batches in this loader
        for i in range(n_batches):
            batch_refs.append((name, i)) #numerote chr + index batch
    
    # Shuffle the references
    order = torch.randperm(len(batch_refs)) #shuffle index batch
    shuffled_refs = [batch_refs[i] for i in order]
    

    # Create iterators for each loader
    iters = {name: iter(loader) for name, loader in trainloaders.items()} #{chr : iterator over train loader}
    
    # Yield batches in shuffled order
    for name, batch_index in shuffled_refs:
        yield name, batch_index, next(iters[name]) #retourne chr, batch index et le batch suivant

# if origin=="true":
#     C_ref =  np.load(f"ref/{dim_ref}_chr_{1}_ref_{resolution}_norm_HiC_duan_intra_all.npy")
#     C_ref = torch.from_numpy(C_ref).float() 
# else:
#     C_ref = simulator(list(chr_cen_ref.values()), resolution, sigma_spot, noisy_ref)

# print("orginal size", origin, C_ref.size())
#C_ref = C_ref + torch.triu(C_ref).transpose(1,0)



##################### TRAINING SCHEDULER #################################
if 1:
    # writer = SummaryWriter(log_dir=path)
    patch_size = 4
    
    # C_ref = pad_input_C(patch_size, C_ref) # to give a size for the train set
    # C_ref = C_ref.reshape(1,1,C_ref.size(0), C_ref.size(1))


    
    # C_ref = pad_input(patch_size, chr_seq_ref, start_bead_ref, nb_bead_ref, C_ref, 1)
    # print('padded size', C_ref.size())
    # # plt.matshow(C_ref)
    # # plt.show()
    # C_ref = C_ref.reshape(1,1,C_ref.size(0), C_ref.size(1))

    embed_dim = 24 #4*patch_size*patch_size
    depth = 4
    num_heads = 4

    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DNN = BioVisionTransformer(in_chans = 1, patch_size=patch_size, embed_dim = embed_dim,no_embed_class=True, device=device, depth = depth, num_heads=num_heads)#, chr_seq=chr_seq_ref, chr_cen=chr_cen_ref, dim=dim_ref, resolution=resolution)
    print(DNN)

    # for name, param in DNN.named_parameters():
    #     print(f"{name}: {param.shape}")
   
    

    learning_rate = 5e-4 #scheduler
    max_num_epochs= 100
    
    print("data loader")
    train_loader, val_loader = get_dataloaders(nb_train_dnn, patch_size)
    print("train loader and val loader done") 
    
    num_training_steps = len(train_loader) * max_num_epochs
    
    # num_warmup_steps = int(0.1*num_training_steps)
    
    # with open(path+"train_loader", "wb") as f:
    #      pickle.dump(train_loader, f)
    # with open(path+"val_loader", "wb") as f:
    #      pickle.dump(val_loader, f)
    # print("loading data loader")
    # with open(path+"train_loader", "rb") as f:
    #      train_loader = pickle.load(f)
    # with open(path+"val_loader", "rb") as f:
    #      val_loader = pickle.load(f)   

    # Move entire net to device for training.
    DNN.to(device)
  
    

    optimizer = torch.optim.AdamW(list(DNN.parameters()), lr=learning_rate)

    # scheduler = get_linear_schedule_with_warmup(
    # optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    # )

    # scheduler = get_cosine_schedule_with_warmup(
    # optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    # )
    
    # # see learning rate
    # lrs = []
    # for step in range(num_training_steps):
    #     scheduler.step()
    #     lrs.append(scheduler.get_last_lr()[0])

    # import matplotlib.pyplot as plt
    # plt.plot(lrs)
    # plt.title("Learning Rate Schedule")
    # plt.xlabel("Step")
    # plt.ylabel("LR")
    # plt.show()

    epoch = 0
    train_loss_list = []
    val_loss_list = []
    loss_theta_ref = []
    distance_to_theta_train = {key : [] for key in chr_seq_ref.keys()}
    distance_to_theta_val = {key : [] for key in chr_seq_ref.keys()}
    distance_to_theta = {key : [] for key in chr_seq_ref.keys()}
    distance_simu_to_theta = {key : [] for key in chr_seq_ref.keys()}



    while epoch <= max_num_epochs: 
        #iter_loader = {name: iter(loader) for name, loader in train_loader.items()}

        print("epoch", epoch)
        # Train for a single epoch.
        # DNN.train()
        train_loss_sum = 0
        val_loss_sum = 0
            
        # chr_range = torch.randperm(dim_ref) + 1
        # #print(chr_range)
        # for chr_id in chr_range:
        # # for chr_id in range(1, dim_ref+1):
        # # for chr_id in range(dim_ref,0, -1):
        #     chr_id = chr_id.item()
            #print(chr_id)
            # Train for a single epoch.
            # DNN.train()
            
            
            # for batch in train_loader['chr'+str(chr_id)]:
        # for name, batch_index, batch in shuffled_batches(train_loader):
        #     print(name, batch_index)
        print()
        
        
        # for k in range(sum(len(loader) for loader in train_loader.values())):
        #     chr_id = torch.randint(1, dim_ref+1, (1,)).item()
        #     loader_name = 'chr'+str(chr_id)
        #     try:
        #         batch = next(iter_loader[loader_name])
        #     except StopIteration:
        #         # Iterator exhausted, recreate it
        #         iter_loader[loader_name] = iter(train_loader[loader_name])
        #         batch = next(iter_loader[loader_name])
        #     #     print(batch)
        #     print(chr_id)
        print('train')
        estimation_train = 0
        for batch in train_loader['chr1']:
            
            optimizer.zero_grad()
            # Get batches on current device.
            theta_batch, x_batch= (
                batch[0].to(device),
                batch[1].to(device),
            )
            # print("batch", x_batch.size())
            # plt.matshow(x_batch[0,0,:,:])
            # plt.show()
            theta_hat = DNN(x_batch, nb_bead_ref)*prior_range_ref[0]  
            estimation_train += torch.mean(torch.abs(theta_hat.squeeze(-1)-theta_batch))
            
            train_losses = loss(theta_batch,x_batch, DNN) #losses over batch i
            
            train_loss = torch.mean(train_losses) #mean of losses over batch i
            train_loss_sum += train_losses.sum().item() #sum losses over all item of all batches

            train_loss.backward()
            
            optimizer.step()
                # scheduler.step()

        
    

            # Calculate validation performance.
            # DNN.eval()
            # val_loss_sum = 0
        distance_to_theta_train['chr1'].append(estimation_train*1.0/len(train_loader['chr1']))

        print("eval")
        with torch.no_grad():
            # for chr_id in range(1,dim_ref+1):
            #     print(chr_id)
            estimation_val = 0
            for batch in val_loader['chr1']:
                    theta_batch, x_batch = (
                        batch[0].to(device),
                        batch[1].to(device),
                        
                    )
                    # theta_list, C_list = zip(*batch)  # unzip the batch
                
                    # theta_batch = torch.stack(theta_list).to(device)
                    # x_batch = [c.to(device) for c in C_list]
                    # Take negative loss here to get validation log_prob.
                    theta_hat = DNN(x_batch, nb_bead_ref)*prior_range_ref[0]  
                    estimation_val += torch.mean(torch.abs(theta_hat.squeeze(-1)-theta_batch))

                    val_losses = loss(theta_batch,x_batch, DNN)
                    val_loss_sum += val_losses.sum().item()

        distance_to_theta_val['chr1'].append(estimation_val*1.0/len(val_loader['chr1']))
        
        distance = 0
        for chr_id in range(1,2):
            #C_ref =  np.load(f"ref/{dim_ref}_chr_{num_chr}_ref_{resolution}_norm_HiC_duan_intra_all.npy")
            # C_ref =  np.load(f"ref/{dim}_chr_end_ref_{resolution}_norm_HiC_duan_intra_all.npy")
            #C_ref = torch.from_numpy(C_ref).float() 
            
            C_ref_row = C_ref[start_bead_ref['chr'+str(chr_id)]:start_bead_ref['chr'+str(chr_id)] + nb_bead_ref['chr'+str(chr_id)], :]
            C_ref_row = C_ref_row.reshape(1,1,C_ref_row.size(0), C_ref_row.size(1))
                
            C_ref_row = pad_input_row(patch_size, C_ref_row, num_chr)


            
            theta_ref = chr_cen_ref['chr'+str(chr_id)]

            theta_hat = DNN(C_ref_row, nb_bead_ref).item()*prior_range_ref[chr_id-1]

            distance_to_theta['chr'+str(chr_id)].append(torch.abs(theta_hat-theta_ref))
            distance += (theta_hat -theta_ref)**2

            distance_simu = 0
            for k in range(10):
                
                theta = prior.sample()

                C_tmp = simulator(theta, resolution, sigma_spot, noisy_ref)  
                C_tmp = C_tmp.reshape(1,1,C_tmp.size(0), C_tmp.size(1))
                C_tmp_row = C_tmp[:,:, start_bead_ref['chr'+str(chr_id)]:start_bead_ref['chr'+str(chr_id)] + nb_bead_ref['chr'+str(chr_id)], :]
                C_tmp_row = pad_input_row(patch_size, C_tmp_row, chr_id)
                
                theta_hat = DNN(C_tmp_row, nb_bead_ref).item()*prior_range_ref[chr_id-1]
                distance_simu+=torch.abs(theta_hat-theta)*0.1
            distance_simu_to_theta['chr'+str(chr_id)].append(distance_simu)

        l2_norm.append(torch.sqrt(distance).item())

        epoch += 1

        train_loss_average = train_loss_sum / (
            len(train_loader['chr1']) * train_loader['chr1'].batch_size  # type: ignore
        )
        # train_loss_average = train_loss_sum / (
        #     len(train_loader) * train_loader.batch_size  # type: ignore
        # )
        # train_loss_average = train_loss_sum / (
        #         len(train_loader)*len(train_loader['chr'+str(chr_id)]) * train_loader['chr'+str(chr_id)].batch_size  # type: ignore
        #     ) #average loss over all items of all batchs of all chr
        # Take mean over all validation samples.
        val_loss = val_loss_sum / (
            len(val_loader['chr1']) * val_loader['chr1'].batch_size  # type: ignore
        )
        # val_loss = val_loss_sum / (
        #     len(val_loader) * len(val_loader['chr'+str(chr_id)]) * val_loader['chr'+str(chr_id)].batch_size  # type: ignore
        # )
        train_loss_list.append(train_loss_average)
        val_loss_list.append(val_loss)
    ## to follow parameters
    #     for name, param in DNN.named_parameters():
    #         writer.add_histogram(f"{name}.weights", param, epoch)
    #         if param.grad is not None:
    #             writer.add_histogram(f"{name}.grads", param.grad, epoch)

    #     writer.add_scalar("Loss/train", train_loss_average, epoch)
    # writer.close()

    # Avoid keeping the gradients in the resulting network, which can
    # cause memory leakage when benchmarking.
    DNN.zero_grad(set_to_none=True)
    
    print(f"converged after {epoch} epochs with a validation loss {val_loss_list[-1]}")
    plt.figure()
    plt.plot(range(epoch), torch.sqrt(torch.tensor(train_loss_list)), label="train")
    plt.plot(range(epoch), torch.sqrt(torch.tensor(val_loss_list)), label="val")
    plt.legend()
    # plt.show()
    plt.savefig(path+f"{nb_train_dnn}_loss_patch_size_{patch_size}_embed_dim_{embed_dim}_nb_blocks_{depth}_nb_heads_{num_heads}.png")
        #plt.savefig(path+f"{nb_train}_loss_linear_{decay_linear}.png")
    plt.figure()
    plt.plot(range(epoch), l2_norm)
    plt.ylabel('||theta_ref - DNN(C_ref)||_2')
    plt.title(f'depth_{depth}_nb_head_{num_heads}')
    plt.axhline(y=resolution)
    plt.savefig(path+f"{nb_train_dnn}_l2_norm_patch_size_{patch_size}_embed_dim_{embed_dim}_nb_blocks_{depth}_nb_heads_{num_heads}.png")

    for num_chr in range(1,2):
        plt.figure()
        plt.plot(range(epoch), distance_to_theta['chr'+str(num_chr)])
        plt.title('chr'+str(num_chr) + f'depth_{depth}_nb_head_{num_heads}')
        plt.ylabel('|theta_ref - DNN(C_ref)|')
        plt.axhline(y=resolution)
        plt.savefig(path+f"{nb_train_dnn}_distance_to_theta_chr_{num_chr}_patch_size_{patch_size}_embed_dim_{embed_dim}_nb_blocks_{depth}_nb_heads_{num_heads}.png")

    for num_chr in range(1,2):
        plt.figure()
        plt.plot(range(epoch), distance_simu_to_theta['chr'+str(num_chr)])
        plt.title('chr'+str(num_chr) + f'depth_{depth}_nb_head_{num_heads}')
        plt.ylabel('|theta_ref - DNN(C_ref)|')
        plt.axhline(y=resolution)
        plt.savefig(path+f"{nb_train_dnn}_distance_simu_to_theta_chr_{num_chr}_patch_size_{patch_size}_embed_dim_{embed_dim}_nb_blocks_{depth}_nb_heads_{num_heads}.png")

    for num_chr in range(1,2):
        plt.figure()
        plt.plot(range(epoch), distance_to_theta_train['chr'+str(num_chr)], label='train')
        plt.plot(range(epoch), distance_to_theta_val['chr'+str(num_chr)], label='val')
        plt.title('chr'+str(num_chr) + f'depth_{depth}_nb_head_{num_heads}')
        plt.ylabel('|theta_ref - DNN(C_ref)|')
        plt.axhline(y=resolution)
        plt.legend()
        plt.savefig(path+f"{nb_train_dnn}_distance_train_val_to_theta_chr_{num_chr}_patch_size_{patch_size}_embed_dim_{embed_dim}_nb_blocks_{depth}_nb_heads_{num_heads}.png")

    torch.save(DNN, path+f"dnn_depth_{depth}_num_heads_{num_heads}.pkl")
    with open(path+f"convergence_info_{nb_train_dnn}_patch_size_{patch_size}_embed_dim_{embed_dim}_nb_blocks_{depth}_nb_heads_{num_heads}.txt", "w") as f:
        #with open(path+f"convergence_info_{nb_train}_linear_{decay_linear}.txt", "w") as f:
            f.write(f"converged after {epoch} epochs with a validation loss {val_loss_list[-1]} \n")
            f.write(f"theta_ref : {theta_ref}, DNN(C_ref) : {DNN(C_ref, nb_bead_ref)*prior_range}\n")
            f.write(f"||theta_ref - DNN(C_ref)|| = {torch.sqrt(torch.mean((DNN(C_ref, nb_bead_ref)*prior_range-theta_ref)**2))}")

##################### TRAINING NO SCHEDULER #################################
if 0:
    patch_size = 4

    C_ref = C_ref + torch.triu(C_ref).transpose(1,0)
    C_ref = pad_input(patch_size, C_ref)
    print('padded size', C_ref.size())
    # plt.matshow(C_ref)
    # plt.show()
    C_ref = C_ref.reshape(1,1,C_ref.size(0), C_ref.size(1))

    embed_dim = 4*patch_size*patch_size
    depth = 4
    num_heads = 16

    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DNN = BioVisionTransformer(img_size=C_ref.size(2), in_chans = 1, patch_size=patch_size, embed_dim = embed_dim,no_embed_class=True, device=device, depth = depth, num_heads=num_heads, chr_seq=chr_seq_16_chr, chr_cen=chr_cen_16_chr, dim=dim, resolution=resolution)
    print(DNN)
    
    stop_after_epochs = 20
    

    learning_rate = 5e-4 #scheduler
    #max_num_epochs= 2**31 - 1
    max_num_epochs= 100
    print("data loader")
    train_loader, val_loader = get_dataloaders(patch_size)
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

    # Move entire net to device for training.
    DNN.to(device)
  
    

    optimizer = torch.optim.Adam(list(DNN.parameters()), lr=learning_rate)
    epoch, epoch_since_last_improvement, val_loss, best_val_loss = 0, 0, float("Inf"), float("Inf")
    train_loss_list = []
    val_loss_list = []

    #while epoch <= max_num_epochs and not converged(epoch_since_last_improvement, stop_after_epochs): #changer critere validation
    while epoch <= max_num_epochs:
        print("epoch", epoch)
        # Train for a single epoch.
        DNN.train()
        train_loss_sum = 0
        
        for batch in train_loader:
            
            optimizer.zero_grad()
            # Get batches on current device.
            theta_batch, x_batch,  = (
                batch[0].to(device),
                batch[1].to(device)
            )
            
            
            train_losses = loss(theta_batch,x_batch, DNN)
            
            train_loss = torch.mean(train_losses)
            train_loss_sum += train_losses.sum().item()

            train_loss.backward()
            
            optimizer.step()
         
        epoch += 1

        train_loss_average = train_loss_sum / (
            len(train_loader) * train_loader.batch_size  # type: ignore
        )
        train_loss_list.append(train_loss_average)

        # Calculate validation performance.
        DNN.eval()
        val_loss_sum = 0

        with torch.no_grad():
            for batch in val_loader:
                theta_batch, x_batch = (
                    batch[0].to(device),
                    batch[1].to(device),
                    
                )
                # Take negative loss here to get validation log_prob.
                val_losses = loss(theta_batch,x_batch, DNN)
                val_loss_sum += val_losses.sum().item()

        # Take mean over all validation samples.
        val_loss = val_loss_sum / (
            len(val_loader) * val_loader.batch_size  # type: ignore
        )
        if len(val_loss_list) > 0 and val_loss > val_loss_list[-1]:#start overfit
            epoch_since_last_improvement +=1

        val_loss_list.append(val_loss)


    # Avoid keeping the gradients in the resulting network, which can
    # cause memory leakage when benchmarking.
    DNN.zero_grad(set_to_none=True)
        
    print(f"converged after {epoch} epochs with a validation loss {val_loss_list[-1]}")

    plt.plot(range(epoch), torch.sqrt(torch.tensor(train_loss_list)), label="train")
    plt.plot(range(epoch), torch.sqrt(torch.tensor(val_loss_list)), label="val")
    plt.legend()
    # plt.show()
    plt.savefig(path+f"{nb_train_dnn}_loss_patch_size_{patch_size}_embed_dim_{embed_dim}_nb_blocks_{depth}_nb_heads_{num_heads}.png")
        #plt.savefig(path+f"{nb_train}_loss_linear_{decay_linear}.png")
        
    torch.save(DNN, path+"dnn.pkl")
    with open(path+f"convergence_info_{nb_train_dnn}_patch_size_{patch_size}_embed_dim_{embed_dim}_nb_blocks_{depth}_nb_heads_{num_heads}.txt", "w") as f:
        #with open(path+f"convergence_info_{nb_train}_linear_{decay_linear}.txt", "w") as f:
            f.write(f"converged after {epoch} epochs with a validation loss {val_loss_list[-1]} \n")
            f.write(f"theta_ref : {theta_ref}, DNN(C_ref) : {DNN(C_ref)*prior_range}\n")
            f.write(f"||theta_ref - DNN(C_ref)|| = {torch.sqrt(torch.mean((DNN(C_ref)*prior_range-theta_ref)**2))}")


if 1:
    device = torch.device('cpu')
    patch_size = 4

    # C_ref = C_ref + torch.triu(C_ref).transpose(1,0)
    # C_ref = pad_input(patch_size, C_ref)
    # C_ref = C_ref.reshape(1,1,C_ref.size(0), C_ref.size(1))
    # img =  np.load(f"ref/{dim_ref}_chr_{1}_ref_{resolution}_norm_HiC_duan_intra_all.npy")
    # img = torch.from_numpy(img).float() 
    # img = pad_input(patch_size, chr_seq_ref, start_bead_ref, nb_bead_ref, img, 1)
    # path = f"simulation_little_genome/{16}_chr/{origin}/res_{resolution}/noisy/sigma_{sigma_spot}/transformer/row/"
    
    embed_dim = 4*patch_size*patch_size
    for (depth, num_heads) in product([4,8], [4,8,16,64]):
         
    
        print("#### depth", depth, "nb_heads", num_heads, "####")
        print()
        state_dict = torch.load(path+f"dnn_state_depth_{depth}_nb_heads_{num_heads}.pkl", map_location="cpu")

        DNN = BioVisionTransformer(in_chans = 1, patch_size=patch_size, embed_dim = embed_dim,no_embed_class=True, device=device, depth = depth, num_heads=num_heads)#, chr_seq=chr_seq_ref, chr_cen=chr_cen_ref, dim=dim_ref, resolution=resolution)
        
        DNN.load_state_dict(state_dict)
        DNN.to("cpu")

        
        # DNN = torch.load(path+"dnn.pkl", map_location=device, weights_only=False)

        # print(DNN)
        distance = 0
        for num_chr in range(1,dim_ref +1):
            C_ref =  np.load(f"ref/{dim_ref}_chr_{num_chr}_ref_{resolution}_norm_HiC_duan_intra_all.npy")
            # C_ref =  np.load(f"ref/{dim}_chr_end_ref_{resolution}_norm_HiC_duan_intra_all.npy")
            C_ref = torch.from_numpy(C_ref).float() 

            C_ref = C_ref.reshape(1,1,C_ref.size(0), C_ref.size(1))

            theta_ref = chr_cen_ref['chr'+str(num_chr)]
            theta_hat = DNN(C_ref).item()*prior_range_ref[num_chr-1]
            distance += (theta_hat -theta_ref)**2
            print(f"theta_ref : {theta_ref}, DNN(C_ref) : {theta_hat}")
            print(f"|theta_ref - DNN(C_ref)| = {torch.abs(theta_hat-theta_ref)}")
            print()
        print("||theta_ref - DNN(C_ref)||_2 = ", torch.sqrt(distance).item())
    nb_seq = 0
    ############################## SMCABC -- P. vector based correlation upper all ####################################
    print("ABC")


    #plot_C_genome(C_ref, resolution, 1, 100, chr_cen)
    
    

    if 0:
        ######## ABC round 0 ###############################

        theta_dnn = {}
        param = []
        for k in range(nb_train_ABC):
                print(k)
                ############# simulate theta ##########
                
                theta_simu = prior.sample()
            
                ####################################### 
                if sigma_spot=="variable":
                    sig_2_simu = random.uniform(0.1, 10)
                # sig_2_simu = 0.5
                intensity_simu, noisy = 100,noisy_ref
                param.append((sig_2_simu, intensity_simu))
                C_simu = simulator(theta_simu, resolution, sigma_spot, noisy)

                #C_simu = C_simu + torch.triu(C_simu).transpose(1,0)
                # C_simu = vectorise_upper_matrix(C_simu, resolution)

                C_simu = C_simu + torch.triu(C_simu).transpose(1,0)
                # print(C_simu.size())
                C_simu = pad_input(patch_size, C_simu)
                C_simu = C_simu.reshape(1,1,C_simu.size(0), C_simu.size(1))
                
                
                theta_dnn[theta_simu] = torch.mean((DNN(C_simu)-DNN(C_ref))**2).detach()

                #plot_C_genome(C_simu, resolution, sig_2_simu, intensity_simu, theta_simu)
        print(theta_ref)       
        #print(DNN(C_ref.unsqueeze(0))) 
        print(DNN(C_ref)*prior_range)
        print(torch.sqrt(torch.mean((DNN(C_ref)*prior_range-theta_ref)**2))) 

        with open(f'{path}/0_param', 'wb') as f:
            pickle.dump(param, f)

        with open(f'{path}/0_theta_dnn', 'wb') as f:
            pickle.dump(theta_dnn, f)
        
        

    for nb_seq in range(nb_seq+1):
        print("sequential", nb_seq)
        ############# load train set at time 0 ######################
        with open(f'{path}{nb_seq}_theta_dnn', 'rb') as f:
                theta_dnn = pickle.load(f)
        #########################################################

        ################## select good thetas ######################
        prop = 0.05
        start = int(len(theta_dnn)*(1-prop))
        theta_dnn_sorted= dict(sorted(theta_dnn.items(), key=lambda item: item[1], reverse=True)) #sort by values
        thetas_accepted = list(dict(list(theta_dnn_sorted.items())[start:]).keys()) #take theta:corr_inter accepted 
        thetas_accepted = torch.stack(thetas_accepted, dim=0)
        
        with open(f'{path}{nb_seq}_thetas_accepted', 'wb') as f:
                    pickle.dump(thetas_accepted, f)
        
        ################################################################
        ############## weights #######################
        if nb_seq == 0:
            weights = torch.ones(len(thetas_accepted))*1.0/len(thetas_accepted) #uniform weights
        
        else:
            sigma = resolution
            with open(f'{path}{nb_seq-1}_thetas_accepted', 'rb') as f:
                thetas_t_1 = pickle.load(f)
            with open(f'{path}{nb_seq-1}_weights', 'rb') as f:
                weights_t_1 = pickle.load(f)

            weights = torch.ones(len(thetas_accepted))
            for i, theta_t in enumerate(thetas_accepted):
                denom = 0
                for j, theta_t_1 in enumerate(thetas_t_1):
                    distr = pdist.MultivariateNormal(theta_t_1, sigma**2*torch.eye(dim)) #N(theta_{t-1}, sigma^2 Id)
                    perturb_kernel = torch.exp(distr.log_prob(theta_t)) 
                    weight_t_1 = weights_t_1[j] #w_{t-1} (theta_{t-1})
                    denom += weight_t_1*perturb_kernel

                num = 1.0
                weights[i] = num/denom
            norm = weights.sum()
            weights /= norm

        with open(f'{path}{nb_seq}_weights', 'wb') as f:
                pickle.dump(weights, f)
            
        ################################################

        ########### sample new thetas ##################
        new_id = torch.multinomial(weights,len(thetas_accepted), replacement=True) #sample from {thetas, weights}
        thetas_accepted = thetas_accepted[new_id]

        sigma = resolution #perturb the thetas
        perturb_dist = pdist.MultivariateNormal(torch.zeros(dim), torch.eye(dim)) #N(0,Id)
        nb_run = int(1 / prop)
        theta = torch.zeros((nb_train_ABC, dim), dtype=int) #(1000, 3)
        
        for k in range(1, nb_run+1): #create 1000 thetas from thetas accepted with perturbation
            perturb_eps = perturb_dist.sample((len(thetas_accepted),))
            theta_proposal = (thetas_accepted+sigma*perturb_eps).int() #theta + sigma*N(0, Id)
            
            theta_out_prior = theta_proposal>prior_range  #check in prior
            theta_proposal[theta_out_prior]=thetas_accepted[theta_out_prior].int() #if out prior : take thetas accepted

            theta_out_prior = theta_proposal < 1.0  #check in prior
            # print("avant", theta_out_prior)
               
            theta_proposal[theta_out_prior]=thetas_accepted[theta_out_prior].int() #if out prior : take thetas accepted
            # print("apres", theta_proposal < 1.0)
            theta[int((k-1)*nb_train_ABC*prop):int(k*nb_train_ABC*prop)] = theta_proposal
            
        #################################################
    
        theta_dnn = {}
        param = []
        for k, theta_simu in enumerate(theta):
            print(k)
            ##### {chr : theta} #####
            if sigma_spot=="variable":
                sig_2_simu = random.uniform(0.1, 10)
            # sig_2_simu = 0.5
            intensity_simu, noisy = 100,noisy_ref
            param.append((sig_2_simu, intensity_simu))
            C_simu = simulator(theta_simu, resolution, sigma_spot, noisy)

            #C_simu = C_simu + torch.triu(C_simu).transpose(1,0)
            # C_simu = vectorise_upper_matrix(C_simu, resolution)

            C_simu = C_simu + torch.triu(C_simu).transpose(1,0)
            # print(C_simu.size())
            C_simu = pad_input(patch_size, C_simu)
            C_simu = C_simu.reshape(1,1,C_simu.size(0), C_simu.size(1))
            
            theta_dnn[theta_simu] = torch.mean((DNN(C_simu)-DNN(C_ref))**2).detach()
           
        with open(f'{path}{nb_seq+1}_param', 'wb') as f:
                    pickle.dump(param, f)
        with open(f'{path}{nb_seq+1}_theta_dnn', 'wb') as f:
                    pickle.dump(theta_dnn, f)