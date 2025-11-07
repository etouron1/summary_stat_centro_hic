import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils import data
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import pyro.distributions as pdist

from sbi import utils as utils
#from sbi.neural_nets.embedding_nets import CNNEmbedding
# from cnn import CNNEmbedding

import numpy as np
# from simulator import get_num_beads_and_start#, chr_seq_3_chr, chr_cen_3_chr, chr_seq_16_chr, chr_cen_16_chr

from itertools import combinations
import random
import pickle
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # For custom legend handle
from matplotlib.colors import to_rgba


import pandas as pd        
import seaborn as sns

from transformer_row_centro import BioVisionTransformer
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from torch.utils.tensorboard import SummaryWriter

import mlxp



torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

@mlxp.launch(config_path='configs')#, seeding_function=set_seed)
def main(ctx: mlxp.Context)->None:

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

    # chr_seq_3_chr = {"chr1": 230209, "chr2": 813179, "chr3": 316619}

    # chr_cen_3_chr = {'chr1': 151584, 'chr2': 238325, 'chr3': 114499}

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

    chr_start = ctx.config['chr_start']
    chr_end = ctx.config['chr_end']
    chr_to_keep = ['chr'+str(i) for i in range(chr_start, chr_end+1)]


    if dim_ref==3:
        chr_seq_ref = {k: chr_seq_16_chr[k] for k in chr_to_keep}
        chr_cen_ref = {k: chr_cen_16_chr[k] for k in chr_to_keep}
        prior_range_ref = torch.tensor(list(chr_seq_ref.values()))
        print(chr_seq_ref, chr_cen_ref, prior_range_ref)

    if dim_ref==16:
        chr_seq_ref = chr_seq_16_chr
        chr_cen_ref = chr_cen_16_chr
        prior_range_ref = torch.tensor([230209, 813179, 316619, 1531918,
                                    576869, 270148, 1090947, 562644,
                                    439885, 745746, 666455, 1078176,
                                    924430, 784334, 1091290, 948063])

    nb_train_dnn =  500# 50000
    nb_train_abc = 1000

    nb_bead_ref, start_bead_ref, nb_tot_bead_ref = get_num_beads_and_start(chr_seq_ref, resolution)
    print('ref', start_bead_ref, nb_bead_ref)

    nb_bead_16_chr, start_bead_16_chr, nb_tot_bead_16_chr = get_num_beads_and_start(chr_seq_16_chr, resolution)
    prior_range_16_chr = torch.tensor([230209, 813179, 316619, 1531918,
                                    576869, 270148, 1090947, 562644,
                                    439885, 745746, 666455, 1078176,
                                    924430, 784334, 1091290, 948063])
    prior_16_chr = utils.BoxUniform(torch.ones(1), prior_range_16_chr-1)

    if origin=="true":
        # C_ref =  np.load(f"ref/{dim_ref}_chr_ref_{resolution}_norm_HiC_duan_intra_all.npy")
        # C_ref = torch.from_numpy(C_ref).float()
        # C_ref = C_ref + torch.transpose(C_ref, 0,1)


        C_ref_16_chr =  np.load(f"ref/16_chr_ref_{resolution}_norm_HiC_duan_intra_all.npy")
        C_ref_16_chr = torch.from_numpy(C_ref_16_chr).float()
        C_ref_16_chr = C_ref_16_chr + torch.transpose(C_ref_16_chr, 0,1)
        
        C_ref = C_ref_16_chr[start_bead_16_chr['chr'+str(chr_start)]:start_bead_16_chr['chr'+str(chr_end)]+nb_bead_16_chr['chr'+str(chr_end)],start_bead_16_chr['chr'+str(chr_start)]:start_bead_16_chr['chr'+str(chr_end)]+nb_bead_16_chr['chr'+str(chr_end)]]

    plt.matshow(C_ref)
    plt.show()

    prior = utils.BoxUniform(torch.ones(1), prior_range_ref-1)

    theta_ref = torch.tensor(list(chr_cen_ref.values()))


    path = f"simulation_little_genome/{dim_ref}_chr/{origin}/res_{resolution}/noisy/sigma_{sigma_spot}/transformer/row/per_chr/3_pos_per_bloc/cls_token/shared/"



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



    def pad_input_row(patch_size, C, chr_row, chr_seq, start_bead, nb_bead):

            """ pad a row of C (B, C, H, W) (add 0 after each block of the row)"""

            if nb_bead['chr'+str(chr_row)]%patch_size !=0:
                padding_row = patch_size-nb_bead['chr'+str(chr_row)]%patch_size

                row_to_insert = torch.zeros((C.size(0), C.size(1), padding_row, C.size(3)), device=device)
                C = torch.cat((C, row_to_insert), dim=-2) #insert bloc of rows


            padding_cumul = 0

            for i, chr in enumerate(list(chr_seq.keys())):

                    pos_bloc = start_bead[chr] + nb_bead[chr] + padding_cumul

                    if nb_bead[chr]%patch_size !=0:
                        padding = patch_size-nb_bead[chr]%patch_size #nb of 0 to add in the row
                        padding_cumul += padding

                        col_to_insert = torch.zeros((C.size(0), C.size(1), C.size(2), padding), device=device)

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

                    row_to_insert = torch.zeros((padding, C.size(0)), device=C.device)

                    C = torch.cat((C[:pos_bloc], row_to_insert, C[pos_bloc:]), dim=0)
                    col_to_insert = torch.zeros((C.size(0), padding), device=C.device)

                    C = torch.cat((C[:, :pos_bloc], col_to_insert, C[:, pos_bloc:]), dim=1)

            return C

    def create_genome(nb_chr):
        """ create genome and centromere of random size given the number of chromosomes """
        chr_seq = {}
        # chr_cen = {}
        for i in range(nb_chr):
             length_chr = random.randint(200000, 2000000)
            #  pos_centro = random.randint(100, length_chr-100)
             chr_seq['chr'+str(i+1)] = length_chr
            #  chr_cen['chr'+str(i+1)] = pos_centro
        return chr_seq #, chr_cen

    ########## simulate one row of C given a simulated genome chr_seq ###########

    def simulator_row(chr_seq, start_bead, nb_bead, nb_tot_bead, chr_row, theta, resolution, sigma_spot, noisy):
        """ simulate one row of C given a simulated genome chr_seq """

        if sigma_spot=="variable":
            sig_2_simu = random.uniform(0.1, 10)
        else:
            sig_2_simu = sig_2_ref

        intensity_simu = 100

        C_simu = torch.zeros((nb_bead['chr'+str(chr_row)],nb_tot_bead), device=device)

        #for (chr_row, chr_col) in combinations(chr_seq.keys(),r=2):
        n_row = nb_bead['chr'+str(chr_row)]
        c_i_simu = theta[chr_row-1]//resolution

        for chr_col in chr_seq.keys():

            n_col = nb_bead[chr_col]

            index_col = int(chr_col[chr_col.find("chr")+3:])-1

            # c_i_simu = theta[index_row-13]//resolution
            # c_j_simu = theta[index_col-13]//resolution
            c_j_simu = theta[index_col]//resolution

            def simulator_1_bloc(n_row, n_col, c_i, c_j, sig_2, intensity, noisy=noisy):

                # Simulate a noisy matrix C_{n_row x n_col} with a gaussian spot at (c_i, c_j) of size sig_2

                C = torch.zeros((n_row, n_col), device=device)

                # distr = MultivariateNormal(torch.tensor([c_i, c_j]), sig_2*torch.eye(2))

                # indices = torch.tensor([[(i, j) for j in range(len(C[0]))] for i in range(len(C))])
                iy, ix = torch.meshgrid(
                    torch.arange(n_row, dtype=torch.float32, device=device),
                    torch.arange(n_col, dtype=torch.float32, device=device),
                    indexing="ij"
                )

                # C = intensity*torch.exp(distr.log_prob(indices))

                C = intensity * torch.exp(-((iy - c_i)**2 + (ix - c_j)**2) / (2*sig_2))

                if noisy:
                    #mean = intensity * torch.exp(distr.log_prob(torch.tensor([c_i,c_j]))) * 0.1 / 2
                    
                    mean = intensity * 0.1 / 2
                    sigma = intensity * 0.1 / 2

                    noise = mean + sigma*torch.randn((n_row, n_col), device=device)

                    sig = np.sqrt(sig_2)

                    i0 = max(0, int(c_i) - int(sig))
                    i1 = min(n_row, int(c_i) + int(sig) + 1)

                    j0 = max(0, int(c_j) - int(sig))
                    j1 = min(n_col, int(c_j) + int(sig) + 1)

                    noise[i0:i1, :] = 0
                    noise[:, j0:j1] = 0

                else:
                    noise = torch.zeros_like(C, device=device)

                return C+noise

            #C_simu[start_bead[chr_row]:start_bead[chr_row]+nb_bead[chr_row]-1, start_bead[chr_col]:start_bead[chr_col]+nb_bead[chr_col]-1] = simulator_1_bloc(n_row,n_col, c_i_simu, c_j_simu, sig_2_simu, intensity_simu, noisy=noisy)
            # print(C_simu[:, start_bead[chr_col]:start_bead[chr_col]+nb_bead[chr_col]].size() )
            # print(simulator_1_bloc(n_row,n_col, c_i_simu, c_j_simu, sig_2_simu, intensity_simu, noisy=noisy).size())
            C_simu[:, start_bead[chr_col]:start_bead[chr_col]+nb_bead[chr_col]] = simulator_1_bloc(n_row,n_col, c_i_simu, c_j_simu, sig_2_simu, intensity_simu, noisy=noisy)
            C_simu[:, start_bead['chr'+str(chr_row)]:start_bead['chr'+str(chr_row)]+nb_bead['chr'+str(chr_row)]] = 0 #bloc intra

        return C_simu
    ###########################################################################

    ######### simulate variable size of row of C ##########
    def get_simulations_variable(patch_size, training_batch_size):
        """ simulate theta and variable size of C for 3 chr"""

        sigma_spot = "variable"
        noisy = 1

        # genomes = []
        thetas = []
        # thetas_ref = []
        C = []
        nb_chr = 3

        if nb_train_dnn > training_batch_size:
            nb_batchs = nb_train_dnn//training_batch_size
        else:
            nb_batchs = 1
            training_batch_size = nb_train_dnn
        print('nb batchs', nb_batchs)
        print('batch_size', training_batch_size)
        genomes = []
        chr_row_chosen = []
        
        for k in range(nb_batchs):
            batch_theta = []
            batch_C = []
            #nb_chr = random.randint(2,20)
            chr_seq = create_genome(nb_chr) #create a genome
            genomes.append(chr_seq)
            # print(chr_seq)
            nb_bead, start_bead, nb_tot_bead = get_num_beads_and_start(chr_seq, resolution)

            chr_row = random.randint(1, nb_chr) #choose the chr in row
            chr_row_chosen.append(chr_row)
            # print('chr row', chr_row)
            prior_range = torch.tensor(list(chr_seq.values()))
            prior = utils.BoxUniform(torch.ones(nb_chr), prior_range-1)

            for l in range(training_batch_size):
                theta = prior.sample() #sample all centromeres 3 dim
                batch_theta.append(theta[chr_row-1]/prior_range[chr_row-1])

                C_tmp = simulator_row(chr_seq, start_bead, nb_bead, nb_tot_bead, chr_row, theta, resolution, sigma_spot, noisy)
                mini = C_tmp.min()
                maxi = C_tmp.max()
                C_tmp = (C_tmp - mini) / (maxi-mini)
                # print(C_tmp.size())
                C_tmp = C_tmp.reshape(1,1,C_tmp.size(0), C_tmp.size(1))

                C_tmp = pad_input_row(patch_size, C_tmp, chr_row, chr_seq, start_bead, nb_bead) #pad to patch size
                
                # print(chr_seq)
                # print(start_bead, nb_bead)
                # print(chr_row)
                # print(theta[chr_row-1]//resolution)
                
                # plt.matshow(C_tmp[0,0,:,:])
                # plt.axhline(theta[chr_row-1]//resolution)
                # plt.show()

                # plt.matshow(C_tmp[0,0,:,:])
                # plt.show()

                batch_C.append(C_tmp.squeeze(0))

                #thetas.append(theta[chr_row-1]/prior_range[chr_row-1])
                #C.append(C_tmp)
            
            thetas.append(torch.stack(batch_theta))
            C.append(torch.stack(batch_C))
        
        return genomes, chr_row_chosen, thetas, C

    class VariableBatchDataset(Dataset):
        """Dataset holding variable-sized batch tensors and their genome metadata."""
        def __init__(self, genomes_list, chr_row_list, theta_list, C_list):
            assert len(genomes_list) == len(chr_row_list) == len(theta_list) == len(C_list)
            self.genomes_list = genomes_list
            self.chr_row_list = chr_row_list
            self.theta_list = theta_list
            self.C_list = C_list

        def __len__(self):
            return len(self.theta_list)

        def __getitem__(self, idx):
            genome = self.genomes_list[idx]
            chr_row = self.chr_row_list[idx]
            theta_i = self.theta_list[idx]
            C_i = self.C_list[idx]
            return genome, chr_row, theta_i, C_i
    


    def get_dataloaders_variable(patch_size,
            training_batch_size: int = 200,
            validation_fraction: float = 0.1,
        ):

            genomes, chr_row, theta, C_row  = get_simulations_variable(patch_size, training_batch_size) #list of tensor batch
            # print(len(genomes), len(theta), theta[0].size())
            nb_batchs = len(theta)
            nb_batchs_train = int((1 - validation_fraction) * nb_batchs)
            nb_batchs_val = nb_batchs - nb_batchs_train
            # Random split
            perm = torch.randperm(nb_batchs) #permute indices of batchs
            train_idx = perm[:nb_batchs_train] #choose the train batchs
            val_idx = perm[nb_batchs_train:] #choose the val batchs

            # print(perm)
            # print(train_idx)
            # print(val_idx)

            dataset = VariableBatchDataset(genomes, chr_row, theta, C_row) 

            # Dataloader kwargs
            train_loader_kwargs = {
                # "batch_size": min(training_batch_size, num_train),
                "drop_last": True,
                "sampler": SubsetRandomSampler(train_idx.tolist()),
            }
            val_loader_kwargs = {
                # "batch_size": min(training_batch_size, num_val),
                "drop_last": True,
                "sampler": SubsetRandomSampler(val_idx.tolist()),
            }

            # Create loaders for this shape group
            train_loader = data.DataLoader(dataset, **train_loader_kwargs)
            val_loader = data.DataLoader(dataset, **val_loader_kwargs)

            return train_loader, val_loader
            
    ############################################################

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
        """ simulate all transblocks of C for 3 ref chr """

        if sigma_spot=="variable":
            sig_2_simu = random.uniform(0.1, 10)
        else:
            sig_2_simu = sig_2_ref

        intensity_simu = 100

        C_simu = torch.zeros((nb_tot_bead_ref,nb_tot_bead_ref), device=device)

        for (chr_row, chr_col) in combinations(chr_seq_ref.keys(),r=2):

            n_row = chr_seq_ref[chr_row]//resolution
            n_col = chr_seq_ref[chr_col]//resolution
            index_row = int(chr_row[chr_row.find("chr")+3:])-chr_start
            index_col = int(chr_col[chr_col.find("chr")+3:])-chr_start
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

            C_simu[start_bead_ref[chr_row]:start_bead_ref[chr_row]+nb_bead_ref[chr_row]-1, start_bead_ref[chr_col]:start_bead_ref[chr_col]+nb_bead_ref[chr_col]-1] = simulator_1_bloc(n_row,n_col, c_i_simu, c_j_simu, sig_2_simu, intensity_simu, noisy=noisy)

        return C_simu + torch.transpose(C_simu, 0,1)
    
    def simulator_16_chr(theta, resolution, sigma_spot, noisy):
        """ simulate C with 16 ref chr"""

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

                # C = torch.zeros((n_row, n_col), device=device)

                # # distr = MultivariateNormal(torch.tensor([c_i, c_j]), sig_2*torch.eye(2))

                # # indices = torch.tensor([[(i, j) for j in range(len(C[0]))] for i in range(len(C))])
                # iy, ix = torch.meshgrid(
                #     torch.arange(n_row, dtype=torch.float32, device=device),
                #     torch.arange(n_col, dtype=torch.float32, device=device),
                #     indexing="ij"
                # )

                # # C = intensity*torch.exp(distr.log_prob(indices))

                # C = intensity * torch.exp(-((iy - c_i)**2 + (ix - c_j)**2) / (2*sig_2))

                y, x = torch.meshgrid(torch.arange(n_row, device=device), torch.arange(n_col, device=device), indexing='ij')
                # print('x', 'y', x.size(),x, y.size(),  y)
                exponent = -((x - c_j)**2 + (y - c_i)**2) / (2 * sig_2)
                #norm_const = 1.0/(2*torch.pi*sig_2)
                norm_const = 1.0
                # print(exponent.size())
                C = intensity * norm_const*torch.exp(exponent)

                if noisy:
                    # mean = intensity * 0.1 / 2
                    # sigma = intensity * 0.1 / 2

                    # noise = mean + sigma*torch.randn((n_row, n_col), device=device)

                    # sig = np.sqrt(sig_2)

                    # i0 = max(0, int(c_i) - int(sig))
                    # i1 = min(n_row, int(c_i) + int(sig) + 1)

                    # j0 = max(0, int(c_j) - int(sig))
                    # j1 = min(n_col, int(c_j) + int(sig) + 1)

                    # noise[i0:i1, :] = 0
                    # noise[:, j0:j1] = 0

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
    
    def simulator_croissant(theta, chr_seq, start_bead, nb_bead, nb_tot_bead, resolution, sigma_spot, noisy):
        """ simulate C with a given chr_seq and a given number of beads """    

        if sigma_spot=="variable":
            sig_2_simu = random.uniform(0.1, 10)
        else:
            sig_2_simu = sig_2_ref
                
        intensity_simu = 100

        C_simu = torch.zeros((nb_tot_bead,nb_tot_bead), device=device)

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
            
            C_simu[start_bead[chr_row]:start_bead[chr_row]+nb_bead[chr_row]-1, start_bead[chr_col]:start_bead[chr_col]+nb_bead[chr_col]-1] = simulator_1_bloc(n_row,n_col, c_i_simu, c_j_simu, sig_2_simu, intensity_simu, noisy=noisy)
        
        return C_simu + torch.transpose(C_simu, 0,1)
    
    def simulator_row_variable(chr_seq, start_bead, nb_bead, nb_tot_bead, num_chr_row, theta, resolution, sigma_spot, noisy):
        """ simulate row of C with a given chr_seq and a given number of beads """
        chr_row = 'chr'+str(num_chr_row)

        if sigma_spot=="variable":
            sig_2_simu = random.uniform(0.1, 10)
        else:
            sig_2_simu = sig_2_ref

        intensity_simu = 100

        C_simu = torch.zeros((nb_bead[chr_row],nb_tot_bead))

        #for (chr_row, chr_col) in combinations(chr_seq.keys(),r=2):
        n_row = nb_bead[chr_row]
       
        c_i_simu = theta[num_chr_row-1]//resolution

        for chr_col in chr_seq.keys():

            n_col = nb_bead[chr_col]

            index_col = int(chr_col[chr_col.find("chr")+3:])-1

            c_j_simu = theta[index_col]//resolution

            def simulator_1_bloc(n_row, n_col, c_i, c_j, sig_2, intensity, noisy=noisy):

                # Simulate a noisy matrix C_{n_row x n_col} with a gaussian spot at (c_i, c_j) of size sig_2

                C = torch.zeros((n_row, n_col))

                # distr = MultivariateNormal(torch.tensor([c_i, c_j]), sig_2*torch.eye(2))

                # indices = torch.tensor([[(i, j) for j in range(len(C[0]))] for i in range(len(C))])
                iy, ix = torch.meshgrid(
                    torch.arange(n_row, dtype=torch.float32),
                    torch.arange(n_col, dtype=torch.float32),
                    indexing="ij"
                )

                # C = intensity*torch.exp(distr.log_prob(indices))

                C = intensity * torch.exp(-((iy - c_i)**2 + (ix - c_j)**2) / (2*sig_2))

                if noisy:
                    #mean = intensity * torch.exp(distr.log_prob(torch.tensor([c_i,c_j]))) * 0.1 / 2
                    mean = intensity * 0.1 / 2
                    sigma = intensity * 0.1 / 2

                    noise = mean + sigma*torch.randn((n_row, n_col))

                    sig = np.sqrt(sig_2)

                    i0 = max(0, int(c_i) - int(sig))
                    i1 = min(n_row, int(c_i) + int(sig) + 1)

                    j0 = max(0, int(c_j) - int(sig))
                    j1 = min(n_col, int(c_j) + int(sig) + 1)

                    noise[i0:i1, :] = 0
                    noise[:, j0:j1] = 0

                else:
                    noise = torch.zeros_like(C)

                return C+noise
            
            if chr_row != chr_col:
                C_simu[:, start_bead[chr_col]:start_bead[chr_col]+nb_bead[chr_col]] = simulator_1_bloc(n_row,n_col, c_i_simu, c_j_simu, sig_2_simu, intensity_simu, noisy=noisy)

        return C_simu
    
    def get_simulations_C(nb_train_dnn, patch_size):

        theta = prior.sample((nb_train_dnn,))

        C = torch.zeros(nb_train_dnn,1,C_ref.size(0), C_ref.size(1), device=device)

        for k in range(nb_train_dnn):

            C_tmp = simulator(theta[k], resolution, sigma_spot, noisy_ref)
            # C_tmp = pad_input_C(patch_size, C_tmp)
            plt.matshow(C_tmp)
            plt.show()
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
            for chr_id in range(chr_start,chr_end+1):
                C_row = C[:, :, start_bead_ref['chr'+str(chr_id)]:start_bead_ref['chr'+str(chr_id)] + nb_bead_ref['chr'+str(chr_id)], :]

                C_row = pad_input_row(patch_size, C_row,chr_id,chr_seq_ref, start_bead_ref, nb_bead_ref)
                # plt.matshow(C_row[1,0,:,:])
                # plt.show()
                dataset = data.TensorDataset(theta[:,chr_id-chr_start], C_row) #out 1 chr
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
    
    ####### cut intra ###########
    # def loss(theta, C, DNN, chr_id):

    #     #print(DNN(C))
    #     theta_hat = DNN(C, nb_bead_ref, chr_id).squeeze(-1)
    #     # print(theta_hat.size())

    #     loss = (theta_hat - theta) ** 2

    #     return loss
    #############################

    def loss(theta, C, DNN):

        #print(DNN(C))
        theta_hat = DNN(C, nb_bead_ref).squeeze(-1)
        # print(theta_hat.size())

        loss = (theta_hat - theta) ** 2

        return loss
    
    def loss_variable(theta, C, DNN, nb_bead):
        """ loss for C with variable sizes"""

        #print(DNN(C))
        theta_hat = DNN(C, nb_bead).squeeze(-1)
        # print(theta_hat.size())

        loss = (theta_hat - theta) ** 2

        return loss

    def slice_to_key(d, end_key, include_end=True):
        result = {}
        for k, v in d.items():
            result[k] = v
            if k == end_key:
                if not include_end:
                    result.pop(k)
                break
        return result

    # if origin=="true":
    #     C_ref =  np.load(f"ref/{dim_ref}_chr_{1}_ref_{resolution}_norm_HiC_duan_intra_all.npy")
    #     C_ref = torch.from_numpy(C_ref).float()
    # else:
    #     C_ref = simulator(list(chr_cen_ref.values()), resolution, sigma_spot, noisy_ref)

    # print("orginal size", origin, C_ref.size())
    #C_ref = C_ref + torch.triu(C_ref).transpose(1,0)



    ##################### TRAINING ON RANDOM GENOME #################################
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

        embed_dim = ctx.config['embed_dim'] #4*patch_size*patch_size
        depth = ctx.config['depth']
        num_heads = ctx.config['num_heads']

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ######for MLP#####
        # C_ref_padded=pad_input_C(patch_size, C_ref)
        # length_col = C_ref_padded.size(-1)
        #DNN = BioVisionTransformer(in_chans = 1, length_col=length_col//patch_size, nb_bead_chr=nb_bead_ref['chr'+str(ctx.config['chr_id'])], patch_size=patch_size, embed_dim = embed_dim,no_embed_class=True, device=device, depth = depth, num_heads=num_heads)#, chr_seq=chr_seq_ref, chr_cen=chr_cen_ref, dim=dim_ref, resolution=resolution)
        #####################

        DNN = BioVisionTransformer(in_chans = 1, patch_size=patch_size, embed_dim = embed_dim,no_embed_class=True, device=device, depth = depth, num_heads=num_heads)#, chr_seq=chr_seq_ref, chr_cen=chr_cen_ref, dim=dim_ref, resolution=resolution)
        print(DNN)

        # for name, param in DNN.named_parameters():
        #     print(f"{name}: {param.shape}")



        learning_rate = 5e-4 #scheduler
        max_num_epochs= 200

        # print("data loader")
        train_loader, val_loader = get_dataloaders_variable(patch_size)

        # print("train loader and val loader done")
        print('nb train', nb_train_dnn)
        print('max num epochs', max_num_epochs)

        # num_training_steps = len(train_loader) * max_num_epochs

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

        # with open(path+f"dnn_structure.txt", "w") as f:
        #  f.write(str(DNN))


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
        best_val_loss = float('inf')
        train_loss_list = []
        val_loss_list = []
        loss_theta_ref = []
        distance_to_theta_train = [] #over all train data
        distance_to_theta_val = [] #over all test data
        distance_to_theta = {key : [] for key in chr_seq_16_chr.keys()} # distance to theta ref over 200 epochs
        distance_simu_to_theta = {key : [] for key in chr_seq_16_chr.keys()} # distance to theta simu over 200 epochs
        distance_simu_to_theta_variable = {key : [] for key in chr_seq_ref.keys()} # distance to theta variable size over 200 epochs


        ################# 10 C row synthetics of variable size for each chr ####################
        nb_C_synth_variable = 10
        theta_synth_variable = {key : [] for key in chr_seq_ref.keys()}
        C_synth_variable = {key : [] for key in chr_seq_ref.keys()}
        genomes_synth_variable = {key : [] for key in chr_seq_ref.keys()}
        
        nb_chr = 3
        for chr_row in range(1, nb_chr+1):

            for k in range(nb_C_synth_variable):
                chr_seq = create_genome(nb_chr) #create a genome
                genomes_synth_variable['chr'+str(chr_row)].append(chr_seq)
                nb_bead, start_bead, nb_tot_bead = get_num_beads_and_start(chr_seq, resolution)


                prior_range = torch.tensor(list(chr_seq.values()))
                prior = utils.BoxUniform(torch.ones(nb_chr), prior_range-1)

                
                theta = prior.sample() #sample all centromeres 3 dim
                    

                C_tmp = simulator_row(chr_seq, start_bead, nb_bead, nb_tot_bead, chr_row, theta, resolution, sigma_spot, noisy_ref)

                mini = C_tmp.min()
                maxi = C_tmp.max()
                C_tmp = (C_tmp - mini) / (maxi-mini)

                C_tmp = C_tmp.reshape(1,1,C_tmp.size(0), C_tmp.size(1))

                C_tmp = pad_input_row(patch_size, C_tmp, chr_row, chr_seq, start_bead, nb_bead) #pad to patch size
                
                theta_synth_variable['chr'+str(chr_row)].append(theta) #append theta in 3 dim
                C_synth_variable['chr'+str(chr_row)].append(C_tmp)
                # print(chr_seq)
                # print(start_bead, nb_bead)
                # print(chr_row)
                # print(C_tmp[0,0,:,:])
                # plt.matshow(C_tmp[0,0,:,:])
                # plt.axhline(theta[chr_row-1]//resolution)
                # plt.show()
        #####################################################
        ################# 10 C synthetics real size ####################
        nb_C_synth = 10
        theta_synth = []
        C_synth = []
        for k in range(nb_C_synth):

            theta = prior_16_chr.sample()
            theta_synth.append(theta)

            C_tmp = simulator_16_chr(theta, resolution, sigma_spot, noisy_ref)
            C_tmp = C_tmp.reshape(1,1,C_tmp.size(0), C_tmp.size(1))
            C_synth.append(C_tmp)
            
            # plt.matshow(C_tmp[0,0,:,:])
            # for j in range(16):         
            #     plt.axhline(start_bead_16_chr['chr'+str(j+1)] + theta[j]//resolution)
            # plt.show()

        #####################################################
        

        while epoch <= max_num_epochs:
            # train_loader, val_loader = get_dataloaders_variable(patch_size)



            print("epoch", epoch)
            # Train for a single epoch.
            # DNN.train()
            train_loss_sum = 0
            val_loss_sum = 0
            

            estimation_train = 0
            for batch in train_loader:
                
                # print(chr_id)



                optimizer.zero_grad()
                # Get batches on current device.
                chr_seq_variable, chr_row_variable, theta_batch, x_batch= (
                    batch[0],
                    batch[1].item(),
                    batch[2].to(device).squeeze(0),
                    batch[3].to(device).squeeze(0),
                )
                # print(chr_seq_variable)
                # print(chr_row_variable)
                # print(theta_batch[0])
                # print(theta_batch.size())
                # print(x_batch.size())
                

                nb_bead_variable, start, _ = get_num_beads_and_start(chr_seq_variable, resolution)
                # print(nb_bead_variable, start)
                

                prior_range_variable = torch.tensor(list(chr_seq_variable.values()))
                
                theta_hat = DNN(x_batch, nb_bead_variable)*prior_range_variable[chr_row_variable-1]
                
                # print((theta_batch[0]*prior_range_variable[chr_row_variable-1])//resolution)
                # plt.matshow(x_batch[0,0,:,:])
                # plt.axhline((theta_batch[0]*prior_range_variable[chr_row_variable-1])//resolution)
                # plt.show()

                # ######### cut intra ############
                # theta_hat = DNN(x_batch, nb_bead_ref, chr_id)*prior_range_ref[chr_id-1]
                # ################################
                
                estimation_train += torch.mean(torch.abs(theta_hat.squeeze(-1)-theta_batch*prior_range_variable[chr_row_variable-1]))
                #mean over the batch

                train_losses = loss_variable(theta_batch,x_batch, DNN, nb_bead_variable) #losses over batch i
                
                # ############ cut intra #############
                # train_losses = loss(theta_batch,x_batch, DNN, chr_id) #losses over batch i
                # ####################################

                train_loss = torch.mean(train_losses) #mean of losses over batch i
                train_loss_sum += train_losses.sum().item() #sum losses over all item of all batches

                train_loss.backward()

                optimizer.step()
                # scheduler.step()




            # Calculate validation performance.
            # DNN.eval()
            # val_loss_sum = 0


            
            distance_to_theta_train.append(estimation_train.item()*1.0/len(train_loader))

            with torch.no_grad():
                # print("eval")
                estimation_val = 0
                for batch in val_loader:

                        chr_seq_variable, chr_row_variable, theta_batch, x_batch= (
                            batch[0],
                            batch[1].item(),
                            batch[2].to(device).squeeze(0),
                            batch[3].to(device).squeeze(0),
                        )
                        # print(chr_seq_variable)
                        # print(chr_row_variable)
                        # print(theta_batch.size())
                        # print(x_batch.size())
                        
                        nb_bead_variable, start, _ = get_num_beads_and_start(chr_seq_variable, resolution)
                        # print(nb_bead_variable, start)
                        

                        prior_range_variable = torch.tensor(list(chr_seq_variable.values()))

                        theta_hat = DNN(x_batch, nb_bead_variable)*prior_range_variable[chr_row_variable-1]

                        # print("theta", (theta_batch[1]*prior_range_variable[chr_row_variable-1])//resolution)
                        # plt.matshow(x_batch[1,0,:,:])
                        # plt.axhline((theta_batch[1]*prior_range_variable[chr_row_variable-1])//resolution)
                        # plt.show()

                        # ######### cut intra ############
                        # theta_hat = DNN(x_batch, nb_bead_ref, chr)*prior_range_ref[chr-1]
                        # ################################

                        estimation_val += torch.mean(torch.abs(theta_hat.squeeze(-1)-theta_batch*prior_range_variable[chr_row_variable-1]))
                        #mean over the batch

                        # ########### cut intra ###########
                        # val_losses = loss(theta_batch,x_batch, DNN, chr)
                        # #################################

                        val_losses = loss_variable(theta_batch,x_batch, DNN, nb_bead_variable)

                        val_loss_sum += val_losses.sum().item()



                distance_to_theta_val.append(estimation_val.item()*1.0/len(val_loader))
                
                ################ test on simu variable size ###########################
                nb_chr = 3
                for num_chr in range(1, nb_chr+1):
                    
                    distance_simu = 0
                    for k in range(nb_C_synth_variable):

                        theta = theta_synth_variable['chr'+str(num_chr)][k] #3 dim
                        C_tmp_row = C_synth_variable['chr'+str(num_chr)][k]

                        chr_seq_variable = genomes_synth_variable['chr'+str(num_chr)][k]
                        nb_bead_variable, start, _ = get_num_beads_and_start(chr_seq_variable, resolution)

                        prior_range_variable = torch.tensor(list(chr_seq_variable.values()))
                        theta_hat = DNN(C_tmp_row, nb_bead_variable).item()*prior_range_variable[num_chr-1]
                        # print(nb_bead_variable, start)
                        # print(theta[num_chr-1]//resolution)
                        # print(num_chr)
                        # plt.matshow(C_tmp_row[0,0,:,:])
                        # plt.axhline(theta[num_chr-1]//resolution)
                        # plt.show()
                        
                        distance_simu+=torch.abs(theta_hat-theta[num_chr-1])*1.0/nb_C_synth

                    distance_simu_to_theta_variable['chr'+str(num_chr)].append(distance_simu)

                #######################################################################
                
                ############### test on real size ###################
                nb_chr = 16
                loss_ref = 0
                for num_chr in range(1, nb_chr+1):
                    # chr_start = num_chr-1
                    # chr_end = num_chr+1
                    # if chr_start==0:
                    #      chr_start=1
                    #      chr_end = 3
                    # if chr_end==17:
                    #      chr_start = 14
                    #      chr_end = 16
                    group = num_chr//3-int(num_chr%3==0)
                    chr_start = group*3+1
                    chr_end = chr_start+2
                    # print('start - end ', chr_start, chr_end)
                    # print(num_chr)
                    if num_chr==16:
                        chr_start = 14
                        chr_end = 16

                    chr_seq_cut = {k: chr_seq_16_chr[k] for k in ['chr'+str(num_chr) for num_chr in range(chr_start, chr_end+1)]} #select chr_seq for 3 neighbor chr 
                    nb_bead_cut, start_bead_cut, _ = get_num_beads_and_start(chr_seq_cut, resolution)
                    # print(chr_seq_cut)
                    # print(nb_bead_cut, start_bead_cut)
                    C_ref_row = C_ref_16_chr[start_bead_16_chr['chr'+str(num_chr)]:start_bead_16_chr['chr'+str(num_chr)] + nb_bead_16_chr['chr'+str(num_chr)], start_bead_16_chr['chr'+str(chr_start)]:start_bead_16_chr['chr'+str(chr_end)] + nb_bead_16_chr['chr'+str(chr_end)]]
                    
                    mini = C_ref.min()
                    maxi = C_ref.max()
                    C_ref = (C_ref - mini) / (maxi-mini)

                    C_ref_row = C_ref_row.reshape(1,1,C_ref_row.size(0), C_ref_row.size(1))
                    C_ref_row = pad_input_row(patch_size, C_ref_row, num_chr, chr_seq_cut, start_bead_cut, nb_bead_cut)

                    theta_ref = chr_cen_16_chr['chr'+str(num_chr)]

                    # print(chr_seq_cut, nb_bead_cut, start_bead_cut)
                    # print(C_ref_row.size())
                    # plt.matshow(C_ref_row[0,0,:,:])
                    # plt.axhline(theta_ref//resolution)
                    # plt.show()

                    theta_hat = DNN(C_ref_row, nb_bead_cut).item()*prior_range_16_chr[num_chr-1]
                    
                    loss_ref += ((theta_hat - theta_ref)/prior_range_16_chr[num_chr-1])**2
                    
                    distance_to_theta['chr'+str(num_chr)].append(torch.abs(theta_hat-theta_ref))
                    
                    distance_simu = 0
                    for k in range(nb_C_synth):

                        theta = theta_synth[k] #16 dim
                        C_tmp = C_synth[k]
                        
                        # plt.matshow(C_tmp[0,0,:,:])
                        # plt.show()

                        C_tmp_row = C_tmp[:,:, start_bead_16_chr['chr'+str(num_chr)]:start_bead_16_chr['chr'+str(num_chr)] + nb_bead_16_chr['chr'+str(num_chr)], start_bead_16_chr['chr'+str(chr_start)]:start_bead_16_chr['chr'+str(chr_end)] + nb_bead_16_chr['chr'+str(chr_end)]]
                        
                        mini = C_tmp_row[0,0,:,:].min()
                        maxi = C_tmp_row[0,0,:,:].max()
                        C_tmp_row = (C_tmp_row - mini) / (maxi-mini)

                        C_tmp_row = pad_input_row(patch_size, C_tmp_row, num_chr, chr_seq_cut, start_bead_cut, nb_bead_cut)
                        # print(num_chr)
                        # print(nb_bead_cut)
                        # print(C_tmp_row.size())
                        # # print(theta[num_chr-1]//resolution)
                        # plt.matshow(C_tmp_row[0,0,:,:])
                        # plt.axhline(theta[num_chr-1]//resolution)
                        # plt.show()
                        theta_hat = DNN(C_tmp_row, nb_bead_cut).item()*prior_range_16_chr[num_chr-1]
                        
                        distance_simu+=torch.abs(theta_hat-theta[num_chr-1])*1.0/nb_C_synth

                    distance_simu_to_theta['chr'+str(num_chr)].append(distance_simu)

                loss_theta_ref.append(loss_ref/nb_chr)
                

                # ################ test C croissant ######################
                # for num_chr in range(1, 16+1):
                #     chr_id = 'chr'+str(num_chr)
                #     for taille in taille_list:
                #         chr_seq_croissant = slice_to_key(chr_seq_16_chr, taille) #cut dict until taille
                #         # print(chr_seq_croissant)
                #         nb_bead_croissant, start_bead_croissant, nb_tot_bead_croissant = get_num_beads_and_start(chr_seq_croissant, resolution)


                #         C_ref_row = C_ref_16_chr[start_bead_16_chr[chr_id]:start_bead_16_chr[chr_id] + nb_bead_16_chr[chr_id], :start_bead_croissant[taille] + nb_bead_croissant[taille]]
                #         C_ref_row = C_ref_row.reshape(1,1,C_ref_row.size(0), C_ref_row.size(1))
                #         C_ref_row = pad_input_row(patch_size, C_ref_row, num_chr, chr_seq_croissant, start_bead_16_chr, nb_bead_16_chr)
                #         # print(chr_id)
                #         # print(taille)
                #         # print(C_ref_row.size())
                #         # plt.matshow(C_ref_row[0,0,:,:])  
                #         # plt.show()
                #         theta_ref = chr_cen_16_chr[chr_id]
                #         theta_hat = DNN(C_ref_row, nb_bead_croissant).item()*prior_range_16_chr[num_chr-1]
                #         croissant_theta_ref[chr_id][taille].append(torch.abs(theta_hat-theta_ref))

                #         # prior_range_croissant = torch.tensor(list(chr_seq_croissant.values()))
                #         # prior_croissant = utils.BoxUniform(torch.ones(1), prior_range_croissant-1)
                #         # print(num_chr)

                #         croissant_simu = 0
                #         for k in range(nb_C_synth):
                #             # theta = prior_16_chr.sample()
                #             # C_tmp_row = simulator_row_variable(chr_seq_croissant, start_bead_16_chr, nb_bead_16_chr, nb_tot_bead_croissant, num_chr, theta, resolution, sigma_spot, noisy_ref)
                #             # C_tmp_row = C_tmp_row.reshape(1,1,C_tmp_row.size(0), C_tmp_row.size(1))
                #             # C_tmp_row = pad_input_row(patch_size, C_tmp_row, num_chr, chr_seq_croissant, start_bead_16_chr, nb_bead_16_chr)
                            
                            
                #             theta = theta_synth_croissant[k]
                #             C_tmp_row = C_synth_croissant[k][:,:,start_bead_16_chr[chr_id]:start_bead_16_chr[chr_id] + nb_bead_16_chr[chr_id], :start_bead_croissant[taille] + nb_bead_croissant[taille]]
                #             C_tmp_row = pad_input_row(patch_size, C_tmp_row, num_chr, chr_seq_croissant, start_bead_16_chr, nb_bead_16_chr)
                #             # print(theta)
                #             # print(chr_id)
                #             # plt.matshow(C_tmp_row[0,0,:,:])
                #             # plt.show()

                #             theta_hat = DNN(C_tmp_row, nb_bead_croissant).item()*prior_range_16_chr[num_chr-1]
                #             croissant_simu += torch.abs(theta_hat-theta[num_chr-1])*1.0/nb_C_synth
                #         croissant_theta_simu[chr_id][taille].append(croissant_simu)



                # #######################################################

            #nb_item_train = sum(len(train_loader['chr'+str(chr)]) * train_loader['chr'+str(chr)].batch_size for chr in range(train_chr_start,train_chr_end+1))
            nb_item_train = len(train_loader) * train_loader.batch_size
            train_loss_average = train_loss_sum / (
                    nb_item_train  
                ) #average loss over all items of all batchs of all chr

            #nb_item_val = sum(len(val_loader['chr'+str(chr)]) * val_loader['chr'+str(chr)].batch_size for chr in range(train_chr_start, train_chr_end+1))
            nb_item_val = len(val_loader) * val_loader.batch_size
            val_loss = val_loss_sum / (
                nb_item_val  
            ) #average loss over all items of all batchs of all chr

            train_loss_list.append(train_loss_average)
            val_loss_list.append(val_loss)

            epoch += 1
            ### save model with lowest val loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(DNN.state_dict(), 'best_model.pth')
                print(f"Saved new best model at epoch {epoch+1} with val_loss={val_loss:.4f}")


        ## to follow parameters
        #     for name, param in DNN.named_parameters():
        #         writer.add_histogram(f"{name}.weights", param, epoch)
        #         if param.grad is not None:
        #             writer.add_histogram(f"{name}.grads", param.grad, epoch)

        #     writer.add_scalar("Loss/train", train_loss_average, epoch)
        # writer.close()

        # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        # print(distance_to_theta)
        # print(distance_simu_to_theta)
        DNN.zero_grad(set_to_none=True)

        # compute min/mean/max of the distance over the last 50 epochs
        nb_chr = 16
        plt.figure()
        
        for chr in range(1, nb_chr+1):
            values = distance_simu_to_theta['chr'+str(chr)][-50:]
            min_simu = min(values).item()
            max_simu = max(values).item()
            moy_simu = (sum(values)/len(values)).item()

            values = distance_to_theta['chr'+str(chr)][-50:]
            min_ref = min(values).item()
            max_ref = max(values).item()
            moy_ref = (sum(values)/len(values)).item()

            # results = {"chr" : ctx.config['chr_train'],"min_simu": min_simu,"max_simu": max_simu,"moy_simu": moy_simu,"min_ref": min_ref,"max_ref": max_ref,"moy_ref": moy_ref}
            df_min = []
            df_moy = []
            df_max = []
            df_min.append({'chr': f"chr{chr}",'origin': 'ref','stat': 'min','value': min_ref})
            df_min.append({'chr': f"chr{chr}",'origin': 'simu','stat': 'min','value': min_simu})
            df_moy.append({'chr': f"chr{chr}",'origin': 'ref','stat': 'mean','value': moy_ref})
            df_moy.append({'chr': f"chr{chr}",'origin': 'simu','stat': 'mean','value': moy_simu})
            df_max.append({'chr': f"chr{chr}",'origin': 'ref','stat': 'max','value': max_ref})
            df_max.append({'chr': f"chr{chr}",'origin': 'simu','stat': 'max','value': max_simu})
            df_min = pd.DataFrame(df_min)
            df_moy = pd.DataFrame(df_moy)
            df_max = pd.DataFrame(df_max)
            df = [df_min, df_moy, df_max]
            # --- save to pickle file ---
            with open(f"chr_{chr}_stat.pkl", "wb") as f:
                pickle.dump(df, f)
            with open(f"chr_{chr}_stat.pkl", "rb") as f:
                df = pickle.load(f)
            print(df)
            color_min = [to_rgba('green') for _ in range(1)]
            color_moy = [to_rgba('orange') for _ in range(1)]
            color_max = [to_rgba('red') for _ in range(1)]
            colors = [color_min, color_moy, color_max]
            #plt.figure()
            
            for k in range(3):
                sns.stripplot(data=df[k],x='chr',y='value',hue='origin', hue_order=['simu', 'ref'],dodge=True,jitter=False,size=6, palette=colors[k])
            plt.title(f'Min-mean-max over last 50 epochs')
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
        # plt.savefig('stripplot.png')
            # plt.savefig(f"chr_{ctx.config['chr_train']}_stripplot.png")

        print(f"converged after {epoch} epochs with a validation loss {val_loss_list[-1]}")

        plt.figure()
        plt.title('loss')
        plt.plot(range(epoch), torch.sqrt(torch.tensor(train_loss_list)), label="train")
        plt.plot(range(epoch), torch.sqrt(torch.tensor(val_loss_list)), label="val")
        plt.legend()
        plt.tight_layout()
        plt.show()
        # plt.savefig('loss.png')
        # plt.savefig(f"chr_{ctx.config['chr_train']}_loss.png")

        # plt.savefig(path+f"chr_{ctx.config['chr_id']}_{nb_train_dnn}_loss_patch_size_{patch_size}_embed_dim_{embed_dim}_nb_blocks_{depth}_nb_heads_{num_heads}.png")
            #plt.savefig(path+f"{nb_train}_loss_linear_{decay_linear}.png")

        plt.figure()
        plt.title('loss C_ref')
        plt.plot(range(epoch), torch.sqrt(torch.tensor(loss_theta_ref)))
        plt.legend()
        plt.tight_layout()
        plt.show()
        # plt.savefig('loss_c_ref.png')
        # plt.savefig(f"chr_{ctx.config['chr_train']}_loss_c_ref.png")


        for num_chr in range(1, nb_chr+1):
            # print('chr'+str(num_chr))
            plt.figure()
            # print(distance_to_theta)
            # print(distance_to_theta['chr'+str(num_chr)])
            plt.plot(range(epoch), distance_to_theta['chr'+str(num_chr)])
            plt.title('chr'+str(num_chr) + ' ref' + f'depth_{depth}_nb_head_{num_heads}')
            plt.ylabel('|theta_ref - DNN(C_ref)|')
            plt.axhline(y=resolution)
            plt.tight_layout()
            plt.show()
            # plt.savefig(f'chr_{num_chr}_distance_ref.png')

            #plt.savefig(path+f"chr_{ctx.config['chr_id']}_{nb_train_dnn}_distance_to_theta_chr_{num_chr}_patch_size_{patch_size}_embed_dim_{embed_dim}_nb_blocks_{depth}_nb_heads_{num_heads}.png")

        for num_chr in range(1, nb_chr+1):
            # print('chr'+str(num_chr))
            plt.figure()
            # print(distance_simu_to_theta)
            # print(distance_simu_to_theta['chr'+str(num_chr)])
            plt.plot(range(epoch), distance_simu_to_theta['chr'+str(num_chr)])
            plt.title('chr'+str(num_chr) + ' simu' + f'depth_{depth}_nb_head_{num_heads}')
            plt.ylabel('|theta - DNN(C)|')
            plt.axhline(y=resolution)
            plt.tight_layout()
            plt.show()
            # plt.savefig(f'chr_{num_chr}_distance_simu.png')

            #plt.savefig(path+f"chr_{ctx.config['chr_id']}_{nb_train_dnn}_distance_simu_to_theta_chr_{num_chr}_patch_size_{patch_size}_embed_dim_{embed_dim}_nb_blocks_{depth}_nb_heads_{num_heads}.png")
        nb_chr = 3
        for num_chr in range(1, nb_chr+1):
            # print('chr'+str(num_chr))
            plt.figure()
            # print(distance_simu_to_theta)
            # print(distance_simu_to_theta['chr'+str(num_chr)])
            plt.plot(range(epoch), distance_simu_to_theta_variable['chr'+str(num_chr)])
            plt.title('chr'+str(num_chr) + ' simu variable ' + f'depth_{depth}_nb_head_{num_heads}')
            plt.ylabel('|theta - DNN(C)|')
            plt.axhline(y=resolution)
            plt.tight_layout()
            plt.show()
            # plt.savefig(f'chr_{num_chr}_distance_simu_variable.png')

        plt.figure()
        plt.plot(range(epoch), distance_to_theta_train, label='train')
        plt.plot(range(epoch), distance_to_theta_val, label='val')
        plt.title(' train/val' + f'depth_{depth}_nb_head_{num_heads}')
        plt.ylabel('|theta - DNN(C)|')
        plt.axhline(y=resolution)
        plt.legend()
        plt.tight_layout()
        plt.show()
        # plt.savefig('distance_train_val.png')

            #plt.savefig(path+f"{nb_train_dnn}_distance_train_val_to_theta_chr_{num_chr}_patch_size_{patch_size}_embed_dim_{embed_dim}_nb_blocks_{depth}_nb_heads_{num_heads}.png")
        
        # ############### croissant test #################
        # df_min = []
        # df_moy = []
        # df_max = []
        # for num_chr in range(1, 16+1):
        #     # print('chr'+str(num_chr))
        #     for taille in taille_list:
        #         values = croissant_theta_ref['chr'+str(num_chr)][taille][-50:]
        #         min_croissant_theta_ref = min(values)
        #         max_croissant_theta_ref = max(values)
        #         moy_croissant_theta_ref = sum(values)/len(values)
        #         df_min.append({'chr': f'chr{num_chr}',
        #             'taille': taille,
        #             'stat': 'min',
        #             'value': min_croissant_theta_ref.item()})
        #         df_moy.append({'chr': f'chr{num_chr}',
        #             'taille': taille,
        #             'stat': 'mean',
        #             'value': moy_croissant_theta_ref.item()})
        #         df_max.append({'chr': f'chr{num_chr}',
        #             'taille': taille,
        #             'stat': 'max',
        #             'value': max_croissant_theta_ref.item()})
        # df_min = pd.DataFrame(df_min)
        # df_moy = pd.DataFrame(df_moy)
        # df_max = pd.DataFrame(df_max)
        # df = [df_min, df_moy, df_max]
       
        # color_min = [to_rgba('green') for _ in range(len(taille_list))]
        # color_moy = [to_rgba('orange') for _ in range(len(taille_list))]
        # color_max = [to_rgba('red') for _ in range(len(taille_list))]
        # colors = [color_min, color_moy, color_max]
        # plt.figure()
        # for k in range(3):
        #     sns.stripplot(data=df[k],x='chr',y='value',hue='taille',dodge=True,jitter=False,size=6, palette=colors[k])
        # plt.title(f'ref - Min-mean-max over last 50 epochs')
        # plt.ylabel('|theta_ref - DNN(C_ref)|')
        # handles = [
        # Line2D([0], [0], marker='o', color='red', label='max',
        #     markerfacecolor='red', markersize=6),
        # Line2D([0], [0], marker='o', color='orange', label='mean',
        #     markerfacecolor='orange', markersize=6),
        # Line2D([0], [0], marker='o', color='green', label='min',
        #     markerfacecolor='green', markersize=6),
        # ]
        # plt.axhline(y = resolution, linestyle="--", color="black")

        # plt.legend(handles=handles)
        # plt.tight_layout()
        # plt.show()

        # for num_chr in range(1, 16+1):
        #     # print('chr'+str(num_chr))
        #     plt.figure()
        #     for taille in taille_list:
        #         plt.plot(range(epoch), croissant_theta_ref['chr'+str(num_chr)][taille], label=taille)
        #     plt.title('chr'+str(num_chr) + ' ref - croissant' + f'depth_{depth}_nb_head_{num_heads}')
        #     plt.ylabel('|theta_ref - DNN(C_ref)|')
        #     plt.axhline(y=resolution)
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.show()
        #     # plt.savefig(path+f"chr_{ctx.config['chr_id']}_{nb_train_dnn}_decroissant_ref_patch_size_{patch_size}_embed_dim_{embed_dim}_nb_blocks_{depth}_nb_heads_{num_heads}.png")
        
        # df_min = []
        # df_moy = []
        # df_max = []
        # for num_chr in range(1, 16+1):
        #     # print('chr'+str(num_chr))
        #     for taille in taille_list:
        #         values = croissant_theta_simu['chr'+str(num_chr)][taille][-50:]
        #         min_croissant_theta_simu = min(values)
        #         max_croissant_theta_simu = max(values)
        #         moy_croissant_theta_simu = sum(values)/len(values)
        #         df_min.append({'chr': f'chr{num_chr}',
        #             'taille': taille,
        #             'stat': 'min',
        #             'value': min_croissant_theta_simu.item()})
        #         df_moy.append({'chr': f'chr{num_chr}',
        #             'taille': taille,
        #             'stat': 'mean',
        #             'value': moy_croissant_theta_simu.item()})
        #         df_max.append({'chr': f'chr{num_chr}',
        #             'taille': taille,
        #             'stat': 'max',
        #             'value': max_croissant_theta_simu.item()})
        # df_min = pd.DataFrame(df_min)
        # df_moy = pd.DataFrame(df_moy)
        # df_max = pd.DataFrame(df_max)
        # df = [df_min, df_moy, df_max]
       
        # color_min = [to_rgba('green') for _ in range(len(taille_list))]
        # color_moy = [to_rgba('orange') for _ in range(len(taille_list))]
        # color_max = [to_rgba('red') for _ in range(len(taille_list))]
        # colors = [color_min, color_moy, color_max]
        # plt.figure()
        # for k in range(3):
        #     sns.stripplot(data=df[k],x='chr',y='value',hue='taille',dodge=True,jitter=False,size=6, palette=colors[k])
        # plt.title(f'simu - Min-mean-max over 50 last epochs')
        # plt.ylabel('|theta - DNN(C)|')
        # handles = [
        # Line2D([0], [0], marker='o', color='red', label='max',
        #     markerfacecolor='red', markersize=6),
        # Line2D([0], [0], marker='o', color='orange', label='mean',
        #     markerfacecolor='orange', markersize=6),
        # Line2D([0], [0], marker='o', color='green', label='min',
        #     markerfacecolor='green', markersize=6),
        # ]
        # plt.axhline(y = resolution, linestyle="--", color="black")

        # plt.legend(handles=handles)
        # plt.tight_layout()
        # plt.show()

        # for num_chr in range(1,16+1):
        #     # print('chr'+str(num_chr))
        #     plt.figure()
        #     for taille in taille_list:
        #         plt.plot(range(epoch), croissant_theta_simu['chr'+str(num_chr)][taille], label=taille)
        #     plt.title('chr'+str(num_chr) + ' simu - croissant' + f'depth_{depth}_nb_head_{num_heads}')
        #     plt.ylabel('|theta - DNN(C)|')
        #     plt.axhline(y=resolution)
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.show()
        #     # plt.savefig(path+f"chr_{ctx.config['chr_id']}_{nb_train_dnn}_decroissant_simu_patch_size_{patch_size}_embed_dim_{embed_dim}_nb_blocks_{depth}_nb_heads_{num_heads}.png")

        # ##############################################

        #torch.save(DNN, path+f"chr_{ctx.config['chr_id']}_dnn.pkl")
        with open(path+f"chr_{ctx.config['chr_id']}_convergence_info_{nb_train_dnn}_patch_size_{patch_size}_embed_dim_{embed_dim}_nb_blocks_{depth}_nb_heads_{num_heads}.txt", "w") as f:
            #with open(path+f"convergence_info_{nb_train}_linear_{decay_linear}.txt", "w") as f:
                f.write(f"converged after {epoch} epochs with a validation loss {val_loss_list[-1]} \n")
                f.write(f"theta_ref : {theta_ref}, DNN(C_ref) : {DNN(C_ref)*prior_range}\n")
                f.write(f"||theta_ref - DNN(C_ref)|| = {torch.sqrt(torch.mean((DNN(C_ref)*prior_range-theta_ref)**2))}")
    
    ##################### TEST ON THE BEST DNN ####################################
    if 0:
        patch_size = 4

        embed_dim = ctx.config['embed_dim'] #4*patch_size*patch_size
        depth = ctx.config['depth']
        num_heads = ctx.config['num_heads']

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        path= 'simulation_little_genome/3_chr/true/res_32000/noisy/sigma_variable/transformer/row/shared/cluster/50000_train_set/'
        DNN = BioVisionTransformer(in_chans = 1, patch_size=patch_size, embed_dim = embed_dim,no_embed_class=True, device=device, depth = depth, num_heads=num_heads)
        DNN.load_state_dict(torch.load(path+'best_model.pth'))
        DNN.eval()
        
        distance_to_theta = {key : float for key in chr_seq_16_chr.keys()}
        distance_simu_to_theta = {key : [] for key in chr_seq_16_chr.keys()}
        distance_simu_to_theta_cut = {key : [] for key in chr_seq_16_chr.keys()}
        distance_simu_to_theta_variable = {key : [] for key in chr_seq_ref.keys()}

        ################# 10 C row synthetics of variable size for each chr ####################
        nb_C_synth_variable = 100
        theta_synth_variable = {key : [] for key in chr_seq_ref.keys()}
        C_synth_variable = {key : [] for key in chr_seq_ref.keys()}
        genomes_synth_variable = {key : [] for key in chr_seq_ref.keys()}
        
        nb_chr = 3
        for chr_row in range(1, nb_chr+1):
            #per chr generate 10 C_row of variable size
            for k in range(nb_C_synth_variable): 
                chr_seq = create_genome(nb_chr) #create a genome
                genomes_synth_variable['chr'+str(chr_row)].append(chr_seq)
                nb_bead, start_bead, nb_tot_bead = get_num_beads_and_start(chr_seq, resolution)


                prior_range = torch.tensor(list(chr_seq.values()))
                prior = utils.BoxUniform(torch.ones(nb_chr), prior_range-1)

                
                theta = prior.sample() #sample all centromeres 3 dim
                    
                C_tmp = simulator_row(chr_seq, start_bead, nb_bead, nb_tot_bead, chr_row, theta, resolution, sigma_spot, noisy_ref)
                C_tmp = C_tmp.reshape(1,1,C_tmp.size(0), C_tmp.size(1))
                C_tmp = pad_input_row(patch_size, C_tmp, chr_row, chr_seq, start_bead, nb_bead) #pad to patch size
                
                theta_synth_variable['chr'+str(chr_row)].append(theta) #append theta in 3 dim
                C_synth_variable['chr'+str(chr_row)].append(C_tmp)
                # print(chr_seq)
                # print(start_bead, nb_bead)
                # print(chr_row)
                # plt.matshow(C_tmp[0,0,:,:])
                # plt.axhline(theta[chr_row-1]//resolution)
                # plt.show()
        #####################################################

        ################# 10 C synthetics real size ####################
        nb_C_synth = 100
        theta_synth = {key : [] for key in chr_seq_16_chr.keys()}
        C_synth = {key : [] for key in chr_seq_16_chr.keys()}
        genomes_synth = {key : [] for key in chr_seq_16_chr.keys()}
        
        nb_chr = 16
        nb_blocks = 3
        for chr_row in range(1, nb_chr+1):
            #per chr generate 10 C_row of variable size
            group = chr_row//3-int(chr_row%3==0)
            
            chr_start = group*3+1
            chr_end = chr_start+2
            # print('start - end ', chr_start, chr_end)
            # print(chr_row)
            if chr_row==16:
                    chr_start = 14
                    chr_end = 16
            chr_seq_cut = {k: chr_seq_16_chr[k] for k in ['chr'+str(chr_row) for chr_row in range(chr_start, chr_end+1)]}
            
            chr_seq = {}
            for i,chr in enumerate(chr_seq_ref.keys()):
                 chr_seq[chr] = list(chr_seq_cut.values())[i]
            # print('old', chr_row)
            if chr_row==16:
                 chr_row_relative = 3
            else:
                chr_row_relative = chr_row%3 +3*int(chr_row%3==0)
            
            # print('new', chr_row)
            for k in range(nb_C_synth): 
                
                genomes_synth['chr'+str(chr_row)].append(chr_seq)
                nb_bead, start_bead, nb_tot_bead = get_num_beads_and_start(chr_seq, resolution)

                prior_range = torch.tensor(list(chr_seq.values()))
                prior = utils.BoxUniform(torch.ones(nb_blocks), prior_range-1)
                
                theta = prior.sample() #sample all centromeres 3 dim
                    
                C_tmp = simulator_row(chr_seq, start_bead, nb_bead, nb_tot_bead, chr_row_relative, theta, resolution, sigma_spot, noisy_ref)
                C_tmp = C_tmp.reshape(1,1,C_tmp.size(0), C_tmp.size(1))
                C_tmp = pad_input_row(patch_size, C_tmp, chr_row_relative, chr_seq, start_bead, nb_bead) #pad to patch size
                # print(nb_bead)
                # plt.matshow(C_tmp[0,0,:,:])
                # plt.show()
                theta_synth['chr'+str(chr_row)].append(theta) #append theta in 3 dim
                C_synth['chr'+str(chr_row)].append(C_tmp)

        ############# cut 16 chr #####################
        nb_C_synth_cut = 100
        theta_synth_cut = []
        C_synth_cut = []
        for k in range(nb_C_synth_cut):
            #generate C with 16 chr
            theta = prior_16_chr.sample()
            theta_synth_cut.append(theta)

            C_tmp = simulator_16_chr(theta, resolution, sigma_spot, noisy_ref)
            C_tmp = C_tmp.reshape(1,1,C_tmp.size(0), C_tmp.size(1))
            # plt.matshow(C_tmp[0,0,:,:])
            # plt.show()
            C_synth_cut.append(C_tmp)
        #####################################################

        ################ test on simu variable size ###########################
        nb_chr = 3
        for num_chr in range(1, nb_chr+1):
            
            distance_simu = 0
            for k in range(nb_C_synth_variable):
                
                theta = theta_synth_variable['chr'+str(num_chr)][k] #3 dim
                C_tmp_row = C_synth_variable['chr'+str(num_chr)][k]

                chr_seq_variable = genomes_synth_variable['chr'+str(num_chr)][k]
                nb_bead_variable, start, _ = get_num_beads_and_start(chr_seq_variable, resolution)

                prior_range_variable = torch.tensor(list(chr_seq_variable.values()))
                theta_hat = DNN(C_tmp_row, nb_bead_variable).item()*prior_range_variable[num_chr-1]
                # print(nb_bead_variable, start)
                # print(num_chr)
                # print(theta[num_chr-1]//resolution)
                # print(theta_hat//resolution)
                # plt.matshow(C_tmp_row[0,0,:,:])
                # plt.axhline(theta[num_chr-1]//resolution, color='green', label='true')
                # plt.axhline(theta_hat//resolution, color='red', label='estim')
                # plt.legend()
                # plt.show()
                
                #distance_simu+=torch.abs(theta_hat-theta[num_chr-1])*1.0/nb_C_synth
                distance_simu_to_theta_variable['chr'+str(num_chr)].append(torch.abs(theta_hat-theta[num_chr-1]))


            #distance_simu_to_theta_variable['chr'+str(num_chr)] = distance_simu

        #######################################################################

        ##################### test on real size ###############################

        nb_chr = 16
        for num_chr in range(1, nb_chr+1):
                # chr_start = num_chr-1
                # chr_end = num_chr+1
                # if chr_start==0:
                #      chr_start=1
                #      chr_end = 3
                # if chr_end==17:
                #      chr_start = 14
                #      chr_end = 16
                group = num_chr//3-int(num_chr%3==0)
                chr_start = group*3+1
                chr_end = chr_start+2
                # print('start - end ', chr_start, chr_end)
                # print(num_chr)
                if num_chr==16:
                     chr_start = 14
                     chr_end = 16


                C_ref_row = C_ref_16_chr[start_bead_16_chr['chr'+str(num_chr)]:start_bead_16_chr['chr'+str(num_chr)] + nb_bead_16_chr['chr'+str(num_chr)], start_bead_16_chr['chr'+str(chr_start)]:start_bead_16_chr['chr'+str(chr_end)] + nb_bead_16_chr['chr'+str(chr_end)]]
                C_ref_row = C_ref_row.reshape(1,1,C_ref_row.size(0), C_ref_row.size(1))
                
                chr_seq_cut = {k: chr_seq_16_chr[k] for k in ['chr'+str(num_chr) for num_chr in range(chr_start, chr_end+1)]}
                nb_bead_cut, start_bead_cut, _ = get_num_beads_and_start(chr_seq_cut, resolution)
                
                C_ref_row = pad_input_row(patch_size, C_ref_row, num_chr, chr_seq_cut, start_bead_cut, nb_bead_cut)
                
                theta_ref = chr_cen_16_chr['chr'+str(num_chr)]
                theta_hat = DNN(C_ref_row, nb_bead_cut).item()*prior_range_16_chr[num_chr-1]
                
                # print(num_chr)
                # print(chr_seq_cut, nb_bead_cut, start_bead_cut)
                # print(theta_ref//resolution)
                # print(theta_hat//resolution)
                # C_ref_cut = C_ref_16_chr[start_bead_16_chr['chr'+str(chr_start)]:start_bead_16_chr['chr'+str(chr_end)] + nb_bead_16_chr['chr'+str(chr_end)], start_bead_16_chr['chr'+str(chr_start)]:start_bead_16_chr['chr'+str(chr_end)] + nb_bead_16_chr['chr'+str(chr_end)]]
                # if num_chr==chr_start:
                #     plt.matshow(C_ref_cut)
                # plt.axhline(start_bead_cut['chr'+str(num_chr)]+theta_ref//resolution, color='green', label='true')
                # plt.axhline(start_bead_cut['chr'+str(num_chr)] + theta_hat//resolution, color='red', label='estim')
                # if num_chr==chr_end:
                #     # plt.legend()
                #     plt.show()

                distance_to_theta['chr'+str(num_chr)] = torch.abs(theta_hat-theta_ref)
                
                
                if num_chr==16:
                    num_chr_relative = 3
                else:
                    num_chr_relative = num_chr%3 +3*int(num_chr%3==0)
                distance_simu = 0
                for k in range(nb_C_synth):
                    
                    theta = theta_synth['chr'+str(num_chr)][k] #3 dim
                    C_tmp_row = C_synth['chr'+str(num_chr)][k]

                    chr_seq = genomes_synth['chr'+str(num_chr)][k]
                    nb_bead, start, _ = get_num_beads_and_start(chr_seq, resolution)

                    prior_range = torch.tensor(list(chr_seq.values()))
                    theta_hat = DNN(C_tmp_row, nb_bead).item()*prior_range[num_chr_relative-1]
                    # print(nb_bead_variable, start)
                    # print(num_chr)
                    # print(theta[num_chr_relative-1]//resolution)
                    # print(theta_hat//resolution)
                    # plt.matshow(C_tmp_row[0,0,:,:])
                    # plt.axhline(theta[num_chr_relative-1]//resolution, color='green', label='true')
                    # plt.axhline(theta_hat//resolution, color='red', label='estim')
                    # plt.legend()
                    # plt.show()
                    
                    #distance_simu+=torch.abs(theta_hat-theta[num_chr-1])*1.0/nb_C_synth
                    distance_simu_to_theta['chr'+str(num_chr)].append(torch.abs(theta_hat-theta[num_chr_relative-1]))
                
                ########## cut 16 chr ############
                distance_simu = 0
                for k in range(nb_C_synth_cut):

                    theta = theta_synth_cut[k]
                    C_tmp = C_synth_cut[k]
                    # fig, ax = plt.subplots(1,2)
                    # ax[0].matshow(C_tmp[0,0,:,:])
                    # plt.show()

                    C_tmp_row = C_tmp[:,:, start_bead_16_chr['chr'+str(num_chr)]:start_bead_16_chr['chr'+str(num_chr)] + nb_bead_16_chr['chr'+str(num_chr)], start_bead_16_chr['chr'+str(chr_start)]:start_bead_16_chr['chr'+str(chr_end)] + nb_bead_16_chr['chr'+str(chr_end)]]
                    # ax[1].matshow(C_tmp_row[0,0,:,:])
                    # plt.show()
                    C_tmp_row = pad_input_row(patch_size, C_tmp_row, num_chr, chr_seq_cut, start_bead_cut, nb_bead_cut)
                    theta_hat = DNN(C_tmp_row, nb_bead_cut).item()*prior_range_16_chr[num_chr-1]
                    
                    # print(num_chr)
                    # print(chr_seq_cut, nb_bead_cut, start_bead_cut)
                    # print(theta[num_chr-1]//resolution)
                    # print(theta_hat//resolution)
                    # plt.matshow(C_tmp_row[0,0,:,:])
                    # plt.axhline(theta[num_chr-1]//resolution, color='green', label='true')
                    # plt.axhline(theta_hat//resolution, color='red', label='estim')
                    # plt.legend()
                    # plt.show()
                    # C_simu_cut = C_tmp[:,:,start_bead_16_chr['chr'+str(chr_start)]:start_bead_16_chr['chr'+str(chr_end)] + nb_bead_16_chr['chr'+str(chr_end)], start_bead_16_chr['chr'+str(chr_start)]:start_bead_16_chr['chr'+str(chr_end)] + nb_bead_16_chr['chr'+str(chr_end)]]
                    # if k==0:
                    #     if num_chr==16 :
                    #         plt.matshow(C_simu_cut[0,0,:,:])
                    # # plt.matshow(C_tmp_row[0,0,:,:])
                    #     plt.axhline(start_bead_cut['chr'+str(num_chr)]+theta[num_chr-1]//resolution, color='green', label='true')
                    #     plt.axhline(start_bead_cut['chr'+str(num_chr)]+theta_hat//resolution, color='red', label='estim')
                    #     if num_chr==chr_end:
                    #         plt.legend()
                    #         plt.show()

                    # distance_simu+=torch.abs(theta_hat-theta[num_chr-1])*1.0/nb_C_synth
                    distance_simu_to_theta_cut['chr'+str(num_chr)].append(torch.abs(theta_hat-theta[num_chr-1]))

                # distance_simu_to_theta['chr'+str(num_chr)] = distance_simu
        
        #######################################################################
        fig, ax = plt.subplots(1,3, sharey=True)

        for k, chr in enumerate(chr_seq_ref):
        
            ax[k].boxplot(distance_simu_to_theta_variable[chr])
            ax[k].axhline(y = resolution, linestyle="--", color="black")
            ax[k].set_title(f'{chr}')
        plt.show()

        chr_seq_sorted = dict(sorted(chr_seq_16_chr.items(), key=lambda item: item[1]))

        fig, ax = plt.subplots(4,4, sharey=True)
        for k,chr in enumerate(chr_seq_sorted):
        
            ax[k//4, k%4].boxplot(distance_simu_to_theta[chr])
            ax[k//4, k%4].axhline(y = resolution, linestyle="--", color="black")
            ax[k//4, k%4].set_title(f'{chr}')
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(4,4, sharey=True)
        for k,chr in enumerate(chr_seq_sorted):
        
            ax[k//4, k%4].boxplot(distance_simu_to_theta_cut[chr])
            ax[k//4, k%4].axhline(y = resolution, linestyle="--", color="black")
            ax[k//4, k%4].set_title(f'{chr}')
        plt.tight_layout()
        plt.show()

        color = {'chr1':'red', 'chr2':'green', 'chr3':'blue'}
        for chr in chr_seq_ref:
            
            for k in range(len(distance_simu_to_theta_variable[chr])):
                plt.scatter(k+1, distance_simu_to_theta_variable[chr][k].item(), color=color[chr])
            plt.axhline(y = resolution, linestyle="--", color="black")
            plt.title(f'{chr}')

        handles = [
        Line2D([0], [0], marker='o', color='red', label='chr1',
            markerfacecolor='red', markersize=6),
        Line2D([0], [0], marker='o', color='green', label='chr2',
            markerfacecolor='green', markersize=6),
        Line2D([0], [0], marker='o', color='blue', label='chr3',
            markerfacecolor='blue', markersize=6),
            ]
            

        plt.legend(handles=handles)
        plt.show()


        plt.figure()

        for chr in chr_seq_ref:
                        
            df = []
            
            df.append({'chr': f"{chr}",'origin': 'simu_variable','value': distance_simu_to_theta_variable[chr].item()})
            
            df = pd.DataFrame(df)
            
            color = {'simu_variable': 'green'}
            
            
            sns.stripplot(data=df,x='chr',y='value',hue='origin', hue_order=['simu_variable'],dodge=True,jitter=False,size=6, palette=color, legend=False)
        plt.title(f'Distance to theta variable for min val loss DNN')
        plt.ylabel('|theta - DNN(C)|')

        handles = [
            Line2D([0], [0], marker='o', color='green', label='simu variable',
                markerfacecolor='green', markersize=6)]
            

        plt.legend(handles=handles)
        plt.axhline(y = resolution, linestyle="--", color="black")

            
        plt.tight_layout()
        plt.show()

        plt.figure()
        chr_seq_sorted = dict(sorted(chr_seq_16_chr.items(), key=lambda item: item[1]))

        for chr in chr_seq_sorted:
                        
            df = []
            
            df.append({'chr': f"{chr}",'origin': 'ref','value': distance_to_theta[chr].item()})
            df.append({'chr': f"{chr}",'origin': 'simu','value': distance_simu_to_theta[chr].item()})
            
            df = pd.DataFrame(df)
            
            # color = [to_rgba('red') for _ in range(1)]
            color = {'simu': 'red', 'ref': 'blue'}
            
            

            sns.stripplot(data=df,x='chr',y='value',hue='origin', hue_order=['simu', 'ref'],dodge=True,jitter=False,size=6, palette=color, legend=False)
        plt.title(f'Distance to theta for min val loss DNN')
        plt.ylabel('|theta - DNN(C)|')

        handles = [
            Line2D([0], [0], marker='o', color='red', label='simu',
                markerfacecolor='red', markersize=6),
            Line2D([0], [0], marker='o', color='blue', label='ref',
                markerfacecolor='blue', markersize=6),
        ]

        plt.legend(handles=handles)
        plt.axhline(y = resolution, linestyle="--", color="black")

            
        plt.tight_layout()
        plt.show()
        #################################################################################################




    ##################### TRAINING ON CHOOSEN CHR #################################
    if 0:
        # writer = SummaryWriter(log_dir=path)
        patch_size = 4
        # C_ref = pad_input_C(patch_size, C_ref) # to give a size for the train set
        # C_ref = C_ref.reshape(1,1,C_ref.size(0), C_ref.size(1))



        # C_ref = pad_input(patch_size, chr_seq_ref, start_bead_ref, nb_bead_ref, C_ref, 1)
        # print('padded size', C_ref.size())
        # # plt.matshow(C_ref)
        # # plt.show()
        # C_ref = C_ref.reshape(1,1,C_ref.size(0), C_ref.size(1))

        embed_dim = ctx.config['embed_dim'] #4*patch_size*patch_size
        depth = ctx.config['depth']
        num_heads = ctx.config['num_heads']

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ######for MLP#####
        # C_ref_padded=pad_input_C(patch_size, C_ref)
        # length_col = C_ref_padded.size(-1)
        #DNN = BioVisionTransformer(in_chans = 1, length_col=length_col//patch_size, nb_bead_chr=nb_bead_ref['chr'+str(ctx.config['chr_id'])], patch_size=patch_size, embed_dim = embed_dim,no_embed_class=True, device=device, depth = depth, num_heads=num_heads)#, chr_seq=chr_seq_ref, chr_cen=chr_cen_ref, dim=dim_ref, resolution=resolution)
        #####################

        DNN = BioVisionTransformer(in_chans = 1, patch_size=patch_size, embed_dim = embed_dim,no_embed_class=True, device=device, depth = depth, num_heads=num_heads)#, chr_seq=chr_seq_ref, chr_cen=chr_cen_ref, dim=dim_ref, resolution=resolution)
        print(DNN)

        # for name, param in DNN.named_parameters():
        #     print(f"{name}: {param.shape}")



        learning_rate = 5e-4 #scheduler
        max_num_epochs= 200

        print("data loader")
        train_loader, val_loader = get_dataloaders(nb_train_dnn, patch_size)
        
        print("train loader and val loader done")
        print('nb train', nb_train_dnn)

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

        # with open(path+f"dnn_structure.txt", "w") as f:
        #  f.write(str(DNN))


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
        distance_to_theta_train = []
        distance_to_theta_val = []
        distance_to_theta = {key : [] for key in chr_seq_ref.keys()}
        distance_simu_to_theta = {key : [] for key in chr_seq_ref.keys()}

        # taille_list = ['chr3', 'chr6', 'chr9', 'chr12', 'chr15']
        # croissant_theta_ref = {key : { k : [] for k in taille_list} for key in chr_seq_16_chr.keys()}
        # croissant_theta_simu = {key : { k : [] for k in taille_list} for key in chr_seq_16_chr.keys()}

        train_chr_start = ctx.config['chr_train']
        train_chr_end = ctx.config['chr_train']

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
        
        # ################# 10 C synthetics croissant ####################
        
        # theta_synth_croissant = []
        # C_synth_croissant = [] #10 C de taille 16 chr
   
        # for k in range(nb_C_synth):

        #     theta = prior_16_chr.sample()
        #     theta_synth_croissant.append(theta)

        #     C_tmp = simulator_16_chr(theta, resolution, sigma_spot, noisy_ref)
        #     C_tmp = C_tmp.reshape(1,1,C_tmp.size(0), C_tmp.size(1))
        #     C_synth_croissant.append(C_tmp)

        # #######################################################################

        while epoch <= max_num_epochs:
            iter_loader = {name: iter(loader) for name, loader in train_loader.items()}


            print("epoch", epoch)
            # Train for a single epoch.
            # DNN.train()
            train_loss_sum = 0
            val_loss_sum = 0
            #for chr_id in range(1,dim_ref+1):


                # Train for a single epoch.
                # DNN.train()
            # print('train')

            estimation_train = 0
            for k in range(sum(len(loader) for loader in train_loader.values())):
                chr_id = torch.randint(train_chr_start, train_chr_end+1, (1,)).item() #choose chr
                loader_name = 'chr'+str(chr_id) #choose train loader
                try:
                    batch = next(iter_loader[loader_name])

                except StopIteration:

                    # Iterator exhausted, recreate it
                    iter_loader[loader_name] = iter(train_loader[loader_name])
                    batch = next(iter_loader[loader_name])
                #     print(batch)
                # print(chr_id)



                optimizer.zero_grad()
                # Get batches on current device.
                theta_batch, x_batch= (
                    batch[0].to(device),
                    batch[1].to(device),
                )
                # print(chr_id)
                # print(x_batch.size())

                # plt.matshow(x_batch[0,0,:,:])
                # plt.show()

                theta_hat = DNN(x_batch, nb_bead_ref)*prior_range_ref[chr_id-chr_start]

                # ######### cut intra ############
                # theta_hat = DNN(x_batch, nb_bead_ref, chr_id)*prior_range_ref[chr_id-1]
                # ################################

                estimation_train += torch.mean(torch.abs(theta_hat.squeeze(-1)-theta_batch*prior_range_ref[chr_id-chr_start]))

                train_losses = loss(theta_batch,x_batch, DNN) #losses over batch i

                # ############ cut intra #############
                # train_losses = loss(theta_batch,x_batch, DNN, chr_id) #losses over batch i
                # ####################################

                train_loss = torch.mean(train_losses) #mean of losses over batch i
                train_loss_sum += train_losses.sum().item() #sum losses over all item of all batches

                train_loss.backward()

                optimizer.step()
                # scheduler.step()




            # Calculate validation performance.
            # DNN.eval()
            # val_loss_sum = 0



            nb_simu_train = sum(len(train_loader['chr'+str(chr)]) for chr in range(train_chr_start,train_chr_end+1)) #nb_batchs for all chr
            distance_to_theta_train.append(estimation_train.item()*1.0/nb_simu_train)


            with torch.no_grad():
                # print("eval")
                estimation_val = 0
                for chr in range (train_chr_start, train_chr_end+1):
                    for batch in val_loader['chr'+str(chr)]:

                        theta_batch, x_batch = (
                            batch[0].to(device),
                            batch[1].to(device),

                        )
                        # print(chr)
                        # plt.matshow(x_batch[1,0,:,:])
                        # plt.show()

                        theta_hat = DNN(x_batch, nb_bead_ref)*prior_range_ref[chr-chr_start]

                        # ######### cut intra ############
                        # theta_hat = DNN(x_batch, nb_bead_ref, chr)*prior_range_ref[chr-1]
                        # ################################

                        estimation_val += torch.mean(torch.abs(theta_hat.squeeze(-1)-theta_batch*prior_range_ref[chr-chr_start]))

                        # ########### cut intra ###########
                        # val_losses = loss(theta_batch,x_batch, DNN, chr)
                        # #################################

                        val_losses = loss(theta_batch,x_batch, DNN)

                        val_loss_sum += val_losses.sum().item()



            nb_simu_val = sum(len(val_loader['chr'+str(chr)]) for chr in range(train_chr_start,train_chr_end+1)) #nb batchs for all chr
            distance_to_theta_val.append(estimation_val.item()*1.0/nb_simu_val)

            loss_ref = 0
            for num_chr in range(train_chr_start, train_chr_end+1):

                C_ref_row = C_ref[start_bead_ref['chr'+str(num_chr)]:start_bead_ref['chr'+str(num_chr)] + nb_bead_ref['chr'+str(num_chr)], :]
                C_ref_row = C_ref_row.reshape(1,1,C_ref_row.size(0), C_ref_row.size(1))
                
                C_ref_row = pad_input_row(patch_size, C_ref_row, num_chr, chr_seq_ref, start_bead_ref, nb_bead_ref)
                # print(num_chr)
                # plt.matshow(C_ref_row[0,0,:,:])
                # plt.show()
                theta_ref = chr_cen_ref['chr'+str(num_chr)]
                theta_hat = DNN(C_ref_row, nb_bead_ref).item()*prior_range_ref[num_chr-chr_start]

                # ######### cut intra ############
                # theta_hat = DNN(C_ref_row, nb_bead_ref, num_chr).item()*prior_range_ref[num_chr-1]
                # ################################
                
                distance_to_theta['chr'+str(num_chr)].append(torch.abs(theta_hat-theta_ref))
                loss_ref += ((theta_hat - theta_ref)/prior_range_ref[num_chr-chr_start])**2
                
                distance_simu = 0
                for k in range(nb_C_synth):

                    # theta = prior.sample()

                    # C_tmp = simulator(theta, resolution, sigma_spot, noisy_ref)
                    # C_tmp = C_tmp.reshape(1,1,C_tmp.size(0), C_tmp.size(1))

                    # C_tmp_row = C_tmp[:,:, start_bead_ref['chr'+str(num_chr)]:start_bead_ref['chr'+str(num_chr)] + nb_bead_ref['chr'+str(num_chr)], :]
                    # C_tmp_row = pad_input_row(patch_size, C_tmp_row, num_chr, chr_seq_ref, start_bead_ref, nb_bead_ref)
                    theta = theta_synth[k]
                    C_tmp = C_synth[k]

                    C_tmp_row = C_tmp[:,:, start_bead_ref['chr'+str(num_chr)]:start_bead_ref['chr'+str(num_chr)] + nb_bead_ref['chr'+str(num_chr)], :]
                    
                    C_tmp_row = pad_input_row(patch_size, C_tmp_row, num_chr, chr_seq_ref, start_bead_ref, nb_bead_ref)
                    # print(num_chr)
                    # plt.matshow(C_tmp_row[0,0,:,:])
                    # plt.show()
                    theta_hat = DNN(C_tmp_row, nb_bead_ref).item()*prior_range_ref[num_chr-chr_start]

                    # ######### cut intra ############
                    # theta_hat = DNN(C_tmp_row, nb_bead_ref, num_chr).item()*prior_range_ref[num_chr-1]
                    # ################################

                    distance_simu+=torch.abs(theta_hat-theta[num_chr-chr_start])*1.0/nb_C_synth

                distance_simu_to_theta['chr'+str(num_chr)].append(distance_simu)

            loss_theta_ref.append(loss_ref/(train_chr_end-train_chr_start+1))
            

            # ################ test C croissant ######################
            # for num_chr in range(1, 16+1):
            #     chr_id = 'chr'+str(num_chr)
            #     for taille in taille_list:
            #         chr_seq_croissant = slice_to_key(chr_seq_16_chr, taille) #cut dict until taille
            #         # print(chr_seq_croissant)
            #         nb_bead_croissant, start_bead_croissant, nb_tot_bead_croissant = get_num_beads_and_start(chr_seq_croissant, resolution)


            #         C_ref_row = C_ref_16_chr[start_bead_16_chr[chr_id]:start_bead_16_chr[chr_id] + nb_bead_16_chr[chr_id], :start_bead_croissant[taille] + nb_bead_croissant[taille]]
            #         C_ref_row = C_ref_row.reshape(1,1,C_ref_row.size(0), C_ref_row.size(1))
            #         C_ref_row = pad_input_row(patch_size, C_ref_row, num_chr, chr_seq_croissant, start_bead_16_chr, nb_bead_16_chr)
            #         # print(chr_id)
            #         # print(taille)
            #         # print(C_ref_row.size())
            #         # plt.matshow(C_ref_row[0,0,:,:])  
            #         # plt.show()
            #         theta_ref = chr_cen_16_chr[chr_id]
            #         theta_hat = DNN(C_ref_row, nb_bead_croissant).item()*prior_range_16_chr[num_chr-1]
            #         croissant_theta_ref[chr_id][taille].append(torch.abs(theta_hat-theta_ref))

            #         # prior_range_croissant = torch.tensor(list(chr_seq_croissant.values()))
            #         # prior_croissant = utils.BoxUniform(torch.ones(1), prior_range_croissant-1)
            #         # print(num_chr)

            #         croissant_simu = 0
            #         for k in range(nb_C_synth):
            #             # theta = prior_16_chr.sample()
            #             # C_tmp_row = simulator_row_variable(chr_seq_croissant, start_bead_16_chr, nb_bead_16_chr, nb_tot_bead_croissant, num_chr, theta, resolution, sigma_spot, noisy_ref)
            #             # C_tmp_row = C_tmp_row.reshape(1,1,C_tmp_row.size(0), C_tmp_row.size(1))
            #             # C_tmp_row = pad_input_row(patch_size, C_tmp_row, num_chr, chr_seq_croissant, start_bead_16_chr, nb_bead_16_chr)
                        
                        
            #             theta = theta_synth_croissant[k]
            #             C_tmp_row = C_synth_croissant[k][:,:,start_bead_16_chr[chr_id]:start_bead_16_chr[chr_id] + nb_bead_16_chr[chr_id], :start_bead_croissant[taille] + nb_bead_croissant[taille]]
            #             C_tmp_row = pad_input_row(patch_size, C_tmp_row, num_chr, chr_seq_croissant, start_bead_16_chr, nb_bead_16_chr)
            #             # print(theta)
            #             # print(chr_id)
            #             # plt.matshow(C_tmp_row[0,0,:,:])
            #             # plt.show()

            #             theta_hat = DNN(C_tmp_row, nb_bead_croissant).item()*prior_range_16_chr[num_chr-1]
            #             croissant_simu += torch.abs(theta_hat-theta[num_chr-1])*1.0/nb_C_synth
            #         croissant_theta_simu[chr_id][taille].append(croissant_simu)



            # #######################################################

            nb_item_train = sum(len(train_loader['chr'+str(chr)]) * train_loader['chr'+str(chr)].batch_size for chr in range(train_chr_start,train_chr_end+1))
            train_loss_average = train_loss_sum / (
                    nb_item_train  
                ) #average loss over all items of all batchs of all chr

            nb_item_val = sum(len(val_loader['chr'+str(chr)]) * val_loader['chr'+str(chr)].batch_size for chr in range(train_chr_start, train_chr_end+1))
            val_loss = val_loss_sum / (
                nb_item_val  
            ) #average loss over all items of all batchs of all chr

            train_loss_list.append(train_loss_average)
            val_loss_list.append(val_loss)

            epoch += 1


        ## to follow parameters
        #     for name, param in DNN.named_parameters():
        #         writer.add_histogram(f"{name}.weights", param, epoch)
        #         if param.grad is not None:
        #             writer.add_histogram(f"{name}.grads", param.grad, epoch)

        #     writer.add_scalar("Loss/train", train_loss_average, epoch)
        # writer.close()

        # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        # print(distance_to_theta)
        # print(distance_simu_to_theta)
        DNN.zero_grad(set_to_none=True)

        values = distance_simu_to_theta['chr'+str(ctx.config['chr_train'])][-50:]
        min_simu = min(values).item()
        max_simu = max(values).item()
        moy_simu = (sum(values)/len(values)).item()

        values = distance_to_theta['chr'+str(ctx.config['chr_train'])][-50:]
        min_ref = min(values).item()
        max_ref = max(values).item()
        moy_ref = (sum(values)/len(values)).item()

        # results = {"chr" : ctx.config['chr_train'],"min_simu": min_simu,"max_simu": max_simu,"moy_simu": moy_simu,"min_ref": min_ref,"max_ref": max_ref,"moy_ref": moy_ref}
        df_min = []
        df_moy = []
        df_max = []
        df_min.append({'chr': f"chr{ctx.config['chr_train']}",'origin': 'ref','stat': 'min','value': min_ref})
        df_min.append({'chr': f"chr{ctx.config['chr_train']}",'origin': 'simu','stat': 'min','value': min_simu})
        df_moy.append({'chr': f"chr{ctx.config['chr_train']}",'origin': 'ref','stat': 'mean','value': moy_ref})
        df_moy.append({'chr': f"chr{ctx.config['chr_train']}",'origin': 'simu','stat': 'mean','value': moy_simu})
        df_max.append({'chr': f"chr{ctx.config['chr_train']}",'origin': 'ref','stat': 'max','value': max_ref})
        df_max.append({'chr': f"chr{ctx.config['chr_train']}",'origin': 'simu','stat': 'max','value': max_simu})
        df_min = pd.DataFrame(df_min)
        df_moy = pd.DataFrame(df_moy)
        df_max = pd.DataFrame(df_max)
        df = [df_min, df_moy, df_max]
        # --- save to pickle file ---
        with open(f"chr_{ctx.config['chr_train']}_stat.pkl", "wb") as f:
            pickle.dump(df, f)
        with open(f"chr_{ctx.config['chr_train']}_stat.pkl", "rb") as f:
            df = pickle.load(f)
        print(df)
        color_min = [to_rgba('green') for _ in range(1)]
        color_moy = [to_rgba('orange') for _ in range(1)]
        color_max = [to_rgba('red') for _ in range(1)]
        colors = [color_min, color_moy, color_max]
        plt.figure()
        
        for k in range(3):
            sns.stripplot(data=df[k],x='chr',y='value',hue='origin', hue_order=['simu', 'ref'],dodge=True,jitter=False,size=6, palette=colors[k])
        plt.title(f'Min-mean-max over last 50 epochs')
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
        # plt.show()
        plt.savefig(f"chr_{ctx.config['chr_train']}_stripplot.png")

        print(f"converged after {epoch} epochs with a validation loss {val_loss_list[-1]}")

        plt.figure()
        plt.title('loss')
        plt.plot(range(epoch), torch.sqrt(torch.tensor(train_loss_list)), label="train")
        plt.plot(range(epoch), torch.sqrt(torch.tensor(val_loss_list)), label="val")
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"chr_{ctx.config['chr_train']}_loss.png")

        # plt.savefig(path+f"chr_{ctx.config['chr_id']}_{nb_train_dnn}_loss_patch_size_{patch_size}_embed_dim_{embed_dim}_nb_blocks_{depth}_nb_heads_{num_heads}.png")
            #plt.savefig(path+f"{nb_train}_loss_linear_{decay_linear}.png")

        plt.figure()
        plt.title('loss C_ref')
        plt.plot(range(epoch), torch.sqrt(torch.tensor(loss_theta_ref)))
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"chr_{ctx.config['chr_train']}_loss_c_ref.png")


        for num_chr in range(train_chr_start, train_chr_end+1):
            # print('chr'+str(num_chr))
            plt.figure()
            # print(distance_to_theta)
            # print(distance_to_theta['chr'+str(num_chr)])
            plt.plot(range(epoch), distance_to_theta['chr'+str(num_chr)])
            plt.title('chr'+str(num_chr) + ' ref' + f'depth_{depth}_nb_head_{num_heads}')
            plt.ylabel('|theta_ref - DNN(C_ref)|')
            plt.axhline(y=resolution)
            plt.tight_layout()
            # plt.show()
            plt.savefig(f'chr_{num_chr}_distance_ref.png')

            #plt.savefig(path+f"chr_{ctx.config['chr_id']}_{nb_train_dnn}_distance_to_theta_chr_{num_chr}_patch_size_{patch_size}_embed_dim_{embed_dim}_nb_blocks_{depth}_nb_heads_{num_heads}.png")

        for num_chr in range(train_chr_start, train_chr_end+1):
            # print('chr'+str(num_chr))
            plt.figure()
            # print(distance_simu_to_theta)
            # print(distance_simu_to_theta['chr'+str(num_chr)])
            plt.plot(range(epoch), distance_simu_to_theta['chr'+str(num_chr)])
            plt.title('chr'+str(num_chr) + ' simu' + f'depth_{depth}_nb_head_{num_heads}')
            plt.ylabel('|theta - DNN(C)|')
            plt.axhline(y=resolution)
            plt.tight_layout()
            # plt.show()
            plt.savefig(f'chr_{num_chr}_distance_simu.png')

            #plt.savefig(path+f"chr_{ctx.config['chr_id']}_{nb_train_dnn}_distance_simu_to_theta_chr_{num_chr}_patch_size_{patch_size}_embed_dim_{embed_dim}_nb_blocks_{depth}_nb_heads_{num_heads}.png")


        plt.figure()
        plt.plot(range(epoch), distance_to_theta_train, label='train')
        plt.plot(range(epoch), distance_to_theta_val, label='val')
        plt.title(' train/val' + f'depth_{depth}_nb_head_{num_heads}')
        plt.ylabel('|theta - DNN(C)|')
        plt.axhline(y=resolution)
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'chr_{num_chr}_distance_train_val.png')

            #plt.savefig(path+f"{nb_train_dnn}_distance_train_val_to_theta_chr_{num_chr}_patch_size_{patch_size}_embed_dim_{embed_dim}_nb_blocks_{depth}_nb_heads_{num_heads}.png")
        
        # ############### croissant test #################
        # df_min = []
        # df_moy = []
        # df_max = []
        # for num_chr in range(1, 16+1):
        #     # print('chr'+str(num_chr))
        #     for taille in taille_list:
        #         values = croissant_theta_ref['chr'+str(num_chr)][taille][-50:]
        #         min_croissant_theta_ref = min(values)
        #         max_croissant_theta_ref = max(values)
        #         moy_croissant_theta_ref = sum(values)/len(values)
        #         df_min.append({'chr': f'chr{num_chr}',
        #             'taille': taille,
        #             'stat': 'min',
        #             'value': min_croissant_theta_ref.item()})
        #         df_moy.append({'chr': f'chr{num_chr}',
        #             'taille': taille,
        #             'stat': 'mean',
        #             'value': moy_croissant_theta_ref.item()})
        #         df_max.append({'chr': f'chr{num_chr}',
        #             'taille': taille,
        #             'stat': 'max',
        #             'value': max_croissant_theta_ref.item()})
        # df_min = pd.DataFrame(df_min)
        # df_moy = pd.DataFrame(df_moy)
        # df_max = pd.DataFrame(df_max)
        # df = [df_min, df_moy, df_max]
       
        # color_min = [to_rgba('green') for _ in range(len(taille_list))]
        # color_moy = [to_rgba('orange') for _ in range(len(taille_list))]
        # color_max = [to_rgba('red') for _ in range(len(taille_list))]
        # colors = [color_min, color_moy, color_max]
        # plt.figure()
        # for k in range(3):
        #     sns.stripplot(data=df[k],x='chr',y='value',hue='taille',dodge=True,jitter=False,size=6, palette=colors[k])
        # plt.title(f'ref - Min-mean-max over last 50 epochs')
        # plt.ylabel('|theta_ref - DNN(C_ref)|')
        # handles = [
        # Line2D([0], [0], marker='o', color='red', label='max',
        #     markerfacecolor='red', markersize=6),
        # Line2D([0], [0], marker='o', color='orange', label='mean',
        #     markerfacecolor='orange', markersize=6),
        # Line2D([0], [0], marker='o', color='green', label='min',
        #     markerfacecolor='green', markersize=6),
        # ]
        # plt.axhline(y = resolution, linestyle="--", color="black")

        # plt.legend(handles=handles)
        # plt.tight_layout()
        # plt.show()

        # for num_chr in range(1, 16+1):
        #     # print('chr'+str(num_chr))
        #     plt.figure()
        #     for taille in taille_list:
        #         plt.plot(range(epoch), croissant_theta_ref['chr'+str(num_chr)][taille], label=taille)
        #     plt.title('chr'+str(num_chr) + ' ref - croissant' + f'depth_{depth}_nb_head_{num_heads}')
        #     plt.ylabel('|theta_ref - DNN(C_ref)|')
        #     plt.axhline(y=resolution)
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.show()
        #     # plt.savefig(path+f"chr_{ctx.config['chr_id']}_{nb_train_dnn}_decroissant_ref_patch_size_{patch_size}_embed_dim_{embed_dim}_nb_blocks_{depth}_nb_heads_{num_heads}.png")
        
        # df_min = []
        # df_moy = []
        # df_max = []
        # for num_chr in range(1, 16+1):
        #     # print('chr'+str(num_chr))
        #     for taille in taille_list:
        #         values = croissant_theta_simu['chr'+str(num_chr)][taille][-50:]
        #         min_croissant_theta_simu = min(values)
        #         max_croissant_theta_simu = max(values)
        #         moy_croissant_theta_simu = sum(values)/len(values)
        #         df_min.append({'chr': f'chr{num_chr}',
        #             'taille': taille,
        #             'stat': 'min',
        #             'value': min_croissant_theta_simu.item()})
        #         df_moy.append({'chr': f'chr{num_chr}',
        #             'taille': taille,
        #             'stat': 'mean',
        #             'value': moy_croissant_theta_simu.item()})
        #         df_max.append({'chr': f'chr{num_chr}',
        #             'taille': taille,
        #             'stat': 'max',
        #             'value': max_croissant_theta_simu.item()})
        # df_min = pd.DataFrame(df_min)
        # df_moy = pd.DataFrame(df_moy)
        # df_max = pd.DataFrame(df_max)
        # df = [df_min, df_moy, df_max]
       
        # color_min = [to_rgba('green') for _ in range(len(taille_list))]
        # color_moy = [to_rgba('orange') for _ in range(len(taille_list))]
        # color_max = [to_rgba('red') for _ in range(len(taille_list))]
        # colors = [color_min, color_moy, color_max]
        # plt.figure()
        # for k in range(3):
        #     sns.stripplot(data=df[k],x='chr',y='value',hue='taille',dodge=True,jitter=False,size=6, palette=colors[k])
        # plt.title(f'simu - Min-mean-max over 50 last epochs')
        # plt.ylabel('|theta - DNN(C)|')
        # handles = [
        # Line2D([0], [0], marker='o', color='red', label='max',
        #     markerfacecolor='red', markersize=6),
        # Line2D([0], [0], marker='o', color='orange', label='mean',
        #     markerfacecolor='orange', markersize=6),
        # Line2D([0], [0], marker='o', color='green', label='min',
        #     markerfacecolor='green', markersize=6),
        # ]
        # plt.axhline(y = resolution, linestyle="--", color="black")

        # plt.legend(handles=handles)
        # plt.tight_layout()
        # plt.show()

        # for num_chr in range(1,16+1):
        #     # print('chr'+str(num_chr))
        #     plt.figure()
        #     for taille in taille_list:
        #         plt.plot(range(epoch), croissant_theta_simu['chr'+str(num_chr)][taille], label=taille)
        #     plt.title('chr'+str(num_chr) + ' simu - croissant' + f'depth_{depth}_nb_head_{num_heads}')
        #     plt.ylabel('|theta - DNN(C)|')
        #     plt.axhline(y=resolution)
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.show()
        #     # plt.savefig(path+f"chr_{ctx.config['chr_id']}_{nb_train_dnn}_decroissant_simu_patch_size_{patch_size}_embed_dim_{embed_dim}_nb_blocks_{depth}_nb_heads_{num_heads}.png")

        # ##############################################

        #torch.save(DNN, path+f"chr_{ctx.config['chr_id']}_dnn.pkl")
        with open(path+f"chr_{ctx.config['chr_id']}_convergence_info_{nb_train_dnn}_patch_size_{patch_size}_embed_dim_{embed_dim}_nb_blocks_{depth}_nb_heads_{num_heads}.txt", "w") as f:
            #with open(path+f"convergence_info_{nb_train}_linear_{decay_linear}.txt", "w") as f:
                f.write(f"converged after {epoch} epochs with a validation loss {val_loss_list[-1]} \n")
                f.write(f"theta_ref : {theta_ref}, DNN(C_ref) : {DNN(C_ref)*prior_range}\n")
                f.write(f"||theta_ref - DNN(C_ref)|| = {torch.sqrt(torch.mean((DNN(C_ref)*prior_range-theta_ref)**2))}")

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


    if 0:
        device = torch.device('cpu')
        patch_size = 4

        # C_ref = C_ref + torch.triu(C_ref).transpose(1,0)
        # C_ref = pad_input(patch_size, C_ref)
        # C_ref = C_ref.reshape(1,1,C_ref.size(0), C_ref.size(1))
        # img =  np.load(f"ref/{dim_ref}_chr_{1}_ref_{resolution}_norm_HiC_duan_intra_all.npy")
        # img = torch.from_numpy(img).float()
        # img = pad_input(patch_size, chr_seq_ref, start_bead_ref, nb_bead_ref, img, 1)
        state_dict = torch.load(path+"dnn_state.pkl", map_location="cpu")

        embed_dim = 4*patch_size*patch_size
        depth = 4
        num_heads = 16


        DNN = BioVisionTransformer(in_chans = 1, patch_size=patch_size, embed_dim = embed_dim,no_embed_class=True, device=device, depth = depth, num_heads=num_heads)#, chr_seq=chr_seq_ref, chr_cen=chr_cen_ref, dim=dim_ref, resolution=resolution)

        DNN.load_state_dict(state_dict)
        DNN.to("cpu")
        # DNN = torch.load(path+"dnn.pkl", map_location=device, weights_only=False)
        print(DNN)

        for num_chr in range(1,17):
            C_ref =  np.load(f"ref/{dim_ref}_chr_{num_chr}_ref_{resolution}_norm_HiC_duan_intra_all.npy")
            # C_ref =  np.load(f"ref/{dim}_chr_end_ref_{resolution}_norm_HiC_duan_intra_all.npy")
            C_ref = torch.from_numpy(C_ref).float()

            C_ref = C_ref.reshape(1,1,C_ref.size(0), C_ref.size(1))

            theta_ref = chr_cen_ref['chr'+str(num_chr)]


            print(f"theta_ref : {theta_ref}, DNN(C_ref) : {DNN(C_ref).item()*prior_range_ref[num_chr-1]}\n")
            print(f"||theta_ref - DNN(C_ref)|| = {torch.sqrt(torch.mean((DNN(C_ref)*prior_range_ref[num_chr-1]-theta_ref)**2))}")
            print()
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

if __name__ == "__main__":
    main()
