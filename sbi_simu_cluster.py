import torch
from sbi import utils as utils
from torch.distributions import MultivariateNormal
from itertools import combinations

from sbi.inference import NPE, NLE, DirectPosterior
from sbi.neural_nets.embedding_nets import CNNEmbedding
from sbi.neural_nets import posterior_nn
import random
import numpy as np
import pywt
import pickle

import mlxp 

torch.manual_seed(0)
@mlxp.launch(config_path='configs')#, seeding_function=set_seed)
def main(ctx: mlxp.Context)->None:
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
    sig_2_ref = 1.0
    sigma_spot = "variable"
    noisy = 1
    wavelets=''

    dim=16

    if dim==3:
        chr_seq = chr_seq_3_chr
        chr_cen = chr_cen_3_chr
        prior_range = torch.tensor([230209, 813179, 316619])
        nb_train=1000

        
    if dim==16:
        chr_seq = chr_seq_16_chr
        chr_cen = chr_cen_16_chr
        prior_range = torch.tensor([230209, 813179, 316619, 1531918, 
                                    576869, 270148, 1090947, 562644,
                                    439885, 745746, 666455, 1078176,
                                    924430, 784334, 1091290, 948063])
        nb_train = 5000

    nb_posterior_samples = int(nb_train*0.05)

    nb_bead, start_bead, nb_tot_bead = get_num_beads_and_start(chr_seq, resolution)



    prior = utils.BoxUniform(torch.ones(dim), prior_range-1)
    prior_post = utils.BoxUniform(torch.ones(dim)/prior_range, (prior_range-1)/prior_range)

    theta_ref = torch.tensor(list(chr_cen.values()))

    if origin == "true":
        C_ref =  torch.from_numpy(np.load(f"ref/{dim}_chr_ref_{resolution}_norm_HiC_duan_intra_all.npy")).float()
        if wavelets=="wavelets":
            C_ref, (LH, HL, HH) = pywt.dwt2(C_ref, 'bior1.3')
            C_ref = torch.from_numpy(C_ref) 

    else:
        C_ref = simulator(list(chr_cen.values()), resolution, sigma_spot, noisy)
        if wavelets=="wavelets":
            C_ref, (LH, HL, HH) = pywt.dwt2(C_ref, 'bior1.3')
            C_ref = torch.from_numpy(C_ref)

    def get_simulations_sbi_pretrained(proposal, round, nb_train, DNN):

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

    def get_simulations_sbi(proposal, round, nb_train):

        if round == 0:
            theta = prior.sample((nb_train,))
        else:
            theta=proposal.sample((nb_train,))
            #theta = theta*prior_range

        C = torch.zeros(nb_train,C_ref.size(0), C_ref.size(1))
        
        for k in range(nb_train):
        
            C_tmp = simulator(theta[k], resolution, sigma_spot, noisy)
            # C[k] = vectorise_upper_matrix(C_tmp, resolution)
            if wavelets=="wavelets":
                C_tmp, (LH, HL, HH) = pywt.dwt2(C_tmp, 'bior1.3')
                C_tmp = torch.from_numpy(C_tmp)
            
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

        path_dnn = f"simulation_little_genome/{dim}_chr/{origin}/res_{resolution}/noisy/sigma_{sigma_spot}/summary_stat/kernel_5_linear_decay_4_first_hidden_8192/"
        
        path_sbi = f"simulation_little_genome/{dim}_chr/{origin}/res_{resolution}/noisy/sigma_{sigma_spot}/sbi/"
        nb_seq = 10
        DNN = torch.load(path_dnn+"dnn.pkl", map_location=torch.device('cpu'))
        inference_method = NPE
        density_estimator = "nsf"
        proposal = prior
        #inference = inference_method(prior_post, density_estimator=density_estimator)
        inference = inference_method(prior, density_estimator=density_estimator)


        for k in range(nb_seq+1):
            print("round", k)

            theta, theta_reg = get_simulations_sbi_pretrained(proposal, k, nb_train, DNN)

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
                
            #posterior.set_default_x(C_ref.reshape(1,C_ref.size(0)*C_ref.size(1)))
            proposal = posterior.set_default_x(DNN(C_ref.unsqueeze(0)))

            samples = posterior.sample((nb_posterior_samples,))

            with open(f'{path_sbi}{k}_thetas_accepted', 'wb') as f:
                        #pickle.dump(samples*prior_range, f)
                        pickle.dump(samples, f)

    #embedding net trained jointly
    if 0:
        dim = 3
        nb_train = 10000
        path_sbi = f"simulation_little_genome/{dim}_chr/{origin}/res_{resolution}/noisy/sigma_{sigma_spot}/sbi/"

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

                

    
     
if __name__ == "__main__":
    main()