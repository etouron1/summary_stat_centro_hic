
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
import pyro.distributions as pdist

from sbi import utils as utils
from sbi.neural_nets.embedding_nets import CNNEmbedding

import numpy as np
from simulator import get_num_beads_and_start, chr_seq_3_chr, chr_cen_3_chr, chr_seq_16_chr, chr_cen_16_chr

from itertools import combinations
import random
import pickle
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
from simulator import plot_C_genome
from performance import mean_distance_closest_theta


torch.manual_seed(0)

origin="true"
sigma_spot = "variable"
noisy_ref = 1
sig_2_ref = 1

dim = 16

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
    
resolution = 32000
nb_bead, start_bead, nb_tot_bead = get_num_beads_and_start(chr_seq, resolution)
    
prior = utils.BoxUniform(torch.ones(dim), prior_range-1)

theta_ref = torch.tensor(list(chr_cen.values()))


# C_ref = vectorise_upper_matrix(C_ref, resolution)

#path = f"simulation_little_genome/{origin}/res_{resolution}/noisy/sigma_{sigma_spot}/summary_stat/CNN/"
#path = f"simulation_little_genome/{origin}/res_{resolution}/noisy/sigma_{sigma_spot}/summary_stat/CNN/sigmoid/sequential/"
path = f"simulation_little_genome/{dim}_chr/{origin}/res_{resolution}/noisy/sigma_{sigma_spot}/summary_stat/kernel_5_linear_decay_4_first_hidden_8192/"

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
            
            C = intensity*torch.exp(distr.log_prob(indices))
            
            if noisy:
                mean = intensity * torch.exp(distr.log_prob(torch.tensor([c_i,c_j]))) * 0.1 / 2 
                sigma = intensity * torch.exp(distr.log_prob(torch.tensor([c_i,c_j]))) * 0.1 / 2
       
                noise = mean + sigma*torch.randn((n_row, n_col))

            else:
                noise = torch.zeros_like(C)
            
            return C+noise
        
        C_simu[start_bead[chr_row]:start_bead[chr_row]+nb_bead[chr_row]-1, start_bead[chr_col]:start_bead[chr_col]+nb_bead[chr_col]-1] = simulator_1_bloc(n_row,n_col, c_i_simu, c_j_simu, sig_2_simu, intensity_simu, noisy=noisy)
    
    return C_simu

def vectorise_upper_matrix(C, resolution):
    
    first_bloc = 1
    for (chr_1, chr_2) in combinations(chr_seq.keys(), r=2):
            
        start_1, end_1 = start_bead[chr_1], start_bead[chr_1] + nb_bead[chr_1] #get the start bead id and the end bead id for the chr
        start_2, end_2 = start_bead[chr_2], start_bead[chr_2] + nb_bead[chr_2] #get the start bead id and the end bead id for the chr
        inter_bloc = C[start_1:end_1, start_2:end_2] 
        if first_bloc:
            C_vector = torch.flatten(inter_bloc)
            first_bloc = 0
        else: 
            C_vector = torch.hstack((C_vector, torch.flatten(inter_bloc)))
   
    return C_vector
       
def get_simulations(nb_train):

    theta = prior.sample((nb_train,))
    sigma_spot = "variable"
    noisy = 1
    # C = torch.zeros(nb_train,C_ref.size(0))
    C = torch.zeros(nb_train,C_ref.size(0), C_ref.size(1))
  
    for k in range(nb_train):
       
        C_tmp = simulator(theta[k], resolution, sigma_spot, noisy)
        # C[k] = vectorise_upper_matrix(C_tmp, resolution)
        C[k] = C_tmp

    theta = theta/prior_range #norm thetas

    return theta, C

     
    
def get_dataloaders(
        training_batch_size: int = 200,
        validation_fraction: float = 0.1,
    ):

        theta, C  = get_simulations(nb_train)

        dataset = data.TensorDataset(theta, C)
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
        
        train_loader = data.DataLoader(dataset, **train_loader_kwargs)
        val_loader = data.DataLoader(dataset, **val_loader_kwargs)

        return train_loader, val_loader



def converged(epochs_since_last_improvement, stop_after_epochs):

        converged = False

        # If no validation improvement over many epochs, stop training.
        if epochs_since_last_improvement > stop_after_epochs - 1:
            converged = True

        return converged

def loss(theta, C, DNN):
    
    return torch.mean((DNN(C)-theta)**2, dim=1)
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

if origin=="true":
    C_ref =  np.load(f"ref/{dim}_chr_ref_{resolution}_norm_HiC_duan_intra_all.npy")
    C_ref = torch.from_numpy(C_ref).float() 
else:
    C_ref = simulator(list(chr_cen.values()), resolution, sigma_spot, noisy_ref)

print(origin, C_ref.size())
##################### TRAINING #################################
if 1:

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # input_dim = C_ref.size(0)
    input_dim = (C_ref.size(0), C_ref.size(1))
    output_dim = theta_ref.size(0)
    # hidden_dim = 256
    # DNN = nn.Sequential(nn.Linear(input_dim,hidden_dim), #548 -> 256
    #                      nn.ReLU(),
    #                      nn.Linear(hidden_dim,hidden_dim//4), #256 -> 64
    #                      nn.ReLU(),
    #                      nn.Linear(hidden_dim//4, hidden_dim//4**2), #64 -> 16
    #                      nn.ReLU(),
    #                      nn.Linear(hidden_dim//4**2, output_dim), #16 -> 3
    #                      #nn.ReLU()) 
    #                      nn.Sigmoid()) 

    # DNN = nn.Sequential(nn.Linear(input_dim,hidden_dim), #548 -> 256
    #                      nn.ReLU(),
    #                      nn.Linear(hidden_dim,hidden_dim//2), #256 -> 128
    #                      nn.ReLU(),
    #                      nn.Linear(hidden_dim//2, hidden_dim//4), #128 -> 64
    #                      nn.ReLU(),
    #                      nn.Linear(hidden_dim//4, hidden_dim//8), #64 -> 32
    #                      nn.ReLU(),
    #                      nn.Linear(hidden_dim//8, hidden_dim//16), #32 -> 16
    #                      nn.ReLU(),
    #                      nn.Linear(hidden_dim//16, hidden_dim//32), #16 -> 8
    #                      nn.ReLU(),
    #                      nn.Linear(hidden_dim//32, output_dim), #8 -> 3
    #                      #nn.ReLU())
    #                      nn.Sigmoid())

    decay_linear= 4
    nb_hidden_layers= 6
    kernel_size=5
    pool_kernel_size = 2
    first_hidden_dim = 8192
    out_channels_per_layer = [3,6]
    # DNN0 = torch.nn.ReLU()

    # DNN1 =torch.nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # DNN2 =torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    # DNN3 = torch.nn.Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    #print(C_ref.size())
    #print(DNN1(C_ref.unsqueeze(0)).size())
    # print(DNN0(DNN1(C_ref.unsqueeze(0))).size())
    # print(DNN2(DNN1(C_ref.unsqueeze(0))).size())
    # print(DNN3(DNN2(DNN1(C_ref.unsqueeze(0)))).size())
    #print(DNN2(DNN3(DNN2(DNN1(C_ref.unsqueeze(0))))).size())
    #DNN = CNNEmbedding(input_shape=input_dim, in_channels=1, output_dim=output_dim, num_linear_layers=9, num_linear_units=512, kernel_size=3)
    #DNN = CNNEmbedding(input_shape=input_dim, in_channels=1, output_dim=output_dim, num_linear_layers=9, num_linear_units=512, kernel_size=5)
    #DNN = CNNEmbedding(input_shape=input_dim, in_channels=1, output_dim=output_dim, num_linear_layers=4, num_linear_units=512, kernel_size=3)
    DNN = CNNEmbedding(input_shape=input_dim, in_channels=1, output_dim=output_dim, num_linear_layers=nb_hidden_layers, num_linear_units=first_hidden_dim, kernel_size=kernel_size, pool_kernel_size=pool_kernel_size, out_channels_per_layer=out_channels_per_layer)
    # DNN = CNNEmbedding(input_shape=input_dim, in_channels=1, output_dim=output_dim, num_linear_layers=2, num_linear_units=50, kernel_size=5)
    # DNN = CNNEmbedding(input_shape=input_dim, in_channels=1, output_dim=output_dim, num_linear_layers=2, num_linear_units=50, kernel_size=3)
    DNN = DNN.to(device)
    


    stop_after_epochs = 20
    nb_train=25000

    learning_rate = 5e-4
    max_num_epochs= 2**31 - 1
    # print("data loader")
    train_loader, val_loader = get_dataloaders()
    # print("train loader and val loader done") 

    with open(path+"train_loader", "wb") as f:
         pickle.dump(train_loader, f)
    with open(path+"val_loader", "wb") as f:
         pickle.dump(val_loader, f)
    print("loading data loader")
    with open(path+"train_loader", "rb") as f:
         train_loader = pickle.load(f)
    with open(path+"val_loader", "rb") as f:
         val_loader = pickle.load(f)   

    # Move entire net to device for training.
    DNN.to(device)
  
    

    optimizer = torch.optim.Adam(list(DNN.parameters()), lr=learning_rate)
    epoch, epoch_since_last_improvement, val_loss, best_val_loss = 0, 0, float("Inf"), float("Inf")
    train_loss_list = []
    val_loss_list = []

    while epoch <= max_num_epochs and not converged(epoch_since_last_improvement, stop_after_epochs):
        
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
    plt.show()
    #plt.savefig(path+f"{nb_train}_loss_linear_{decay_linear}_kernel_{kernel_size}.png")
        #plt.savefig(path+f"{nb_train}_loss_linear_{decay_linear}.png")
        
    torch.save(DNN, path+"dnn.pkl")
    with open(path+f"convergence_info_{nb_train}_linear_{decay_linear}_kernel_{kernel_size}.txt", "w") as f:
        #with open(path+f"convergence_info_{nb_train}_linear_{decay_linear}.txt", "w") as f:
            f.write(f"converged after {epoch} epochs with a validation loss {val_loss_list[-1]} \n")
            f.write(f"theta_ref : {theta_ref}, DNN(C_ref) : {DNN(C_ref.unsqueeze(0))*prior_range}\n")
            f.write(f"||theta_ref - DNN(C_ref)|| = {torch.sqrt(torch.mean((DNN(C_ref.unsqueeze(0))*prior_range-theta_ref)**2))}")


if 1:
    
    DNN = torch.load(path+"dnn.pkl", map_location=torch.device('cpu'), weights_only=False)
    print(DNN)
    nb_seq = 10
    ############################## SMCABC -- P. vector based correlation upper all ####################################
    print("ABC")

    #plot_C_genome(C_ref, resolution, 1, 100, chr_cen)
    
    nb_train_ABC= 5000

    if 1:
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
                C_simu = simulator(theta_simu,resolution,  sigma_spot, noisy)
                # C_simu = vectorise_upper_matrix(C_simu, resolution)
                
                theta_dnn[theta_simu] = torch.mean((DNN(C_simu.unsqueeze(0))-DNN(C_ref.unsqueeze(0)))**2).detach()


                plot_C_genome(chr_seq, chr_cen, C_simu, resolution, sig_2_simu, intensity_simu, theta_simu)
        print(theta_ref)       
        print(DNN(C_ref.unsqueeze(0))*prior_range)
        print(torch.sqrt(torch.mean((DNN(C_ref.unsqueeze(0))*prior_range-theta_ref)**2))) 

        # with open(f'{path}0_param', 'wb') as f:
        #     pickle.dump(param, f)

        # with open(f'{path}0_theta_dnn', 'wb') as f:
        #     pickle.dump(theta_dnn, f)

        ################## select good thetas ######################
        prop = 0.05
        start = int(len(theta_dnn)*(1-prop))
        theta_dnn_sorted= dict(sorted(theta_dnn.items(), key=lambda item: item[1], reverse=True)) #sort by values
        thetas_accepted = list(dict(list(theta_dnn_sorted.items())[start:]).keys()) #take theta:corr_inter accepted 
        thetas_accepted = torch.stack(thetas_accepted, dim=0)
        
        with open(f'{path}0_thetas_accepted', 'wb') as f:
                    pickle.dump(thetas_accepted, f)
        ################################################################
        ######## compute mean distance to theta_ref given a range of prop of best thetas (25% to 1%) ########

        eps_dist_min = min(theta_dnn.values())
        print("min dist", eps_dist_min)
      
        theta_dist_sorted = dict(sorted(theta_dnn.items(), key=lambda item: item[1], reverse=True))

        start_stat = int(len(theta_dist_sorted)*0.75) #start with the 25% best theta
  
        eps_dist_start = list(theta_dist_sorted.values())[start_stat]

        eps_dist_list = np.linspace(eps_dist_start,eps_dist_min,100)

        d_mean_list_stat = mean_distance_closest_theta(theta_dnn, theta_ref, eps_dist_list)

        with open(f'{path}0_mean_distance', 'wb') as f:
            pickle.dump(d_mean_list_stat, f)

        #############################################################
        

    for nb_seq in range(nb_seq+1):
       
        print("sequential", nb_seq)
        ############# load thetas accepted set at time 0 ######################
        with open(f'{path}{nb_seq}_thetas_accepted', 'rb') as f:
                thetas_accepted = pickle.load(f)
        #########################################################

        ############## weights #######################
        if nb_seq == 0:
            weights = torch.ones(len(thetas_accepted))*1.0/len(thetas_accepted) #uniform weights
        
        else:
            sigma = resolution
            with open(f'{path}{nb_seq-1}_thetas_accepted', 'rb') as f:
                thetas_t_1 = pickle.load(f)
            with open(f'{path}{nb_seq-1}_weights', 'rb') as f:
                weights_t_1 = pickle.load(f)

            weights = torch.ones(len(thetas_accepted), dtype=torch.float64)
            log_weights = torch.ones(len(thetas_accepted), dtype=torch.float64)
            for i, theta_t in enumerate(thetas_accepted):
            
                log_distr = torch.zeros(len(thetas_t_1))
                
                for j, theta_t_1 in enumerate(thetas_t_1):
                    
                    log_distr[j] = pdist.MultivariateNormal(theta_t_1, sigma**2*torch.eye(dim)).log_prob(theta_t) #log(N(theta_{t-1}, sigma^2 Id))
                log_weight_t_1 = torch.log(weights_t_1) #log_w_t_1
                log_denom = torch.logsumexp(log_weight_t_1 + log_distr, dim=0) # log sum exp (log_w_t_1 + log K)
            
                num = 1.0

                log_weights[i] = -log_denom # w = 1/d, log(w) = - log(d)

            log_norm = torch.logsumexp(log_weights, dim=0) #log sum exp log(w)
            log_weights -= log_norm #w = w / sum(w), log(w) = log(w) - log sum (w)
            weights = torch.exp(log_weights) #w = exp(log(w))
            

        with open(f'{path}{nb_seq}_weights', 'wb') as f:
                pickle.dump(weights, f)
            
        ################################################

        ########### sample new thetas ##################
        prop = 0.05
        with open(f'{path}{nb_seq}_thetas_accepted', 'rb') as f:
                thetas_accepted = pickle.load(f)
        with open(f'{path}{nb_seq}_weights', 'rb') as f:
                weights = pickle.load(f)

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
            theta_out_prior = theta_proposal<torch.ones(dim)  #check in prior
            theta_proposal[theta_out_prior]=thetas_accepted[theta_out_prior].int()

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
            C_simu = simulator(theta_simu,resolution,  sigma_spot, noisy)
            # C_simu = vectorise_upper_matrix(C_simu, resolution)
            
            theta_dnn[theta_simu] = torch.mean((DNN(C_simu.unsqueeze(0))-DNN(C_ref.unsqueeze(0)))**2).detach()
           
        # with open(f'{path}{nb_seq+1}_param', 'wb') as f:
        #             pickle.dump(param, f)
        # with open(f'{path}{nb_seq+1}_theta_dnn', 'wb') as f:
        #             pickle.dump(theta_dnn, f)
        ################## select good thetas ######################
        prop = 0.05
        start = int(len(theta_dnn)*(1-prop))
        theta_dnn_sorted= dict(sorted(theta_dnn.items(), key=lambda item: item[1], reverse=True)) #sort by values
        thetas_accepted = list(dict(list(theta_dnn_sorted.items())[start:]).keys()) #take theta:corr_inter accepted 
        thetas_accepted = torch.stack(thetas_accepted, dim=0)
        
        with open(f'{path}{nb_seq+1}_thetas_accepted', 'wb') as f:
                    pickle.dump(thetas_accepted, f)

        ################################################################
        ######## compute mean distance to theta_ref given a range of prop of best thetas (25% to 1%) ########

        eps_dist_min = min(theta_dnn.values())
        print("min dist", eps_dist_min)
      
        theta_dist_sorted = dict(sorted(theta_dnn.items(), key=lambda item: item[1], reverse=True))

        start_stat = int(len(theta_dist_sorted)*0.75) #start with the 25% best theta
  
        eps_dist_start = list(theta_dist_sorted.values())[start_stat]

        eps_dist_list = np.linspace(eps_dist_start,eps_dist_min,100)

        d_mean_list_stat = mean_distance_closest_theta(theta_dnn, theta_ref, eps_dist_list)

        with open(f'{path}{nb_seq+1}_mean_distance', 'wb') as f:
            pickle.dump(d_mean_list_stat, f)
            
        #############################################################
        