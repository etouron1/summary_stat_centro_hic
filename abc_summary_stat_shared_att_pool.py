
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
import pyro.distributions as pdist

from sbi import utils as utils
#from sbi.neural_nets.embedding_nets import CNNEmbedding
from cnn import CNNEmbedding

import numpy as np
from simulator import get_num_beads_and_start

from itertools import combinations
import random
import pickle
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt

#chr_seq_3_chr_end = {"chr14": 784334, "chr15": 1091290, "chr16": 948063}
#chr_cen_3_chr_end = {'chr14': 628877, 'chr15': 326703, 'chr16': 556070}

chr_seq_3_chr_end = {"chr1": 784334, "chr2": 1091290, "chr3": 948063}
chr_cen_3_chr_end = {'chr1': 628877, 'chr2': 326703, 'chr3': 556070}

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

torch.manual_seed(0)

origin="true"
sigma_spot = "variable"
noisy_ref = 1
sig_2_ref = 1

dim = 3

if dim==3:
    chr_seq = chr_seq_3_chr
    chr_cen = chr_cen_3_chr
    prior_range = torch.tensor([230209, 813179, 316619])
    ### last chr ###
    # chr_seq = chr_seq_3_chr_end
    # chr_cen = chr_cen_3_chr_end
    # prior_range = torch.tensor([784334, 1091290, 948063])
    ###############
    nb_train_ABC=1000
    nb_train_dnn = 5000
     
if dim==16:
    chr_seq = chr_seq_16_chr
    chr_cen = chr_cen_16_chr
    prior_range = torch.tensor([230209, 813179, 316619, 1531918, 
                                576869, 270148, 1090947, 562644,
                                439885, 745746, 666455, 1078176,
                                924430, 784334, 1091290, 948063])
    nb_train_ABC=5000
    #nb_train_dnn = 25000
    nb_train_dnn = 5000
    
resolution = 32000
nb_bead, start_bead, nb_tot_bead = get_num_beads_and_start(chr_seq, resolution)
print(chr_seq)
print(chr_cen)
print(nb_bead, start_bead, nb_tot_bead)
prior = utils.BoxUniform(torch.ones(dim), prior_range-1)

theta_ref = torch.tensor(list(chr_cen.values()))


# C_ref = vectorise_upper_matrix(C_ref, resolution)

#path = f"simulation_little_genome/{origin}/res_{resolution}/noisy/sigma_{sigma_spot}/summary_stat/CNN/"

path = f"simulation_little_genome/{dim}_chr/{origin}/res_{resolution}/noisy/sigma_{sigma_spot}/summary_stat/CNN/sigmoid/sequential/"

# path = f"simulation_little_genome/{dim}_chr/{origin}/res_{resolution}/noisy/sigma_{sigma_spot}/CNN/"

# path = f"simulation_little_genome/{dim}_chr/{origin}/res_{resolution}/noisy/sigma_{sigma_spot}/summary_stat/last_chr/"

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
        ### last chr ###
        # index_row = int(chr_row[chr_row.find("chr")+3:])-14
        # index_col = int(chr_col[chr_col.find("chr")+3:])-14
        ###############
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
       
def get_simulations(nb_train_dnn):

    theta = prior.sample((nb_train_dnn,))
    
    # C = torch.zeros(nb_train_dnn,C_ref.size(0))
    C = torch.zeros(nb_train_dnn,1, C_ref.size(0), C_ref.size(1))
  
    for k in range(nb_train_dnn):
       
        C_tmp = simulator(theta[k], resolution, sigma_spot, noisy_ref).unsqueeze(0)
        # C_tmp = C_tmp / torch.max(C_tmp)
        
        # plt.matshow(C_tmp)
        # plt.show()
        
        # C_tmp = C_tmp + torch.triu(C_tmp).transpose(1,0)

        # C[k] = vectorise_upper_matrix(C_tmp, resolution)
        C[k] = C_tmp

    theta = theta/prior_range #norm thetas

    return theta, C

     
    
def get_dataloaders(
        training_batch_size: int = 200,
        validation_fraction: float = 0.1,
    ):

        theta, C  = get_simulations(nb_train_dnn)

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
    

    # print(DNN(C).size())
    return torch.mean((DNN(C)-theta)**2, dim=1)

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
    
class CNN_FC_Model(nn.Module):
    def __init__(self,input_mlp, output_mlp, decay_mlp = 4, num_conv_layers = 2, in_channels=1, kernel_size=[3,3], pool_kernel_size = [2,2], out_channels_per_layer = [6,12] ):
        super(CNN_FC_Model, self).__init__()
        layers = []
        # CNN subnet
        for k in range(num_conv_layers):
            if k==0:
                
                layers.append(nn.Conv2d(in_channels, out_channels_per_layer[k], kernel_size=kernel_size[k], stride=1, padding=1))  # (0)
                layers.append(nn.ReLU(inplace=True))                                # (1)
                layers.append(nn.MaxPool2d(kernel_size=pool_kernel_size[k], stride=2))
            else:
                
                layers.append(nn.Conv2d(out_channels_per_layer[k-1], out_channels_per_layer[k], kernel_size=kernel_size[k], stride=1, padding=1))  # (0)
                layers.append(nn.ReLU(inplace=True))                               # (1)
                layers.append(nn.MaxPool2d(kernel_size=pool_kernel_size[k], stride=2))
                            
        self.cnn_subnet = nn.Sequential(*layers)
        
        self.pool = AttentionPooling(in_dim=out_channels_per_layer[1])
        input_mlp = out_channels_per_layer[-1]
        
        power = 0
        while decay_mlp**power < input_mlp:
            power += 1
        first_hidden_dim = decay_mlp**(power-1)
        # first_hidden_dim=512
        nb_hidden_layers= power-2

        first_hidden_dim=128
        nb_hidden_layers=4
        
        # Fully connected (linear) subnet
        layers = [nn.Linear(input_mlp, first_hidden_dim), nn.ReLU()]
        # first and last layer is defined by the input and output dimension.
        # therefor the "number of hidden layeres" is num_layers-2
        for k in range(nb_hidden_layers - 2):
            layers.append(nn.Linear(first_hidden_dim//decay_mlp**k, first_hidden_dim//decay_mlp**(k+1))) #//decay_mlp**k //decay_mlp**(k+1)
            layers.append(nn.ReLU())
        
        if nb_hidden_layers !=0:
            layers.append(nn.Linear(first_hidden_dim//decay_mlp**(k+1), output_mlp)) #//decay_mlp**(k+1)
        else:
            layers.append(nn.Linear(first_hidden_dim, output_mlp)) #//decay_mlp**(k+1)
             
        #layers.append(nn.ReLU())
        layers.append(nn.Sigmoid())
        self.mlp_subnet = nn.Sequential(*layers)
    
    def forward(self, x):
        
        x = self.cnn_subnet(x)
        # print(x.size())
        x = self.pool(x)
        # print(x.size())
        # x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.mlp_subnet(x)
        # print(x.size())
        return x
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
    ### last chr ###
    # C_ref =  np.load(f"ref/{dim}_chr_end_ref_{resolution}_norm_HiC_duan_intra_all.npy")
    ################
    C_ref = torch.from_numpy(C_ref).float() 
    # C_ref = C_ref + torch.triu(C_ref).transpose(1,0)

    # C_ref = C_ref/torch.max(C_ref)
    
else:
    C_ref = simulator(list(chr_cen.values()), resolution, sigma_spot, noisy_ref)

print(origin, C_ref.size())
print("nb train ", nb_train_dnn)
plt.matshow(C_ref)
plt.show()
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

    # num_conv_layers = 3
    num_conv_layers = 2
    decay_linear= 4
    # nb_hidden_layers= 6
    kernel_size=3
    # pool_kernel_size = [2,2,4]
    pool_kernel_size = [2,2]
    #out_channels_per_layer = [3,6,12]
    out_channels_per_layer = [6,12]
    # first_hidden_dim = 8192
    # nb_hidden_layers= 6
    input_mlp = input_dim[0]*input_dim[1]
    for i, pool_kernel in enumerate(pool_kernel_size):
        input_mlp /= pool_kernel**2 
        input_mlp = int(input_mlp)
    input_mlp *= out_channels_per_layer[-1]
    
    power = 0
    while 4**power < input_mlp:
        power += 1
    first_hidden_dim = 4**(power-1)
    first_hidden_dim=512
    nb_hidden_layers= power-2

    DNN = CNN_FC_Model(input_mlp, output_mlp=dim)
    print(DNN)

    DNN = DNN.to(device)
    # print(DNN)
    # with open(path+f"dnn_structure.txt", "w") as f:
    #      f.write(str(DNN))


    stop_after_epochs = 20
    

    learning_rate = 5e-4
    max_num_epochs= 2**31 - 1
    print("data loader")
    train_loader, val_loader = get_dataloaders()
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


    optimizer = torch.optim.Adam(list(DNN.parameters()), lr=learning_rate)
    epoch, epoch_since_last_improvement, val_loss, best_val_loss = 0, 0, float("Inf"), float("Inf")
    train_loss_list = []
    val_loss_list = []
    loss_theta_ref = []
    distance_to_theta_train = {key : [] for key in chr_seq.keys()}
    distance_to_theta_val = {key : [] for key in chr_seq.keys()}
    distance_to_theta = {key : [] for key in chr_seq.keys()}
    distance_simu_to_theta = {key : [] for key in chr_seq.keys()}

    # while epoch <= max_num_epochs and not converged(epoch_since_last_improvement, stop_after_epochs):
    while epoch <= 200:
        print("epoch", epoch)
        # Train for a single epoch.
        DNN.train()
        train_loss_sum = 0
        estimation_train = torch.zeros(dim)
        
        for batch in train_loader:
            optimizer.zero_grad()
            # Get batches on current device.
            theta_batch, x_batch,  = (
                batch[0].to(device),
                batch[1].to(device)
            )
            # plt.matshow(x_batch[0,:,:])
            # plt.show()
            
            theta_hat = DNN(x_batch)*prior_range  
         
            estimation_train += torch.mean(torch.abs(theta_hat-theta_batch*prior_range), dim=0)

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
            estimation_val = torch.zeros(dim)

            for batch in val_loader:
                theta_batch, x_batch = (
                    batch[0].to(device),
                    batch[1].to(device),
                    
                )
                # plt.matshow(x_batch[0,:,:])
                # plt.show()
                theta_hat = DNN(x_batch)*prior_range  
                estimation_val += torch.mean(torch.abs(theta_hat-theta_batch*prior_range), dim=0)
                
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
        
        # | theta-DNN(C) | sur données entrainement
        for chr_id in range(1,dim+1):
            
            distance_to_theta_train['chr'+str(chr_id)].append(estimation_train[chr_id-1].item()*1.0/len(train_loader))
            distance_to_theta_val['chr'+str(chr_id)].append(estimation_val[chr_id-1].item()*1.0/len(val_loader))
            ### last chr ###
            # distance_to_theta_train['chr'+str(13+chr_id)].append(estimation_train[chr_id-1].item()*1.0/len(train_loader))
            # distance_to_theta_val['chr'+str(13+chr_id)].append(estimation_val[chr_id-1].item()*1.0/len(val_loader))
            ###############
             
        # plt.matshow(C_ref)
        # plt.show()

        # | theta_ref - DNN(C_ref) |
        C_ref_reshape = C_ref.reshape(1,1,C_ref.size(0), C_ref.size(1))
     
        theta_hat = (DNN(C_ref_reshape)*prior_range).squeeze(0)
        
        
        for chr_id in range(1,dim+1):
            distance_to_theta['chr'+str(chr_id)].append(torch.abs(theta_hat-theta_ref)[chr_id-1].item())
            ### last chr ###
            # distance_to_theta['chr'+str(13+chr_id)].append(torch.abs(theta_hat-theta_ref)[chr_id-1].item())
            #################

        # loss with C_ref
        loss_theta_ref.append(torch.mean((DNN(C_ref_reshape)-theta_ref/prior_range)**2, dim=1).item())
        

        # | theta- DNN(C) |
        distance_simu =0
        for k in range(10):
            
            theta = prior.sample()

            C_tmp = simulator(theta, resolution, sigma_spot, noisy_ref)  
            # fig, ax=plt.subplots(1,2)
            # ax[0].matshow(C_ref)
            # ax[1].matshow(C_tmp)
            # plt.show()
            # plt.matshow(C_tmp)
            # plt.show()
            C_tmp = C_tmp.reshape(1,1,C_tmp.size(0), C_tmp.size(1))
            theta_hat = (DNN(C_tmp)*prior_range).squeeze(0)

            distance_simu += torch.abs(theta_hat-theta)*0.1
            
            
        for chr_id in range(1,dim+1):
            distance_simu_to_theta['chr'+str(chr_id)].append(distance_simu[chr_id-1].item())
            ###last chr ###
            # distance_simu_to_theta['chr'+str(13+chr_id)].append(distance_simu[chr_id-1].item())
            ##############
            


    # Avoid keeping the gradients in the resulting network, which can
    # cause memory leakage when benchmarking.
    DNN.zero_grad(set_to_none=True)
        
    print(f"converged after {epoch} epochs with a validation loss {val_loss_list[-1]}")
    plt.figure()
    plt.title('loss')
    plt.plot(range(epoch), torch.sqrt(torch.tensor(train_loss_list)), label="train")
    plt.plot(range(epoch), torch.sqrt(torch.tensor(val_loss_list)), label="val")
    plt.legend()
    plt.tight_layout()
    plt.show()
    #plt.savefig(path+f"{nb_train}_loss_linear_{decay_linear}_kernel_{kernel_size}.png")
    # plt.savefig(path+f"{nb_train_dnn}_loss_linear_{decay_linear}_kernel_{kernel_size}.pdf")
        #plt.savefig(path+f"{nb_train}_loss_linear_{decay_linear}.png")
    print()
    plt.figure()
    plt.title('loss C_ref')
    plt.plot(range(epoch), torch.sqrt(torch.tensor(loss_theta_ref)))
    plt.legend()
    plt.tight_layout()
    plt.show()
    # plt.savefig(path+f"{nb_train_dnn}_loss_C_ref_linear_{decay_linear}_kernel_{kernel_size}.pdf")


    for num_chr in range(1,dim+1):
    ### last chr ###
    # for num_chr in range(14,dim+14):
    ################
        plt.figure()
        plt.plot(range(epoch), distance_to_theta['chr'+str(num_chr)])
        plt.title('chr'+str(num_chr)+' ref')
        plt.ylabel('|theta_ref - DNN(C_ref)|')
        plt.axhline(y=resolution)
        plt.tight_layout()
        plt.show()
        # plt.savefig(path+f"{nb_train_dnn}_distance_to_theta_chr_{num_chr}.pdf")

    for num_chr in range(1,dim+1):
    ### last chr ###
    # for num_chr in range(14,dim+14):
    ################
        plt.figure()
        plt.plot(range(epoch), distance_simu_to_theta['chr'+str(num_chr)])
        plt.title('chr'+str(num_chr)+' simu')
        plt.ylabel('|theta - DNN(C)|')
        plt.axhline(y=resolution)
        plt.tight_layout()
        plt.show()
        # plt.savefig(path+f"{nb_train_dnn}_distance_simu_to_theta_chr_{num_chr}.pdf")

    for num_chr in range(1,dim+1):
    ### last chr ###
    # for num_chr in range(14,dim+14):
    ################
        plt.figure()
        plt.plot(range(epoch), distance_to_theta_train['chr'+str(num_chr)], label='train')
        plt.plot(range(epoch), distance_to_theta_val['chr'+str(num_chr)], label='val')
        plt.title('chr'+str(num_chr) + ' train/val')
        plt.ylabel('|theta - DNN(C)|')
        plt.axhline(y=resolution)
        plt.legend()
        plt.tight_layout()
        plt.show()
        # plt.savefig(path+f"{nb_train_dnn}_distance_train_val_to_theta_chr_{num_chr}.png")


    print(f"theta_ref : {theta_ref}, DNN(C_ref) : {DNN(C_ref.unsqueeze(0))*prior_range}\n")
    print(f"||theta_ref - DNN(C_ref)|| = {torch.sqrt(torch.mean((DNN(C_ref.unsqueeze(0))*prior_range-theta_ref)**2))}")
    # torch.save(DNN, path+"dnn.pkl")
    # with open(path+f"convergence_info_{nb_train_dnn}_linear_{decay_linear}_kernel_{kernel_size}.txt", "w") as f:
    #     #with open(path+f"convergence_info_{nb_train}_linear_{decay_linear}.txt", "w") as f:
    #         f.write(f"converged after {epoch} epochs with a validation loss {val_loss_list[-1]} \n")
    #         f.write(f"theta_ref : {theta_ref}, DNN(C_ref) : {DNN(C_ref.unsqueeze(0))*prior_range}\n")
    #         f.write(f"||theta_ref - DNN(C_ref)|| = {torch.sqrt(torch.mean((DNN(C_ref.unsqueeze(0))*prior_range-theta_ref)**2))}")


if 0:
    
    DNN = torch.load(path+"dnn.pkl", map_location=torch.device('cpu'), weights_only=False)
    print(DNN)
    nb_seq = 10
    ############################## SMCABC -- P. vector based correlation upper all ####################################
    print("ABC")
    for k in range(dim):
        
        print("theta_ref : ", theta_ref[k].item(), "DNN(C_ref) : ", (DNN(C_ref.unsqueeze(0))[:,k]*prior_range[k]).item() )       
        print("|theta_ref - DNN(C_ref)| =", torch.abs(theta_ref[k] - DNN(C_ref.unsqueeze(0))[:,k]*prior_range[k]).item() )
        print()
    print("||theta_ref - DNN(C_ref)||_2 = ",torch.sqrt(torch.sum((DNN(C_ref.unsqueeze(0)).detach()*prior_range-theta_ref)**2)).item()) 

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
                C_simu = simulator(theta_simu,resolution,  sigma_spot, noisy)
                #C_simu = C_simu + torch.triu(C_simu).transpose(1,0)

                # C_simu = vectorise_upper_matrix(C_simu, resolution)
                
                
                theta_dnn[theta_simu] = torch.mean((DNN(C_simu.unsqueeze(0))-DNN(C_ref.unsqueeze(0)))**2).detach()

                #plot_C_genome(C_simu, resolution, sig_2_simu, intensity_simu, theta_simu)
        print(theta_ref)       
        #print(DNN(C_ref.unsqueeze(0))) 
        print(DNN(C_ref.unsqueeze(0))*prior_range)
        print(torch.sqrt(torch.mean((DNN(C_ref.unsqueeze(0))*prior_range-theta_ref)**2))) 

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
            #C_simu = C_simu + torch.triu(C_simu).transpose(1,0)

            # C_simu = vectorise_upper_matrix(C_simu, resolution)
            
            theta_dnn[theta_simu] = torch.mean((DNN(C_simu.unsqueeze(0))-DNN(C_ref.unsqueeze(0)))**2).detach()
           
        with open(f'{path}{nb_seq+1}_param', 'wb') as f:
                    pickle.dump(param, f)
        with open(f'{path}{nb_seq+1}_theta_dnn', 'wb') as f:
                    pickle.dump(theta_dnn, f)