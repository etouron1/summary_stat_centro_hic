import pyro.distributions as pdist
import torch
import pickle
import pywt
import numpy as np
import random
from simulator import simulator_C, plot_C_genome
from simulator import Correlation_inter_upper_average, Correlation_inter_upper_average_wavelets, Correlation_inter_average_row
from simulator import Pearson_correlation_vector,Spearman_correlation_row, Spearman_correlation_col, Spearman_correlation_vector
from simulator import chr_seq_3_chr, chr_cen_3_chr , get_num_beads_and_start
import matplotlib.pyplot as plt


chr_seq_3_chr_end = {"chr14": 784334, "chr15": 1091290, "chr16": 948063}

chr_cen_3_chr_end = {'chr14': 628877, 'chr15': 326703, 'chr16': 556070}

# chr_seq = chr_seq_3_chr_end
# chr_cen = chr_cen_3_chr_end

chr_seq = chr_seq_3_chr
chr_cen = chr_cen_3_chr

# fig, ax = plt.subplots(1,2)
dim = 3
resolution = 32000
# chr_seq = chr_seq_3_chr
# chr_cen = chr_cen_3_chr

C_ref =  np.load(f"ref/{dim}_chr_ref_{resolution}_norm_HiC_duan_intra_all.npy")
plt.matshow(C_ref)
plt.title(r"$\theta_\text{ref}$")
plt.show()

# plot_C_genome(chr_seq, chr_cen , C_ref, resolution, 1, 100, list(chr_cen.values()))
# ax[0].matshow(C_ref)


# C_simu = simulator_C(chr_seq, resolution, chr_cen, 1, intensity = 100, noisy=True)
# plot_C_genome(chr_seq, chr_cen , C_simu, resolution, 1, 100, list(chr_cen.values()))


# ax[1].matshow(C_simu)

# from simulator import Correlation_inter_average_row, Pearson_correlation_row
# # print(Correlation_inter_upper_average(chr_seq_16_chr, C_simu, C_ref, resolution, Pearson_correlation_vector))
# print(Correlation_inter_average_row(chr_seq, C_simu, C_ref, resolution, Pearson_correlation_row))
# plt.show()

torch.manual_seed(0)
if 1:
    ############################## SMCABC -- P. vector based correlation upper all ####################################
    dim_theta = 3
    resolution = 32000
    origin = "synthetic"
    sigma_spot = "variable"
    noisy_ref = 1
    #wavelets = "wavelets"
    wavelets = ""
 
    nb_levels = 3

    if origin=="synthetic":
        sig_2_ref,intensity_ref, noisy = 1, 100,noisy_ref
        C_ref = simulator_C(chr_seq, resolution, chr_cen, sig_2_ref, intensity_ref, noisy_ref)
        plt.matshow(C_ref)
        plt.show()

    if origin=="true":
        # C_ref =  np.load(f"ref/3_chr_end_ref_{resolution}_norm_HiC_duan_intra_all.npy")
        C_ref =  np.load(f"ref/{dim_theta}_chr_ref_{resolution}_norm_HiC_duan_intra_all.npy")
        
        
        if wavelets=="wavelets":  
            #C_ref, (LH, HL, HH) = pywt.dwt2(C_ref, 'bior1.3')
            wavelets_approx_coeff_ref = []
             
            for level in range(nb_levels + 1):
                c_ref = pywt.wavedec2(C_ref, 'db2', mode='symmetric', level=level)
                # normalize each coefficient array independently for better visibility
                wavelets_approx_coeff_ref.append(c_ref[0])
          
    if sigma_spot=="fixe":
            sig_2_simu = 1       
      
    #plot_C_genome(C_ref, resolution, 1, 100, chr_cen)
    nb_train= 1000
    nb_seq = 10
    
    #path = f'simulation_little_genome/{origin}/res_{resolution}/noisy/sigma_{sigma_spot}/sequential/{wavelets}/{nb_levels}_levels/'
    # path = f'simulation_little_genome/{dim_theta}_chr/{origin}/res_{resolution}/noisy/sigma_{sigma_spot}/sequential/last_chr/'
    path = f'simulation_little_genome/{dim_theta}_chr/{origin}/res_{resolution}/noisy/sigma_{sigma_spot}/sequential/new_simulator/'

         

    if 1:
        ######## ABC round 0 ###############################
        
        # theta_corr_row = {}
        # theta_corr_col = {}
        # theta_corr_row_col = {}
        theta_corr_vector = {}
        param = []
        for k in range(nb_train):
                print(k)
                ############# simulate theta ##########
                theta_simu = {}
                for chr in chr_seq.keys():
                    c = pdist.Uniform(low=1, high=chr_seq[chr]-1).sample()
                    theta_simu[chr]=int(c.detach().item())
                ####################################### 
                if sigma_spot=="variable":
                    sig_2_simu = random.uniform(0.1, 10)
                # sig_2_simu = 0.5
                intensity_simu, noisy = 100,noisy_ref
                param.append((sig_2_simu, intensity_simu))
                C_simu = simulator_C(chr_seq, resolution, theta_simu, sig_2_simu, intensity_simu, noisy)
                
                if wavelets=="wavelets":
                    #C_simu, (LH, HL, HH) = pywt.dwt2(C_simu, 'bior1.3')
                    # fig, axes = plt.subplots(4, 4, figsize=[14, 8])
                    # num_chr_num_bead, chr_bead_start, nb_bead = get_num_beads_and_start(chr_seq, resolution)
                    wavelets_approx_coeff_simu = []
                    for level in range(nb_levels + 1):
                        # start_1, end_1 = chr_bead_start["chr01"], chr_bead_start["chr01"] + num_chr_num_bead["chr01"] #get the start bead id and the end bead id for the chr
                        # start_2, end_2 = chr_bead_start["chr02"], chr_bead_start["chr02"] + num_chr_num_bead["chr02"] #get the start bead id and the end bead id for the chr

                        # c_ref = pywt.wavedec2(C_ref, 'db2', mode='symmetric', level=level)
                        # print("c_ref", c_ref[0].shape)
                        # normalize each coefficient array independently for better visibility
                        # c_ref[0] /= np.abs(c_ref[0]).max()
                        # for detail_level in range(level):
                        #     c_ref[detail_level + 1] = [d/np.abs(d).max() for d in c_ref[detail_level + 1]]
                        # # show the normalized coefficients
                        # arr, slices = pywt.coeffs_to_array(c_ref)
                        # arr_sub = arr[0:np.shape(C_ref)[0]//2**level+1,0:np.shape(C_ref)[1]//2**level+1]
                        # axes[0, level].imshow(c_ref[0], cmap="YlOrRd")
                        # axes[0, level].set_title(f'Coefficients\n({level} level)')
                        # axes[0, level].set_axis_off()
                        # bloc = c_ref[0][start_1//2**level+1:end_1//2**level+1, start_2//2**level+1:end_2//2**level+1] 
                        # axes[0, level].imshow(c_ref[0], cmap="YlOrRd")
                        # axes[1, level].imshow(bloc, cmap="YlOrRd")

                        c_simu = pywt.wavedec2(C_simu, 'db2', mode='symmetric', level=level)
                        wavelets_approx_coeff_simu.append(c_simu[0])
                        # print("c_simu", c_simu[0].shape)
                        # normalize each coefficient array independently for better visibility
                    #     c_simu[0] /= np.abs(c_simu[0]).max()
                    #     for detail_level in range(level):
                    #         c_simu[detail_level + 1] = [d/np.abs(d).max() for d in c_simu[detail_level + 1]]
                    #     # show the normalized coefficients
                    #     arr, slices = pywt.coeffs_to_array(c_simu)
                    #     arr_sub = arr[0:np.shape(C_simu)[0]//2**level+1,0:np.shape(C_simu)[1]//2**level+1]

                        
                        # bloc = c_simu[0][start_1//2**level+1:end_1//2**level+1, start_2//2**level+1:end_2//2**level+1] 
                        # axes[2, level].imshow(c_simu[0], cmap="YlOrRd")
                        # axes[3, level].imshow(bloc, cmap="YlOrRd")
                        # axes[1, level].imshow(c_simu[0], cmap="YlOrRd")
                    #     axes[1, level].set_title(f'Coefficients\n({level} level)')
                    #     axes[1, level].set_axis_off()
                    # plt.tight_layout()
                    # plt.show()
                    theta_corr_vector[tuple(theta_simu.values())] = Correlation_inter_upper_average_wavelets(wavelets_approx_coeff_simu, wavelets_approx_coeff_ref, resolution, Pearson_correlation_vector)  
                else:
                    theta_corr_vector[tuple(theta_simu.values())] = Correlation_inter_upper_average(chr_seq, C_simu, C_ref, resolution, Pearson_correlation_vector)  
                     

                #plot_C_genome(C_simu, resolution, sig_2_simu, intensity_simu, theta_simu)
                # theta_corr_row[tuple(theta_simu.values())] = Correlation_inter_upper_average(C_simu, C_ref, resolution, Spearman_correlation_row)  
                # theta_corr_col[tuple(theta_simu.values())] = Correlation_inter_upper_average(C_simu, C_ref, resolution, Spearman_correlation_col)  
                # theta_corr_row_col[tuple(theta_simu.values())] = 0.5*(theta_corr_row[tuple(theta_simu.values())]+theta_corr_col[tuple(theta_simu.values())])  
                # theta_corr_vector[tuple(theta_simu.values())] = Correlation_inter_upper_average(C_simu, C_ref, resolution, Pearson_correlation_vector)  

        with open(f'{path}0_param', 'wb') as f:
            pickle.dump(param, f)
        # with open(f'{path}0_theta_S_corr_inter_row', 'wb') as f:
        #     pickle.dump(theta_corr_row, f)
        # with open(f'{path}0_theta_S_corr_inter_col', 'wb') as f:
        #     pickle.dump(theta_corr_col, f)
        # with open(f'{path}0_theta_S_corr_inter_row_col', 'wb') as f:
        #     pickle.dump(theta_corr_row_col, f)
        with open(f'{path}0_theta_P_corr_inter_vector', 'wb') as f:
            pickle.dump(theta_corr_vector, f)

    for nb_seq in range(nb_seq+1):
        print("sequential", nb_seq)
        ############# load train set at time 0 ######################
        with open(f'{path}{nb_seq}_theta_P_corr_inter_vector', 'rb') as f:
                theta_corr = pickle.load(f)
        #########################################################

        ################## select good thetas ######################
        prop = 0.05
        start = int(len(theta_corr)*(1-prop))
        theta_corr_sorted= dict(sorted(theta_corr.items(), key=lambda item: item[1])) #sort by values
        thetas_accepted = list(dict(list(theta_corr_sorted.items())[start:]).keys()) #take theta:corr_inter accepted 
        thetas_accepted = torch.tensor(thetas_accepted)

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
            weights_stab = torch.ones(len(thetas_accepted), dtype=torch.float64)
            log_weights = torch.ones(len(thetas_accepted), dtype=torch.float64)
            for i, theta_t in enumerate(thetas_accepted):
                denom = 0
                log_distr = torch.zeros(len(thetas_t_1))

                for j, theta_t_1 in enumerate(thetas_t_1):
                    distr = pdist.MultivariateNormal(theta_t_1, sigma**2*torch.eye(dim_theta)) #N(theta_{t-1}, sigma^2 Id)
                    perturb_kernel = torch.exp(distr.log_prob(theta_t)) 
                    weight_t_1 = weights_t_1[j] #w_{t-1} (theta_{t-1})
                    denom += weight_t_1*perturb_kernel

                    log_distr[j] = pdist.MultivariateNormal(theta_t_1, sigma**2*torch.eye(dim_theta)).log_prob(theta_t) #N(theta_{t-1}, sigma^2 Id)
                log_weight_t_1 = torch.log(weights_t_1)
                log_denom = torch.logsumexp(log_weight_t_1 + log_distr, dim=0)
               
                # volume = torch.prod(torch.tensor(list(chr_seq.values()))-1)
                # num = 1.0/volume #prior
                num = 1.0
                weights[i] = num/denom

                log_weights[i] = -log_denom

            norm = weights.sum()
            weights /= norm

            log_norm = torch.logsumexp(log_weights, dim=0)
            log_weights -= log_norm
            weights_stab = torch.exp(log_weights)
            # print(weights)
            # print(weights_stab)
            
        with open(f'{path}{nb_seq}_weights', 'wb') as f:
                pickle.dump(weights, f)
            
        ################################################

        ########### sample new thetas ##################
        new_id = torch.multinomial(weights,len(thetas_accepted), replacement=True) #sample from {thetas, weights}
        thetas_accepted = thetas_accepted[new_id]

        sigma = resolution #perturb the thetas
        perturb_dist = pdist.MultivariateNormal(torch.zeros(dim_theta), torch.eye(dim_theta)) #N(0,Id)
        nb_run = int(1 / prop)
        theta = torch.zeros((nb_train, dim_theta), dtype=int) #(1000, 3)
        for k in range(1, nb_run+1): #create 1000 thetas from thetas accepted with perturbation
            perturb_eps = perturb_dist.sample((len(thetas_accepted),))
            theta_proposal = (thetas_accepted+sigma*perturb_eps).int() #theta + sigma*N(0, Id)
            
            theta_out_prior = theta_proposal>torch.tensor(list(chr_seq.values()))  #check in prior
            theta_proposal[theta_out_prior]=thetas_accepted[theta_out_prior].int() #if out prior : take thetas accepted

            theta[int((k-1)*nb_train*prop):int(k*nb_train*prop)] = theta_proposal
            

        #################################################
    
        theta_corr_vector = {}
        param = []
        for k, t in enumerate(theta):
            print(k)
            ##### {chr : theta} #####
            theta_simu = {}
            for i, chr in enumerate(chr_seq.keys()):
                theta_simu[chr] = t[i]
            #########################
         
            if sigma_spot=="variable":
                sig_2_simu = random.uniform(0.1, 10)
            # sig_2_simu = 0.5
            intensity_simu, noisy = 100,noisy_ref
            param.append((sig_2_simu, intensity_simu))

            C_simu = simulator_C(chr_seq, resolution, theta_simu, sig_2_simu, intensity_simu, noisy)

            if wavelets=="wavelets":
                #C_simu, (LH, HL, HH) = pywt.dwt2(C_simu, 'bior1.3')
                wavelets_approx_coeff_simu = []
                for level in range(nb_levels + 1):
                    c_simu = pywt.wavedec2(C_simu, 'db2', mode='symmetric', level=level)
                    wavelets_approx_coeff_simu.append(c_simu[0])

                theta_corr_vector[tuple(theta_simu.values())] = Correlation_inter_upper_average_wavelets(chr_seq, wavelets_approx_coeff_simu, wavelets_approx_coeff_ref, resolution, Pearson_correlation_vector)  
            else:

            # plot_C_genome(C_simu, resolution, sig_2_simu, intensity_simu, theta_simu)
            
                theta_corr_vector[tuple(theta_simu.values())] = Correlation_inter_upper_average(chr_seq, C_simu, C_ref, resolution, Pearson_correlation_vector)

        with open(f'{path}{nb_seq+1}_param', 'wb') as f:
                    pickle.dump(param, f)
        with open(f'{path}{nb_seq+1}_theta_P_corr_inter_vector', 'wb') as f:
                    pickle.dump(theta_corr_vector, f)

    #########################################################################################################

      
if 0:
    ############################## SMCABC -- P. vector based correlation row each chr ####################################
    dim_theta = 3
    resolution = 32000
    C_ref =  np.load(f"ref/3_chr_ref_{resolution}_norm_HiC_duan_intra_all.npy")

    nb_train= 1000
    nb_seq = 0
    path = f'simulation_little_genome/true/res_{resolution}/noisy/sigma_fixe/sequential/'
    global theta_corr_dict

    if 0:
        ######## ABC round 0 ###############################
        
        theta_corr_dict = {}
        for chr in chr_seq.keys():
            theta_corr_dict[chr] = {}
            
        
        #param = []
        for k in range(1000):
                print(k)
                ############# simulate theta ##########
                theta_simu = {}
                for chr in chr_seq.keys():
                    c = pdist.Uniform(low=1, high=chr_seq[chr]-1).sample()
                    theta_simu[chr]=int(c.detach().item())
                    
                ####################################### 
                #sig_2_simu = random.uniform(0.1, 10)
                sig_2_simu = 1
                intensity_simu, noisy = 100,1
                #param.append((sig_2_simu, intensity_simu))
                C_simu = simulator_C(chr_seq, resolution, theta_simu, sig_2_simu, intensity_simu, noisy)
                #plot_C_genome(C_simu, resolution, sig_2_simu, intensity_simu, theta_simu)
                
                Correlation_inter_average_row(C_simu, C_ref, resolution, Pearson_correlation_vector, theta_simu, theta_corr_dict)
                
        # with open(f'{path}0_param', 'wb') as f:
        #     pickle.dump(param, f)

        with open(f'{path}0_theta_P_corr_inter_vector_per_row', 'wb') as f:
            pickle.dump(theta_corr_dict, f)

    for nb_seq in range(3):
        print("sequential", nb_seq)
        ############# load train set at time 0 ######################
        with open(f'{path}{nb_seq}_theta_P_corr_inter_vector_per_row', 'rb') as f:
                theta_corr_dict = pickle.load(f)
        #########################################################

        ################## select good thetas ######################
        
        prop = 0.05
        nb_sel = int(prop*nb_train)
        thetas_accepted = torch.zeros((nb_sel, len(chr_seq)))

        for k, chr in enumerate(chr_seq.keys()):
                start_chr = len(theta_corr_dict[chr])-nb_sel
                
                chr_corr_sorted= dict(sorted(theta_corr_dict[chr].items(), key=lambda item: item[1])) #sort by values
                chr_accepted = list(dict(list(chr_corr_sorted.items())[start_chr:]).keys()) #take theta:corr_inter accepted 
                chr_accepted = torch.tensor(chr_accepted)
                thetas_accepted[:,k] = chr_accepted
       

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
                    distr = pdist.MultivariateNormal(theta_t_1, sigma**2*torch.eye(dim_theta)) #N(theta_{t-1}, sigma^2 Id)
                    perturb_kernel = torch.exp(distr.log_prob(theta_t)) 
                    weight_t_1 = weights_t_1[j] #w_{t-1} (theta_{t-1})
                    denom += weight_t_1*perturb_kernel

                # volume = torch.prod(torch.tensor(list(chr_seq.values()))-1)
                # num = 1.0/volume #prior
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
        perturb_dist = pdist.MultivariateNormal(torch.zeros(dim_theta), torch.eye(dim_theta)) #N(0,Id)
        nb_run = int(1 / prop)
        theta = torch.zeros((nb_train, dim_theta), dtype=int) #(1000, 3)
        for k in range(1, nb_run+1): #create 1000 thetas from thetas accepted with perturbation
            perturb_eps = perturb_dist.sample((len(thetas_accepted),))
            theta_proposal = (thetas_accepted+sigma*perturb_eps).int() #theta + sigma*N(0, Id)
            
            theta_out_prior = theta_proposal>torch.tensor(list(chr_seq.values()))  #check in prior
            theta_proposal[theta_out_prior]=thetas_accepted[theta_out_prior].int() #if out prior : take thetas accepted

            theta[int((k-1)*nb_train*prop):int(k*nb_train*prop)] = theta_proposal
            

        #################################################
        theta_corr_dict = {}
        for chr in chr_seq.keys():
            theta_corr_dict[chr] = {}

        #param = []
        for k, t in enumerate(theta):
            print(k)
            ##### {chr : theta} #####
            theta_simu = {}
            for i, chr in enumerate(chr_seq.keys()):
                theta_simu[chr] = t[i]
            #########################
            #sig_2_simu = random.uniform(0.1, 10)
            sig_2_simu = 1
            intensity_simu, noisy = 100,1
            #param.append((sig_2_simu, intensity_simu))
            C_simu = simulator_C(chr_seq, resolution, theta_simu, sig_2_simu, intensity_simu, noisy)
            # plot_C_genome(C_simu, resolution, sig_2_simu, intensity_simu, theta_simu)

            Correlation_inter_average_row(C_simu, C_ref, resolution, Pearson_correlation_vector, theta_simu, theta_corr_dict)
    
            
        # with open(f'{path}{nb_seq+1}_param', 'wb') as f:
        #             pickle.dump(param, f)
  
        with open(f'{path}{nb_seq+1}_theta_P_corr_inter_vector_per_row', 'wb') as f:
                    pickle.dump(theta_corr_dict, f)