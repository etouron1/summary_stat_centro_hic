# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from itertools import combinations, permutations, product

import timm.models.vision_transformer

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

# def get_sincos_pos_embed_from_grid(embed_dim, pos):
#     """
#     embed_dim: output dimension for each position
#     pos: a list of positions to be encoded: size (M,)
#     out: (M, D)
#     """
#     # print("pos", pos.size())
#     assert embed_dim % 8 == 0
#     #omega = np.arange(embed_dim // 2, dtype=np.float)
#     omega = torch.arange(embed_dim // 8, dtype=float) #list d'entiers de 0 à embed_dimm//4-1
    
#     omega /= embed_dim / 2.
#     omega = 1. / 10000**omega  # (D/4,)
#     # print("omega", omega.size())
    
#     fixed_pos_emb = torch.tensor([])
#     for k in range(pos.size(-1)):
#         # print(pos[:,:,k].size())
#         out = torch.einsum('am,d->amd', pos[:,:,k], omega)  # (1,M, D/8), outer product
#         # print("out", out.size())
#         emb_sin = torch.sin(out) # (1,M, D/8)
#         emb_cos = torch.cos(out) # (1,M, D/8)

#         emb = torch.concatenate([emb_sin, emb_cos], axis=-1)  # (1,M, D/4)
#         fixed_pos_emb = torch.cat((fixed_pos_emb, emb), dim=-1)
#     #     print("emb", emb.size())
#     # print("final pos embed", fixed_pos_emb.size())
#     return fixed_pos_emb

class BioVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    ### for MLP ###
    # def __init__(self, device, nb_bead_chr, length_col,  global_pool=False, **kwargs):
    ##############
    def __init__(self, device,  global_pool=False, **kwargs):
        # print("init biovit")
        
        # super(BioVisionTransformer, self).__init__(**kwargs)
        super().__init__(**kwargs)
        self.depth = kwargs['depth']
        
        self.patch_embed =  nn.Conv2d(kwargs["in_chans"], kwargs["embed_dim"], kernel_size=kwargs["patch_size"], stride=kwargs["patch_size"], bias=True)
        # print("init biovit : patch embed", self.patch_embed)
        self.patch_size = kwargs["patch_size"]
        
        self.device = device 
     

        #self.linear_proj_to_chr = nn.Linear(self.embed_dim, 1)

        ### version 1 -- 1 projection for all chr from embed dim to 1 ####

        # self.linear_proj_to_chr = nn.Sequential(
        #         nn.Linear(in_features=self.embed_dim, out_features=1),
        #         nn.Sigmoid()
        #     )

        #################

        # #### version 2 -- one mlp per chr #####

        
        # length_row = nb_bead_chr//self.patch_size + int(nb_bead_chr%self.patch_size !=0)
        
        
        # input_dim = length_row*length_col*self.embed_dim

        # power = 0
        # while 4**power < input_dim:
        #     power += 1
        # hidden_dim = 4**(power-1)
        # nb_layers= power-1
        
        # layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        # # first and last layer is defined by the input and output dimension.
        # # therefor the "number of hidden layeres" is num_layers-2
        # for k in range(nb_layers - 2):
        #     layers.append(nn.Linear(hidden_dim//4**k, hidden_dim//4**(k+1))) #//4**k //4**(k+1)
        #     layers.append(nn.ReLU())
        # layers.append(nn.Linear(hidden_dim//4**(k+1), 1)) #//4**(k+1)
        # #layers.append(nn.ReLU())
        # layers.append(nn.Sigmoid())
        # #self.linear_proj_to_chr.append(nn.Sequential(*layers))           
        # self.linear_proj_to_chr = nn.Sequential(*layers)          
        
        # ##################

        ##### version 3 -- one projection for all chr (residual sum after each block) #####
        
        # self.aggreg_proj_to_chr = nn.Sequential(
        #     nn.Linear(in_features=self.embed_dim, out_features=1),
        #     nn.Sigmoid()
        # )

        ###################

        ##### version 4 -- one class token per chr #####

        # self.cls_token = nn.Parameter(torch.zeros(1,1, self.embed_dim)) #1,nb_chr,embed_dim
        
        # self.linear_proj_to_chr = nn.Sequential( #projection from embed dim to 1
        #         nn.Linear(in_features=self.embed_dim, out_features=1),
        #         nn.Sigmoid()
        #     ) 

        ###################

        ##### version 5 -- 1 class token for all chr #####

        self.cls_token = nn.Parameter(torch.zeros(1,1, self.embed_dim)) #1,nb_chr,embed_dim
        
        self.linear_proj_to_chr = nn.Sequential( #projection from embed dim to 1
                nn.Linear(in_features=self.embed_dim, out_features=1),
                nn.Sigmoid()
            ) 
         
        # power = 0
        # while 4**power < self.embed_dim:
        #     power += 1
        # hidden_dim = 4**(power-1)
        # nb_layers= power-1
        
        # layers = [nn.Linear(self.embed_dim, hidden_dim), nn.ReLU()]
        # # first and last layer is defined by the input and output dimension.
        # # therefor the "number of hidden layeres" is num_layers-2
        # for k in range(nb_layers - 2):
        #     layers.append(nn.Linear(hidden_dim//4**k, hidden_dim//4**(k+1))) #//4**k //4**(k+1)
        #     layers.append(nn.ReLU())
        # layers.append(nn.Linear(hidden_dim//4**(k+1), 1)) #//4**(k+1)

        # layers.append(nn.Sigmoid())
        # self.linear_proj_to_chr = nn.Sequential( #projection from embed dim to 1
        #         *layers)
        # print(self.linear_proj_to_chr)
        ###################




        
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, self.embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # self.initialize_weights()

        # self.pos_proj = nn.Linear(1, self.embed_dim) #proj of pos patch to embed dim
        # self.pos_proj.weight.requires_grad = False
        # self.pos_proj.bias.requires_grad = False

       

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm   
        # print(self.patch_embed)
        # print("fin init biovit")
       
        
    ############## cut intra #############
    #def forward(self, x: torch.Tensor, nb_bead: dict, chr_row: int) -> torch.Tensor:
    ######################################
    def forward(self, x: torch.Tensor, nb_bead: dict) -> torch.Tensor:
        
        # print(self.sincos_pos_embed(self.cls_token))

        # print("1----forward bio vision transformer input----", x.size())
        #x = self.pad_input(self.patch_embed.patch_size[0], x)
        # print('padded', x.size())
        # print('input forward', x.size())
        x = self.patch_embed(x) #patch embedding by conv2d (batch, embed dim, nb patch row, nb patch col)
        # print("output patch embed", x.size())
        

        #x, pos_embed = self.patch_embedding(self.patch_embed.patch_size[0], x)
        grid = self.pos_patch(x, nb_bead) #corresponding position patch to each patch 4 dim (nb_patchs, nb_pos)
        #print("grid pos", grid.size())
        # print("avant", x.size(), self.pos_patchs.size())
        # x = self.remove_patchs_intra(x) #remove patch and position for intra-chr patch
        # print("après", x.size(), self.pos_patchs.size())
        
        pos_embed = self.sincos_pos_embed(grid) #position embedding (nb_patchs, embed_dim)
        # print("output pos embed", pos_embed.size())


        pos_embed_cls_token = self.sin_cos_cls_token_embed() #position embedding for class token (1, embed_dim)
        # print("output cls token pos embed", pos_embed_cls_token.size())
        
        # print(x.size())
        # print(self.cls_token.size())

        cls_token = self.cls_token.expand(x.shape[0], -1, -1) #expand cls token to batch size (batch,1,embed dim)
        #print("duplicate cls token", cls_token.size())
        # print("cls_token", cls_token.size())
        # print('x', x.size())
        x = x.flatten(2).transpose(1, 2) # (batch, nb_patch, embed_dim)
        # print("patch embed output", x.size())

        # ######## cut intra ########
        # x, pos_embed = self.remove_patchs_intra( x, pos_embed, grid, chr_row)
        # ##########################
        x = torch.cat((x, cls_token), dim=1) #concatenate patch with cls token patches (batch, nb_patch + 1, embed dim)
        
       
        pos_embed = torch.cat((pos_embed, pos_embed_cls_token), dim=0) #concatenate pos embed with pos embed cls token (nb_patch+1, embed dim)
        # print("cat pos_embed cls token", pos_embed.size())

        # # pos_embed = self.pos_proj(self.pos_patchs)
        # # pos_embed = self.pos_embedding
     
        # print("pos_embed", self.pos_embed.size())
        
        # print("sincos pos embed output", pos_embed.size())

        
        
        x = self._pos_embed(x, pos_embed) #add patch and position (batch , nb_patch+1, embed dim)
        # print("add pos_embed to x", x.size())
        # print("patch drop", self.patch_drop)
        x = self.patch_drop(x)
        # print("patch drop output", x.size())
        # print("norm", self.norm_pre)
        x = self.norm_pre(x)
        # print("norm", x.size())
        if self.grad_checkpointing and not torch.jit.is_scripting():
            pass
            #x = checkpoint_seq(self.blocks, x)
        else:
            
            x = self.blocks(x)

            ### version 3 -- residual sum after each block ###
            # for i in range(self.depth):
            #     x = self.blocks[i](x)
            #     if i ==0:
            #         y = self.aggregate_per_chr(x)
                    

            #     else:
            #         y += self.aggregate_per_chr(x) #3,200,64
                    

        # y = self.norm(y) 
        # y = self.aggreg_proj_to_chr(y) #3,200,1
        # y = y.view(y.size(1), y.size(0))
        # return y

        ####################################################
        
        


        x = self.norm(x)
       
        # x = self.proj_to_chr(x)
        x = self.proj_cls_token_to_centro(x) # (batch, 1)
        # print("cls token projection output", x.size())
        return x
    
    def remove_patchs_intra(self, x, pos_embed, grid, chr_row):
        # print("grid", grid.size())
        mask = (grid[:, 0] == chr_row-1)  # select intra blocs size : num_patches
        
        # print("pos", pos_embed.size())
        pos_embed = pos_embed[~mask]
        # print("pos", pos_embed.size())
        # print("x", x.size())
        x = x[:,(~mask)]

        # self.pos_embedding = torch.arange(self.patch_embed.num_patches, dtype=torch.float)
        # print(self.pos_embedding.size())

        # self.pos_embedding = self.pos_embedding[(~mask)]
        # print(self.pos_embedding.size())
        # print(self.pos_embed.size())
        # self.pos_embedding = self.pos_embed[:,(~mask)]
        # print(self.pos_embedding.size())

      

   
        return x, pos_embed


    def pos_patch(self, x, nb_bead):
        pos_bloc = torch.tensor([], dtype=torch.float, device=self.device)
        
        n_patchs_row = x.size(2)
        
        for j, chr_col in enumerate(nb_bead.keys()):
                # print('chr', i+1,'chr', j+1)
               
                #chr_col = "chr"+str(j+1)
                # print(j, chr_col)
    
                padding_col = 0 
                if nb_bead[chr_col]%self.patch_size !=0:
                    padding_col = self.patch_size-nb_bead[chr_col]%self.patch_size #nb of 0 to add in the column
                # print("size", self.nb_bead[chr_row], self.nb_bead[chr_col])
                # print("padding", padding_row, padding_col)
                grid_i = torch.arange(n_patchs_row, dtype=torch.float, device=self.device) #grid i bloc
                grid_j = torch.arange((nb_bead[chr_col] + padding_col)//self.patch_size, dtype=torch.float, device=self.device) #grid j bloc
                ii, jj = torch.meshgrid(grid_i, grid_j, indexing='ij')

                # Stack into tuples (i.e., pairs of x, y)
                grid = torch.stack((ii, jj), dim=-1).reshape(-1,2)  # Shape: (11, 11, 2)
                
                
                #grid_bloc = torch.ones_like(grid)*torch.tensor([j])
                grid_bloc = (torch.ones(grid.size(0))*j).unsqueeze(-1)
                grid_bloc.to(self.device)

                # grid_bloc = grid_bloc.repeat(1,2) #duplicate in to 2 columns
                # print("num bloc", grid_bloc.size())
                
                grid = torch.cat((grid_bloc, grid), dim = 1)
                # print(grid)
                #normalisation des coordonnees
                # grid[:,0] /= len(nb_bead)
                # grid[:,1] /= n_patchs_row
                # grid[:,2] /= (nb_bead[chr_col] + padding_col)//self.patch_size
                # print(grid)
                pos_bloc = torch.cat((pos_bloc, grid), dim=0)
                
                

        _, idx_4 = torch.sort(pos_bloc[:, -2], stable=True)
        pos_bloc = pos_bloc[idx_4]
        # print(pos_bloc)
        # _, idx_4 = torch.sort(pos_bloc[:, 0], stable=True)
        # pos_bloc = pos_bloc[idx_4]
        # print("pos", pos_bloc)
        
        # self.pos_patchs = pos_bloc
        
        return pos_bloc

        # # # print("x", x.size())
        # n_patchs_row = x.size(2)
        # n_patchs_col = x.size(3)
        # # # print('patchs', n_patchs_row, n_patchs_col)
        # grid_i = torch.arange(n_patchs_row, dtype=torch.float32) #liste d'entiers de 0 à n_row-1
        # grid_j = torch.arange(n_patchs_col, dtype=torch.float32) #liste d'entiers de 0 à n_col-1
        # ii, jj = torch.meshgrid(grid_i, grid_j, indexing='ij')

        # # Stack into tuples (i.e., pairs of x, y)
        # grid = torch.stack((ii, jj), dim=-1).reshape(-1,2)  # Shape: (nb_patch, 2)
        # grid = grid.to(self.device)
        # print('grid', grid)
        
        # # print(grid)
        # boundaries = []
        # nb_patch = 0

        # for chr in nb_bead:
        #     nb_patch += nb_bead[chr]//self.patch_size + int(nb_bead[chr]%self.patch_size !=0)
        #     # print("nbpatch", nb_patch)
            
        #     boundaries.append(nb_patch-1)

        
        # # print(grid_j)
        # # print(boundaries)
        # chr_pos = torch.bucketize(grid_j, torch.tensor(boundaries)).unsqueeze(-1) #(nb_patch_col, 1)
        # # print('chr_pos', chr_pos.size())
        # chr_pos = chr_pos.repeat(n_patchs_row,1) #repeat in (nb patch_row*nb_patch_col,1)
        # # pad = torch.zeros_like(chr_pos, device=self.device)
        # # chr_pos = torch.cat((pad, chr_pos), dim=1)
        # # print(chr_pos)
        # # chr_pos = chr_pos.repeat(1,2) #duplicate the coord (nb patch_row*nb_patch_col,2)
        # # print('chr_pos', chr_pos.size())
        # #grid = torch.cat((grid, chr_pos.unsqueeze(-1)), dim=1)
        # grid = torch.cat((chr_pos, grid), dim=1) #(nb_patchs, nb_pos)
        # print("grid", grid.size())
        # return grid
    
    def sincos_pos_embed(self, grid):
        """
        grid_size: int of the grid height and width
        return:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
        """
        #input grid : (nb_patchs, nb_pos)
        # print("grid", grid.size()) 
        omega = torch.arange(self.embed_dim // 6, dtype=torch.float, device=self.device) #list d'entiers de 0 à embed_dim//4-1
        
        omega /= self.embed_dim / 2.
        omega = 1. / 10000**omega  # (embed_dim/4,)
        # print("omega", omega.size())
        #grid (N,4))
        # out_final = torch.tensor([])
        # out_i = []
        # for k in range(4):
        #     out_i.append(torch.einsum('m,d->md', grid[:,k], omega))
        # out_final = torch.stack(out_i, dim = -1)
        # print("out_i", out_final)
        # print("out_i", out_final.size())
        out = torch.einsum('ni,d->ndi', grid, omega)  # (nb_patchs, embed_dim/4, nb_pos), outer product
        # print("out", out)
        # print("out", out.size())
        # print(torch.equal(out_final, out))
        emb_sin = torch.sin(out) # (nb_patchs, embed_dim/4, nb_pos)
        emb_cos = torch.cos(out) # (nb_patchs, embed_dim/4, nb_pos)
        # print("sin", emb_sin)
        # print("cos", emb_cos)
        # print("sin", emb_sin.size())
        # print("cos", emb_cos.size())
        emb = torch.concatenate([emb_sin, emb_cos], axis=1)  # (nb_patchs, embed_dim/2, nb_pos)
        # print("concatenate", emb.size())
        # print("emb", emb)
        # n,m,l = emb.size(0), emb.size(1), emb.size(2)
        # pos_embed = emb.reshape(n, m*l)
        pos_embed = emb.permute(0, 2, 1).reshape(emb.shape[0], -1) # (nb patchs, embed_dim)

        # print("emb", emb.size())
        # print("emb final", emb)
        # print("pos_embed", pos_embed.size())

        return pos_embed 
    
    def sin_cos_cls_token_embed(self):
        grid = torch.ones(1,  device=self.device).unsqueeze(-1) # (1,1)
        # print("grid", grid.size())
        
        omega = torch.arange(self.embed_dim//2, dtype=torch.float, device=self.device) #list d'entiers de 0 à embed_dim//2-1
        
        omega /= self.embed_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)
        
        out = torch.einsum('ni,d->ndi', grid, omega)  # (1, embed_dim/2, 1), outer product
       
        # print("embed dim", self.embed_dim)
        # print("out", out.size())

        emb_sin = torch.sin(out) # (1, embed_dim/2, 1)
        emb_cos = torch.cos(out) # (1, embed_dim/2, 1)

        # print("sin", emb_sin.size())
        # print("cos", emb_cos.size())

        emb = torch.concatenate([emb_sin, emb_cos], axis=1)  # (1, embed_dim, 1)
        # print("emb", emb.size())


        pos_embed = emb.permute(0, 2, 1).reshape(emb.shape[0], -1) # (1,embed_dim)
        # print("pos_embed cls_token", pos_embed.size())
        return pos_embed

    



    
    def _pos_embed(self, x: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        
        x = x + pos_embed
        
        return self.pos_drop(x)
    

      
    
    def proj_cls_token_to_centro(self, x):
        # print(x.size())
        cls_token = x[:,-1:,:] #recup cls token of size (batch,1,embed dim)
        # print("cls token", cls_token.size())
        theta = self.linear_proj_to_chr(cls_token) #project to (batch,1,1)
        # print("theta", theta.size())
        return theta.squeeze(-1)

    def proj_to_chr(self, x):
        """ project the patchs corresponding to 1 chr to the centro"""
        
        
  

        ###### version 2 -- flatten all patches per chr then project it to a 1d number (1 MLP per chr) ######

        x = x.view(x.size(0), -1) #flatten all the patches
        centro = self.linear_proj_to_chr(x) #a MLP per chr

        ######################


                  
           
        theta = centro.squeeze(-1)

        return theta

       
        


 