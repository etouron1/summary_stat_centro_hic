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
    def __init__(self, chr_seq, chr_cen, dim, resolution, device, global_pool=False, **kwargs):
        # print("init vit")
        
        # super(BioVisionTransformer, self).__init__(**kwargs)
        super().__init__(**kwargs)
        self.depth = kwargs['depth']
        self.chr_seq = chr_seq
        
        self.chr_cen = chr_cen
        self.dim = dim 
        self.resolution = resolution
        self.device = device
        self.nb_bead, self.start_bead, self.nb_tot_bead = get_num_beads_and_start(self.chr_seq, self.resolution)    


        
        #self.linear_proj_to_chr = nn.Linear(self.embed_dim, 1)

        ### version 1 -- 1 projection for all chr from embed dim to 1 ####

        # self.linear_proj_to_chr = nn.Sequential(
        #         nn.Linear(in_features=self.embed_dim, out_features=1),
        #         nn.Sigmoid()
        #     )

        #################

        ##### version 2 -- one mlp per chr #####

        self.linear_proj_to_chr = nn.ModuleList()

        for k in range(1, self.dim+1):
            length_row = self.nb_bead['chr'+str(k)]//self.patch_embed.patch_size[0] + int(self.nb_bead['chr'+str(k)]%self.patch_embed.patch_size[0] !=0)
            
            
            input_dim = length_row * self.patch_embed.img_size[0] //  self.patch_embed.patch_size[0] - length_row*length_row
            input_dim *= self.embed_dim

            power = 0
            while 4**power < input_dim:
                power += 1
            hidden_dim = 4**(power-1)
            nb_layers= power-1
            
            layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            # first and last layer is defined by the input and output dimension.
            # therefor the "number of hidden layeres" is num_layers-2
            for k in range(nb_layers - 2):
                layers.append(nn.Linear(hidden_dim//4**k, hidden_dim//4**(k+1))) #//4**k //4**(k+1)
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim//4**(k+1), 1)) #//4**(k+1)
            #layers.append(nn.ReLU())
            layers.append(nn.Sigmoid())
            self.linear_proj_to_chr.append(nn.Sequential(*layers))           
        
        ###################

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

    # def initialize_weights(self):
    #     print("initialize weights")
    #     # initialization
    #     # initialize (and freeze) pos_embed by sin-cos embedding
    #     from mae_pos_embed import get_2d_sincos_pos_embed
    #     pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
    #     self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
             
        
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # print(self.sincos_pos_embed(self.cls_token))

        # print("1----forward bio vision transformer input----", x.size())
        #x = self.pad_input(self.patch_embed.patch_size[0], x)
        # print('padded', x.size())

        x = self.patch_embed(x) #patch embedding by conv2d
        #print("patch embed output", x.size())
        

        #x, pos_embed = self.patch_embedding(self.patch_embed.patch_size[0], x)
        self.pos_patch(self.patch_embed.patch_size[0]) #corresponding position patch to each patch 4 dim
        # print("avant", x.size(), self.pos_patchs.size())
        x = self.remove_patchs_intra(x) #remove patch and position for intra-chr patch
        # print("après", x.size(), self.pos_patchs.size())
        
        pos_embed = self.sincos_pos_embed(self.pos_patchs) #position embedding

        # pos_embed_cls_token = self.sin_cos_cls_token_embed() #position embedding for class token
        # print(x.size())
        # print(self.cls_token.size())

        # cls_token = self.cls_token.expand(x.shape[0], len(self.chr_seq), -1) #expand cls token to batch size
        
        # x = torch.cat((x, cls_token), dim=1) #concatenate patch with cls token patches
        #(batch, nb_patch + nb_chr, embed dim)
       
        # pos_embed = torch.cat((pos_embed, pos_embed_cls_token), dim=0) #concatenate pos embed with pos embed cls token
        # (nb_patch+nb_chr, embed dim)
        # print("pos_embed", pos_embed.size())

        # pos_embed = self.pos_proj(self.pos_patchs)
        # pos_embed = self.pos_embedding
     
        # print("pos_embed", self.pos_embed.size())
        
        # print("sincos pos embed output", pos_embed.size())
        
        x = self._pos_embed(x, pos_embed) #add patch and position
        #(batch , nb_patch+nb_chr, embed dim)
        # print("pos embed output", x.size())
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
        x = self.proj_to_chr(x)
        # x = self.proj_cls_token_to_centro(x)
        # print(x.size())
        return x

    
    
    
        
    
    # def patchify(self, patch_size, imgs):
    #     """
    #     imgs: (N, 3, H, W)
    #     x: (N, L, patch_size**2 *3)
    #     """
    #     p = patch_size
    #     #assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    #     h = imgs.shape[2] // p
    #     w = imgs.shape[3] // p
    #     c = imgs.shape[1]

    #     pos_patchs = torch.tensor([[i, j] for i in range(h) for j in range(w)]).float()
        
    #     #print("h", h)
    #     x = imgs.reshape(shape=(imgs.shape[0],c , h, p, w, p))
    #     # print("reshape", x.size())
    #     #print(x)
    #     x = torch.einsum('nchpwq->nhwpqc', x)
    #     # print("einsum", x.size())
    #     #print(x)
    #     x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
    #     # print("reshape", x.size())
    #     # print("patchs", x)
    #     # self.patch_proj = nn.Linear(x.size(-1), self.embed_dim)
    #     # x = self.patch_proj(x)
    #     # print("patch proj", x.size())
    #     return x, pos_patchs
    
    # def pad_input(self, patch_size, C):
    #     # print(self.start_bead, self.nb_bead)
    #     # import matplotlib.pyplot as plt
    #     # plt.matshow(C)
    #     # plt.show()
    #     padding_cumul = 0
    #     # new_start_bead = {'chr1':0}
    #     for i, chr in enumerate(list(self.chr_seq.keys())):
             
    #         pos_bloc = self.start_bead[chr] + self.nb_bead[chr] + padding_cumul
            
    #         # print(chr)
    #         # print(pos_bloc)
            

    #         if self.nb_bead[chr]%patch_size !=0:
    #             padding = patch_size-self.nb_bead[chr]%patch_size #nb of 0 to add in the row
    #             padding_cumul += padding
    #             row_to_insert = torch.ones((padding, C.size(0)))*500

    #             # print(C[:pos_bloc].size(), row_to_insert.size(), C[pos_bloc:].size())
    #             C = torch.cat((C[:pos_bloc], row_to_insert, C[pos_bloc:]), dim=0)
    #             col_to_insert = torch.ones((C.size(0), padding))*500
    #             # print(C[:, :pos_bloc].size(), col_to_insert.size(), C[:, pos_bloc:].size())
                
    #             C = torch.cat((C[:, :pos_bloc], col_to_insert, C[:, pos_bloc:]), dim=1)
    #         # if i <15:
    #         #     new_start_bead['chr'+str(i+2)] = self.start_bead['chr'+str(i+2)] + padding_cumul
            

    #     # import matplotlib.pyplot as plt
    #     # plt.matshow(C)
    #     # print(new_start_bead)
    #     # for chr in list(self.chr_seq.keys()):
    #     #      plt.axvline(x=new_start_bead[chr])
    #     #      plt.axhline(y=new_start_bead[chr])
    #     #      plt.axvline(x=self.start_bead[chr], color = 'black', linestyle="--")
    #     #      plt.axhline(y=self.start_bead[chr], color='black', linestyle="--")
    #     # plt.show()
    #     C = C.reshape(1,1,C.size(0), C.size(1))
    #     return C
    
    def pos_patch(self, patch_size):
        pos_bloc = torch.tensor([], dtype=torch.float, device=self.device)
        
        for (i, j) in product(range(self.dim), range(self.dim)):
                # print('chr', i+1,'chr', j+1)
                chr_row = "chr"+str(i+1)
                chr_col = "chr"+str(j+1)
                # print(i,j,chr_row, chr_col)
    
                
                padding_row = padding_col = 0
                if self.nb_bead[chr_row]%patch_size !=0:
                    padding_row = patch_size-self.nb_bead[chr_row]%patch_size #nb of 0 to add in the row
                if self.nb_bead[chr_col]%patch_size !=0:
                    padding_col = patch_size-self.nb_bead[chr_col]%patch_size #nb of 0 to add in the column
                # print("size", self.nb_bead[chr_row], self.nb_bead[chr_col])
                # print("padding", padding_row, padding_col)
                grid_i = torch.arange((self.nb_bead[chr_row] + padding_row)/patch_size, device=self.device)
                grid_j = torch.arange((self.nb_bead[chr_col] + padding_col)/patch_size, device=self.device)
                ii, jj = torch.meshgrid(grid_i, grid_j, indexing='ij')

                # Stack into tuples (i.e., pairs of x, y)
                grid = torch.stack((ii, jj), dim=-1).reshape(-1,2)  # Shape: (11, 11, 2)
                
                # print("grid", grid.size())
                grid_bloc = torch.ones_like(grid)*torch.tensor([i,j])
                grid_bloc.to(self.device)
                
                # print("grid bloc", grid_bloc.size())
                
                grid = torch.cat((grid_bloc, grid), dim = 1)
                
                pos_bloc = torch.cat((pos_bloc, grid), dim=0)
                
                # print(i,j,chr_row, chr_col)
                # print("size", self.nb_bead[chr_row]+padding_row, self.nb_bead[chr_col]+padding_col)
                # print(grid)
                
                
                # print(grid.size())
                # print(grid)
        # print("pos bloc", pos_bloc.size())
        # print(pos_bloc.size())
        _, idx_4 = torch.sort(pos_bloc[:, -2], stable=True)
        pos_bloc = pos_bloc[idx_4]
        _, idx_4 = torch.sort(pos_bloc[:, 0], stable=True)
        pos_bloc = pos_bloc[idx_4]
        # _, idx_4 = torch.sort(pos_bloc[:, 1], stable=True)
        # pos_bloc = pos_bloc[idx_4]
   

        
        # with open("out.txt", "w") as f:
        #     for val in pos_bloc:
        #         for value in val:
        #             f.write(f"{value.item()} ")
        #         f.write('\n')

        # print(pos_bloc)
        self.pos_patchs = pos_bloc
        # return pos_bloc
    
    def sincos_pos_embed(self, grid):
        """
        grid_size: int of the grid height and width
        return:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
        """
 
        omega = torch.arange(self.embed_dim // 8, dtype=torch.float, device=self.device) #list d'entiers de 0 à embed_dim//8-1
        
        omega /= self.embed_dim / 2.
        omega = 1. / 10000**omega  # (D/8,)
        # print("omega", omega.size())
        #grid (N,4))
        # out_final = torch.tensor([])
        # out_i = []
        # for k in range(4):
        #     out_i.append(torch.einsum('m,d->md', grid[:,k], omega))
        # out_final = torch.stack(out_i, dim = -1)
        # print("out_i", out_final)
        # print("out_i", out_final.size())
        out = torch.einsum('ni,d->ndi', grid, omega)  # (M, D/8, 4), outer product
        # print("out", out)
        # print("out", out.size())
        # print(torch.equal(out_final, out))
        emb_sin = torch.sin(out) # (M, D/8, 4)
        emb_cos = torch.cos(out) # (M, D/8, 4)
        # print("sin", emb_sin)
        # print("cos", emb_cos)
        # print("sin", emb_sin.size())
        # print("cos", emb_cos.size())
        emb = torch.concatenate([emb_sin, emb_cos], axis=1)  # (M, D/4, 4)
        # print("emb", emb.size())
        # print("emb", emb)
        # n,m,l = emb.size(0), emb.size(1), emb.size(2)
        # pos_embed = emb.reshape(n, m*l)
        pos_embed = emb.permute(0, 2, 1).reshape(emb.shape[0], -1)

        # print("emb", emb.size())
        # print("emb final", emb)
        # print("pos_embed", pos_embed.size())
        return pos_embed
    
    def sin_cos_cls_token_embed(self):
        grid = torch.arange(len(self.chr_seq),  device=self.device).unsqueeze(-1)
        
        omega = torch.arange(self.embed_dim//2, dtype=torch.float, device=self.device) #list d'entiers de 0 à embed_dim//2-1
        
        omega /= self.embed_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)
        
        out = torch.einsum('ni,d->ndi', grid, omega)  # (M, D/2, 1), outer product
       
        # print("embed dim", self.embed_dim)
        # print("out", out.size())

        emb_sin = torch.sin(out) # (M, D/2, 1)
        emb_cos = torch.cos(out) # (M, D/2, 1)

        # print("sin", emb_sin.size())
        # print("cos", emb_cos.size())

        emb = torch.concatenate([emb_sin, emb_cos], axis=1)  # (M, D, 1)
        # print("emb", emb.size())


        pos_embed = emb.permute(0, 2, 1).reshape(emb.shape[0], -1)
        # print("pos_embed cls_token", pos_embed.size())
        return pos_embed

    
    def remove_patchs_intra(self, x):
        mask = (self.pos_patchs[:, 0] == self.pos_patchs[:, 1])  # select intra blocs size : num_patches
        

        self.pos_patchs = self.pos_patchs[~mask]

        self.patch_embed.num_patches = len(self.pos_patchs) #re-adjust the number of patches

        # self.pos_embedding = torch.arange(self.patch_embed.num_patches, dtype=torch.float)
        # print(self.pos_embedding.size())

        # self.pos_embedding = self.pos_embedding[(~mask)]
        # print(self.pos_embedding.size())
        # print(self.pos_embed.size())
        # self.pos_embedding = self.pos_embed[:,(~mask)]
        # print(self.pos_embedding.size())

      

   
        return x[:,(~mask)]






            
            
         
    
    # def patch_embedding(self, patch_size, imgs):
    #     patchs_list = torch.tensor([])
    #     pos_list = torch.tensor([])
        
    #     #for (i, chr_row), (j, chr_col) in combinations(enumerate(self.chr_seq.keys()), r=2):
    #     #for (i, chr_row), (j, chr_col) in permutations(enumerate(sorted(self.chr_seq.keys())), r=2):
    #     for (i, j) in permutations(range(self.dim), r=2):
    #             chr_row = "chr"+str(i+1)
    #             chr_col = "chr"+str(j+1)
    #             # print(i,j,chr_row, chr_col)
    
                
    #             padding_row = padding_col = 0
    #             if self.nb_bead[chr_row]%patch_size !=0:
    #                 padding_row = patch_size-self.nb_bead[chr_row]%patch_size #nb of 0 to add in the row
    #             if self.nb_bead[chr_col]%patch_size !=0:
    #                 padding_col = patch_size-self.nb_bead[chr_col]%patch_size #nb of 0 to add in the column
    #             # if chr_row=="chr5" or chr_col=="chr5":
    #             #     print(i, j, chr_row, chr_col)
    #             #     print("size", self.nb_bead[chr_row], self.nb_bead[chr_col])
    #             #     nb_patch_row = self.nb_bead[chr_row]//patch_size + int(padding_row!=0)
    #             #     nb_patch_col = self.nb_bead[chr_col]//patch_size + int(padding_col!=0)
    #             #     print("nb patch row", nb_patch_row, "nb padding row", padding_row)
    #             #     print("nb patch col", nb_patch_col, "nb padding col", padding_col)
                    

    #             bloc_padded = torch.zeros(self.nb_bead[chr_row] + padding_row, self.nb_bead[chr_col] + padding_col)
    #             bloc_padded[:self.nb_bead[chr_row], :self.nb_bead[chr_col]] = imgs[self.start_bead[chr_row]:self.start_bead[chr_row]+self.nb_bead[chr_row], self.start_bead[chr_col]:self.start_bead[chr_col]+self.nb_bead[chr_col]]
    #             # plt.matshow(bloc_padded)
    #             # plt.show()
    #             bloc_padded = bloc_padded.reshape(1,1,bloc_padded.size(0), bloc_padded.size(1))
    #             # if chr_row=="chr1" and chr_col=="chr3":
    #             #     print("avant patch", bloc_padded)
    #             patchs, pos_patch = self.patchify(patch_size, bloc_padded) #list of patch + pos du patch dans le bloc
    #             # if chr_row=="chr1" and chr_col=="chr3":
    #             #     print("patch", patchs)
    #             # print("pos patch", pos_patch.size())
                
    #             bloc_pos = torch.ones((pos_patch.size(0), 2))*torch.tensor([i,j])*1.0 #pos du bloc dans la matrix
    #             # print(bloc_pos.size())
                
    #             # print("bloc pos", bloc_pos.size())
    #             pos_patch = torch.cat((bloc_pos, pos_patch), dim=1)
    #             # print("pos patch", pos_patch.size())
    #             patchs_list = torch.cat((patchs_list, patchs), dim=1)
    #             # print("patch list", patchs_list.size())
    #             pos_list = torch.cat((pos_list, pos_patch.unsqueeze(0)), dim=1)
    #             # print("pos list", pos_list.size())

             
    #     patch_proj = nn.Linear(patchs_list.size(-1), self.embed_dim)
        
    #     patchs_list = patch_proj(patchs_list)

        
    #     # print("patch embed", patchs_list.size())
      
    #     return patchs_list, pos_list
    
    def _pos_embed(self, x: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        

        # self.pos_embed = nn.Parameter(torch.zeros(1, x.size(1), self.embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # pos_embed = get_sincos_pos_embed_from_grid(self.embed_dim, pos_embed).float()
        
        # print("fixed pos embed", pos_embed.size())
        # self.pos_embed.data.copy_(pos_embed)
        # print("pos embed", self.pos_embed.size())


        # self.pos_embed = nn.Parameter(torch.randn(1, x.size(1), self.embed_dim) * .02) #1D pos embed to be learned

        # self.pos_proj = nn.Linear(pos_embed.size(-1), self.embed_dim) #proj of pos patch to embed dim
        # print("pos proj", self.pos_proj)
        
        # pos_embed = self.pos_proj(pos_embed.unsqueeze(-1))
        # print(pos_embed.size())
        
        x = x + pos_embed
        
        # print("add pos embed to x", x.size())
        # print("pos embed output", self.pos_drop(x).size())
        # print("original pos ", self.pos_patchs.size())
        return self.pos_drop(x)
    
    # def initialize_weights(self):
    #     print("initialize weights")
    #     # initialization
    #     # initialize (and freeze) pos_embed by sin-cos embedding
    #     pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
    #     self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        

    def proj_to_chr(self, x):
        """ project the patchs corresponding to 1 chr to the centro"""
        
        # print("proj to chr", x.size())
        # centro = torch.zeros((self.dim, self.embed_dim), dtype=torch.float)
        theta = torch.zeros((x.size(0), self.dim), dtype=torch.float)
        for i in range(self.dim):
            #mask = (model.pos_patchs[:, :, 0] == i) | (model.pos_patchs[:, :, 1] == i)  # select row of blocs
            mask = (self.pos_patchs[:, 0] == i)  # select row of blocs
            
            #print(model.pos_patchs[0, mask[0]])
            
            row_blocs = x[:, mask] #extract the row of corresponding patchs
            # print("row blocs", row_blocs.size())

            # print("row blocs", row_blocs.size())

            #### version 1 -- sum patch to have one patch and project it to a 1d number (1 MLP for all chr) #####

            # vector = torch.sum(row_blocs, dim = 1) #aggregate the patchs in row
            # print("chr vector", vector.size())
            # centro = self.linear_proj_to_chr(vector)  #same proj for all chr
            # print('centro', centro.size())   

            ##################

            ###### version 2 -- flatten all patches per chr then project it to a 1d number (1 MLP per chr) ######

            row_blocs_flat = row_blocs.view(row_blocs.size(0), -1) #flatten all the patches
            centro = self.linear_proj_to_chr[i](row_blocs_flat) #a MLP per chr

            ######################

            #chr_proj = nn.Linear(model.embed_dim, 1) #project the embed to a real number
            # print("chr proj", chr_proj)
            #centro = chr_proj(vector)
                  
           
            theta[:,i] = centro.squeeze(-1)
            # centro[i] = vector
        
            # print("centro", centro.size())
        # theta = self.linear_proj_to_chr(centro)
        
        # print("theta_1", theta)

        ######## accelerated version 1 #######
        # mask = (self.pos_patchs[:, 0].unsqueeze(1) == torch.arange(self.dim, device=self.device))  # (num_patches, dim)
        # # Transpose x to shape (batch_size, embed_dim, num_patches) for broadcasting
        # x_transposed = x.transpose(1, 2)  # (batch_size, embed_dim, num_patches)

        # # Apply mask: (batch_size, embed_dim, dim) = (batch_size, embed_dim, num_patches) @ (num_patches, dim)
        # masked_sum = torch.matmul(x_transposed, mask.float())  # sum of patch embeddings for each row index

        # # Transpose back to (batch_size, dim, embed_dim)
        # aggregated = masked_sum.transpose(1, 2)  # (batch_size, dim, embed_dim)

        # # Project each aggregated vector: apply linear projection
        # centro = self.linear_proj_to_chr(aggregated)  # (batch_size, dim, 1)
        # # prendre mlp à partir des patchs en 1 vecteur
        # # Squeeze last dimension to get (batch_size, dim)
        # theta = centro.squeeze(-1)
        # # # print("theta_2", theta)
        #################################
        return theta
    
    def aggregate_per_chr(self, x):
        """ sum all patchs corresponding to 1 chr into 1 patch (for version 3 with residual sum)"""

        vectors = []
        for i in range(self.dim):
           
            mask = (self.pos_patchs[:, 0] == i)  # select row of blocs
            
      
            
            row_blocs = x[:, mask] #extract the row of corresponding patchs
           
            vectors.append(torch.sum(row_blocs, dim = 1)) #aggregate the patchs in row
            # print("chr vector", vector.size())
        return torch.stack(vectors, dim = 0) #list of aggregated patches size : dim
    
    def proj_cls_token_to_centro(self, x):
        cls_token = x[:,-len(self.chr_seq):,:]
        
        theta = self.linear_proj_to_chr(cls_token)
        
        return theta.squeeze(-1)


       
        


 