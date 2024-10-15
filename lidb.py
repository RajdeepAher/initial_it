import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List


class myLoRALayer():
    def __init__(
            self,
            r: int,
            lora_a : int,
            lora_b : int,
            lora_alpha : int,
            lora_dropout : float,
            merge_weights : bool,

    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_a = lora_a
        self.lora_b = lora_b

        if lora_dropout > 0. :
            self.lora_dropout = nn.Dropout(p = lora_dropout)
        else:
            self.lora_dropout = lambda x : x # can this be removed?

        self.merged = False
        self.merge_weights = merge_weights

class Embedding(nn.Embedding, myLoRALayer):

    def __init__(
            self,
            num_embeddings : int,
            embedding_dim : int,
            r : int = 0,
            lora_a : int = 0,
            lora_b : int = 0,
            lora_alpha: int = 1,
            merge_weights: bool = True,
            **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        myLoRALayer.__init__(self, r = r, lora_a = lora_a,
                             lora_b=lora_b, lora_alpha=lora_alpha,
                             lora_dropout= 0, merge_weights=merge_weights)
        
        if r > 0 and lora_a > 0 and lora_b > 0:
            self.lora_A_aux = nn.Parameter(self.weight.new_zeros((lora_a, num_embeddings)))
            self.lora_A_train = nn.Parameter(self.weight.new_zeros((r, lora_a)))            
            self.lora_B_aux = nn.Parameter(self.weight.new_zeros((embedding_dim, lora_b)))
            self.lora_B_train = nn.Parameter(self.weight.new_zeros((lora_b, r)))
            self.scaling = self.lora_alpha/self.r
            self.weight.requires_grad = False
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)

        if hasattr(self, 'lora_A_aux'):
            nn.init.normal_(self.lora_A_aux)
            nn.init.normal_(self.lora_B_aux)
            nn.init.zeros_(self.lora_A_train)
            nn.init.zeros_(self.lora_B_train)

#hence forth we will assume that r>0 is equivalent to a,b>0
    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:

                    self.weight.data -= (self.lora_B_aux @ self.lora_B_train @ 
                                         self.lora_A_train @ self.lora_A_aux).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:

                    self.weight.data += (self.lora_B_aux @ self.lora_B_train @ 
                                         self.lora_A_train @ self.lora_A_aux).transpose(0, 1) * self.scaling
                self.merged = True

    
    def forward(self, x:torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            lora_A = self.lora_A_train @ self.lora_A_aux
            after_A = F.embedding(
                x, lora_A.transpose(0,1), self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            )
            result += (after_A @ (self.lora_B_aux @ self.lora_B_train).transpose(0,1))*self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)


class Linear(nn.Linear, myLoRALayer):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_a: int = 0,
            lora_b: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            merge_weights: bool = True,
            **kwargs

    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        myLoRALayer.__init__(self, r=r, lora_a=lora_a, lora_b=lora_b,
                             lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                             merge_weights=merge_weights)
        
        self.fan_in_fan_out = fan_in_fan_out

        if r > 0 and lora_a > 0 and lora_b > 0:
            self.lora_A_aux = nn.Parameter(self.weight.new_zeros((lora_a, in_features)))
            self.lora_A_train = nn.Parameter(self.weight.new_zeros((r, lora_a)))            
            self.lora_B_aux = nn.Parameter(self.weight.new_zeros((out_features, lora_b)))
            self.lora_B_train = nn.Parameter(self.weight.new_zeros((lora_b, r)))
            self.scaling = self.lora_alpha/self.r

            self.weight.requires_grad = False
        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0,1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

        if hasattr(self, 'lora_A_aux'):
            nn.init.normal_(self.lora_A_aux)
            nn.init.normal_(self.lora_B_aux)
            nn.init.zeros_(self.lora_A_train)
            nn.init.zeros_(self.lora_B_train)           

    def train(self, mode:bool = True):
        def T(w):
            return w.transpose(0,1) if self.fan_in_fan_out else w
        
        nn.Linear.train(self,mode)

        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:

                    self.weight.data -= T((self.lora_B_aux @ self.lora_B_train @ 
                                         self.lora_A_train @ self.lora_A_aux)) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:

                    self.weight.data += T((self.lora_B_aux @ self.lora_B_train @ 
                                         self.lora_A_train @ self.lora_A_aux)) * self.scaling
                self.merged = True
    
    def forward(self, x : torch.Tensor):
        def T(w):
            return w.transpose(0,1) if self.fan_in_fan_out else w
        #print("Input:", x[:10])
        #print("Weight requires grad:", self.weight.requires_grad)
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias = self.bias)
            #print("Result after linear:", result[:10])
            lora_output = (self.lora_dropout(x) @
                       (self.lora_A_train @ self.lora_A_aux).transpose(0,1) @
                       (self.lora_B_aux @ self.lora_B_train).transpose(0,1))* self.scaling
            result+=lora_output
            #print("Result after LoRA:", result[:10])
                           # Gradient checking
            # if self.training:
            #        result.register_hook(lambda grad: print(f"Gradient norm in Linear layer: {grad.norm()}"))
            #        lora_output.register_hook(lambda grad: print(f"LoRA output gradient norm: {grad.norm()}"))

            return result
        else:
            return F.linear(x, T(self.weight), bias = self.bias)
        


class MergedLinear(nn.Linear, myLoRALayer):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_a: int = 0,
            lora_b: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            enable_lora: List[bool] = [False],
            fan_in_fan_out: bool = False,
            merge_weights: bool = True,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        myLoRALayer.__init__(self, r=r, lora_a=lora_a, lora_b=lora_b,
                             lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                             merge_weights=merge_weights)
        
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        #A and B are created only when asked for it
        if r > 0 and any(enable_lora):
            self.lora_A_aux = nn.Parameter(
                self.weight.new_zeros((lora_a, in_features))
                #,requires_grad=False
            )
            self.lora_A_train = nn.Parameter(
                self.weight.new_zeros((r*sum(enable_lora), lora_a))
            )
            self.lora_B_aux = nn.Parameter(
                self.weight.new_zeros((out_features//len(enable_lora)*sum(enable_lora),lora_b))
                #,requires_grad=False
            )
            self.lora_B_train = nn.Parameter(
                self.weight.new_zeros((lora_b, r))
            )      

            self.scaling = self.lora_alpha /self.r

            self.weight.requires_grad = False

            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype = torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
    
    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

        if hasattr(self, 'lora_A_aux'):
            nn.init.normal_(self.lora_A_aux)
            nn.init.normal_(self.lora_B_aux)
            nn.init.zeros_(self.lora_A_train)
            nn.init.zeros_(self.lora_B_train) 

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        delta_w = F.conv1d(
            (self.lora_A_train @ self.lora_A_aux).unsqueeze(0),
            (self.lora_B_aux @ self.lora_B_train).unsqueeze(-1),
            group = sum(self.enable_lora)
        ).squeeze(0)

        return T(self.zero_pad(delta_w))
    
    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB() * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling
            return result               


class MyConvLoRA(nn.Module, myLoRALayer):
    def __init__(
            self,
            conv_module,
            in_channels,
            out_channels,
            kernel_size,
            r=0,
            lora_a=0,
            lora_b=0,
            lora_alpha=1,
            lora_dropout=0.,
            merge_weights=True,
            **kwargs
    ):
        super(MyConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        myLoRALayer.__init__(self, r=r, lora_a=lora_a, lora_b=lora_b,
                             lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                             merge_weights=merge_weights)
        
        assert isinstance(kernel_size, int)

        if r > 0:
            self.lora_A_aux = nn.Parameter(
                self.conv.weight.new_zeros((lora_a, in_channels*kernel_size)),
                requires_grad=False
            )
            self.lora_A_train = nn.Parameter(
                self.conv.weight.new_zeros((r*kernel_size, lora_a))
            )
            self.lora_B_aux = nn.Parameter(
                self.conv.weight.new_zeros((out_channels//self.conv.groups*kernel_size, lora_b)),
                requires_grad=False
            )
            self.lora_B_train = nn.Parameter(
                self.conv.weight.new_zeros((lora_b, r*kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A_aux'):
            nn.init.normal_(self.lora_A_aux)
            nn.init.normal_(self.lora_B_aux)
            nn.init.zeros_(self.lora_A_train)
            nn.init.zeros_(self.lora_B_train)

    
    def train(self, mode = True):
        super(MyConvLoRA, self).train(mode)
        
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.lora_B_aux @ self.lora_B_train
                                               @ self.lora_A_train @ self.lora_A_aux).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.lora_B_aux @ self.lora_B_train
                                               @ self.lora_A_train @ self.lora_A_aux).view(self.conv.weight.shape) * self.scaling
                self.merged = True 

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x,
                self.conv.weight + (self.lora_B_aux @ self.lora_B_train
                                               @ self.lora_A_train @ self.lora_A_aux).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        return self.conv(x)

class Conv2d(MyConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)

class Conv1d(MyConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)


