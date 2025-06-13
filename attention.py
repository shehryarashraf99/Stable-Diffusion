import torch
from torch import nn
from torch.nn import functional as F
import math



## we will have to learn to code transformer architecture to make self attention class
class SelfAttention(nn.Module):
    def __init__(self, n_heads:int, d_embed:int, in_proj_bias=True, out_proj_bias =True):
        super().__init__()
        # we have QKV matrices created together so 3*d_embed. also bias is there if we want it
        self.in_proj=nn.Linear(d_embed,3*d_embed,bias=in_proj_bias)
        self.out_proj=nn.Linear(d_embed,d_embed,bias=out_proj_bias)
        self.n_heads=n_heads
        self.d_heads=d_embed// n_heads

    def forward(self,x:torch.Tensor,causal_mask=False):
        input_shape=x.shape
        batch_size,sequence_length,d_embed =input_shape
        intermim_shape=(batch_size,sequence_length,self.n_heads,self.d_head)

        #(Batch_Size,Seq_Len,Dim)->(Batch_Size,Seq_Len,Dim *3)-> 3
        q,k,v=self.in_proj(x).chunk(3,dim=1)

        #(batch_size,seq_len,dim)->(Batch_size,Seq_Len,H,Dim/H)->(Batch_Size,H,Seq_len,Dim/H)
        q=q.view(intermim_shape).transpose(1,2)
        k=k.view(intermim_shape).transpose(1,2)
        v=v.view(intermim_shape).transpose(1,2)

        weight= q @ k.transpose(-1,-2)

        if causal_mask:
            mask=torch.ones_like(weight,dtype=torch.bool).triu(1)
            weight.masked_fill(mask,-torch.inf)

        weight /=math.sqrt(self.d_head)
        weight=F.softmax(weight,dim=-1)

        output =weight @ v
        output=output.transpose(1,2)

        output =self.out_proj(output)

        return output
    class CrossAttention(nn.Module):
        def __init__(self,n_heads:int,d_embed:int,d_cross:int,in_proj_bias=True,out_proj_bias=True):
            super.__init__()
            self.q_proj=nn.Linear(d_embed,d_embed,bias=in_proj_bias)
            self.k_proj =nn.Linear(d_cross,d_embed,bias=in_proj_bias)
            self.v_proj=nn.Linear(d_cross,d_embed,bias=in_proj_bias)
            self.out_proj=nn.Linear(d_embed,d_embed,bias=out_proj_bias)
            self.n_heads=n_heads
            self.d_head =d_embed //n_heads

        def forward(self,x,y):
            input_shape=x.shape
            batch_size,sequence_length,d_embed=input_shape
            interim_shape=(batch_size,-1,self.n_heads,self.d_head)

            # Multiply query by Wq
            q=self.q_proj(x)
            k=self.k_proj(y)
            v=self.v_proj(y)

            q=q.view(interim_shape.transpose(1,2))
            k=k.view(interim_shape.transpose(1,2))
            v=v.view(interim_shape.transpose(1,2))

            weight=q@k.tranpose(-1,-2)
            weight/=math.sqrt(self.d_head)
            weight=F.softmax(weight,dim=-1)
            output=weight@v
            output=output.transpose(1,2).contiguous()
            output=output.view(input_shape)
            return output
            

            #x:latent space  (Batch_Size,Seq_Len_Q,Dim_Q)
            #y :context (Batch_Size,Seq_Len_KV,Dim KV)









