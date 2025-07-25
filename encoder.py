import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock,VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            #(Batch_Size,Channel,Height,Width)->(Batch_Size,128,Height,Width)
            nn.Conv2d(3,128, kernel_size=3, padding = 1),

            #(Batch_Size,128,Height,Width)->(Batch_Size,128,Height,Width)
            VAE_ResidualBlock(128,128),

            # (Batch_Size,128,Height,Width)->(Batch_Size,128,Height,Width)
            VAE_ResidualBlock(128, 128),

            #(Batch_size,128,Height,Width)->(Batch_Size,128,Height/2,Width/2)
            nn.Conv2d(128,128,kernel_size=3,stride=2,padding=0),

            #(Batch_Size,128,Height/2,Width/2)->(Batch_Size,256,Height/2,Width/2)
            VAE_ResidualBlock(128,256),

            #(Batch_Size,256,Height/2,Width/2)->(Batch_Size,256,Height/2,Width/2)
            VAE_ResidualBlock(256,256),

            #(Batch_Size,256,Height/2,Width/2)-(Batch_Size,256,Height/4,Width/4)
            nn.Conv2d(256,256,kernel_size=3,stride=2,padding=0),

            #(Batch_Size,256,Height/4,Width/4)->(Batch_Size,512,Height/4,Width/4)
            VAE_ResidualBlock(256, 512),

            #(Batch_Size,512,Height/4,Width/4)->(Batch_Size,512,Height/4,Width/4)
            VAE_ResidualBlock(512, 512),

            #(Batch_Size,512,Height/4,Width/4)->(Batch_Size,512,Height/8,Width/8)
            nn.Conv2d(512,512,kernel_size=3,stride=2,padding=0),

            #(Batch_Size,512,Height/8,Width/8)->(Batch_Size,512,Height/8,Width/8)
            VAE_ResidualBlock(512,512),

            # (Batch_Size,512,Height/8,Width/8)->(Batch_Size,512,Height/8,Width/8)
            VAE_ResidualBlock(512,512),

            # (Batch_Size,512,Height/8,Width/8)->(Batch_Size,512,Height/8,Width/8)
            VAE_ResidualBlock(512,512),

            # (Batch_Size,512,Height/8,Width/8)->(Batch_Size,512,Height/8,Width/8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size,512,Height/8,Width/8)->(Batch_Size,512,Height/8,Width/8)
            VAE_AttentionBlock(512),

            nn.GroupNorm(32,512),

            nn.SiLU(), # not sure why, but it works

            #(Batch_Size,512,Height/8,Width/8)->(Batch_Size,8,Height/8,Width/8)
            nn.Conv2d(512,8,kernel_size=3,padding=1),

            #(Batch_Size,8,Height/8,Width/8)->(Batch_Size,8,Height/8,Width/8)
            nn.Conv2d(8,8,kernel_size=1,padding=0)
        )

    def forward(self,x:torch.Tensor,noise:torch.Tensor)->torch.Tensor:
        # x:(Batch_Size,Channel,Height,Width)
        #noise: (Batch_Size,Output_channels,Height/8,Width/8)
        for module in self:
            if getattr(module,'stride',None) == (2,2):
                ## assymetrical padding on right and bottom of image if stride =2
                x = F.pad(x(0,1,0,1))
            x =module(x)

        #(batch_size,8,Height/8,Width/8)-> 2 tensors (batch_size,8,Height/8,Width/8)
        mean,log_variance=torch.chunk(x,2,dim=1)

        #(batch_size,8,Height/8,Width/8)-> (batch_size,8,Height/8,Width/8)
        #function below ensures the log variance stays between the range
        log_variance = torch.clamp(log_variance,-30,20)

        variance = log_variance.exp()

        stdev=variance.sqrt()

        # now we want to sample from this distribution, to a mean and stdev of our choosing
        x = mean + stdev*noise
        # scale it
        x = x*0.18215










