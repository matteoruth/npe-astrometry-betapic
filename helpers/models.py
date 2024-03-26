import torch.nn as nn
from torch import Tensor

from lampe.nn import ResMLP
from lampe.inference import NPE

class NPEWithEmbedding(nn.Module): 
    def __init__(self, 
                 num_obs, 
                 embedding_output_dim,
                 embedding_hidden_features,
                 activation,
                 transforms,
                 flow,
                 NPE_hidden_features):
        super().__init__()
        
        self.embedding = nn.Sequential(ResMLP(num_obs * 2,
                                           embedding_output_dim,
                                           hidden_features = embedding_hidden_features,
                                           activation = activation))
        
        self.npe = NPE(7, # The 7 parameters of an orbit
                       embedding_output_dim, 
                       transforms = transforms, 
                       build = flow, 
                       hidden_features = NPE_hidden_features, 
                       activation = activation)
        
    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        return self.npe(theta, self.embedding(x))

    def flow(self, x: Tensor): 
        return self.npe.flow(self.embedding(x))