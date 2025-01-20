# Version 0.0.1

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import sys
import os
from tqdm import tqdm


class Node_Embedding_Model(nn.Module):

    def __init__(
        self, node_in_features: int, node_out_features: int,
        bias: bool, activation: nn.modules.activation, device: str = "cpu",
        h0:int = 8, h1: int = 16, h2: int = 32, h3: int = 64, h4: int = 128, h5 = 256
    ):
        super().__init__()
        self.node_embeding = nn.Sequential(
            nn.Linear(in_features=node_in_features, out_features=h1, bias=bias, device=device),
            activation,
            nn.Linear(in_features=h1, out_features=h2, bias=bias, device=device),
            activation,
            nn.Linear(in_features=h2, out_features=h3, bias=bias, device=device),
            activation,
            nn.Linear(in_features=h3, out_features=h4, bias=bias, device=device),
            activation,
            nn.Linear(in_features=h4, out_features=node_out_features, bias=bias, device=device)
        )
        self.regression = nn.Sequential(
            nn.Linear(in_features=node_out_features, out_features=1, bias=False, device=device),
        )

    def forward(self, node_x):
        node_y = self.node_embeding(node_x).unsqueeze(dim=1)
        y = self.regression(node_y)
        return y
   
class Node_Layer_Embedding_Model(nn.Module):

    def __init__(
        self, node_in_features: int, node_out_features: int,
        layer_in_features: int, layer_out_features: int,        
        bias: bool, activation: nn.modules.activation, device: str = "cpu",
        h0:int = 8, h1: int = 16, h2: int = 32, h3: int = 64, h4: int = 128, h5 = 256
    ):
        super().__init__()
        self.node_embeding = nn.Sequential(
            nn.Linear(in_features=node_in_features, out_features=h1, bias=bias, device=device),
            activation,
            nn.Linear(in_features=h1, out_features=h2, bias=bias, device=device),
            activation,
            nn.Linear(in_features=h2, out_features=h3, bias=bias, device=device),
            activation,
            nn.Linear(in_features=h3, out_features=h4, bias=bias, device=device),
            activation,
            nn.Linear(in_features=h4, out_features=node_out_features, bias=bias, device=device)
        )
        self.layer_embeding = nn.Sequential(
            nn.Linear(in_features=layer_in_features, out_features=h1, bias=bias, device=device),
            activation,
            nn.Linear(in_features=h1, out_features=h2, bias=bias, device=device),
            activation,
            nn.Linear(in_features=h2, out_features=h3, bias=bias, device=device),
            activation,
            nn.Linear(in_features=h3, out_features=h4, bias=bias, device=device),
            activation,
            nn.Linear(in_features=h4, out_features=layer_out_features, bias=bias, device=device)
            )
        
        self.regression = nn.Sequential(
            nn.Linear(in_features=node_out_features, out_features=1, bias=False, device=device),
        )

    def forward(self, node_x, layer_x):
        node_y = self.node_embeding(node_x).unsqueeze(dim=2)
        layer_y = self.layer_embeding(layer_x).unsqueeze(dim=1)
        y = torch.matmul(node_y, layer_y)
        y = torch.mean(y, dim=1)
        y = self.regression(y)
        return y

class Multilayer_Full_Model(nn.Module):

    def __init__(
        self, node_in_features: int, node_out_features: int,
        layer_in_features: int, layer_out_features: int,
        encoder_head: int, num_encoder:int, encoder_activation: str,
        bias: bool, dropout: float,
        activation: nn.modules.activation, device: str = "cpu",
        h0:int = 8, h1: int = 16, h2: int = 32, h3: int = 64, h4: int = 128, h5 = 256
    ):
        super().__init__()
        self.node_embedding = nn.Sequential(
            nn.Linear(in_features=node_in_features, out_features=h1, bias=bias, device=device),
            activation,
            nn.Linear(in_features=h1, out_features=h2, bias=bias, device=device),
            activation,
            nn.Linear(in_features=h2, out_features=h3, bias=bias, device=device),
            activation,
            nn.Linear(in_features=h3, out_features=h4, bias=bias, device=device),
            activation,
            nn.Linear(in_features=h4, out_features=node_out_features, bias=bias, device=device)
        )
       
        self.node_embedding_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=node_out_features, nhead=encoder_head,
                dim_feedforward=(4 * node_out_features), dropout=dropout,
                activation=encoder_activation, bias=bias,
                batch_first=True, device=device),
            1)

        self.layer_embedding = nn.Sequential(
            nn.Linear(in_features=layer_in_features, out_features=h1, bias=bias, device=device),
            activation,
            nn.Linear(in_features=h1, out_features=h2, bias=bias, device=device),
            activation,
            nn.Linear(in_features=h2, out_features=h3, bias=bias, device=device),
            activation,
            nn.Linear(in_features=h3, out_features=h4, bias=bias, device=device),
            activation,
            nn.Linear(in_features=h4, out_features=layer_out_features, bias=bias, device=device)
            )
        
        self.layer_embedding_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=node_out_features, nhead=encoder_head,
                dim_feedforward=(4 * node_out_features), dropout=dropout,
                activation=encoder_activation, bias=bias,
                batch_first=True, device=device),
            1)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=node_out_features, nhead=encoder_head,
                dim_feedforward=(4 * node_out_features), dropout=dropout,
                activation=encoder_activation, bias=bias,
                batch_first=True, device=device),
            num_encoder)
       
        self.regression = nn.Sequential(
            nn.Linear(in_features=node_out_features, out_features=h4, bias=bias, device=device),
            activation,
            nn.Linear(in_features=h4, out_features=h3, bias=bias, device=device),
            activation,
            nn.Linear(in_features=h3, out_features=h2, bias=bias, device=device),
            activation,
            nn.Linear(in_features=h2, out_features=h1, bias=bias, device=device),
            activation,
            nn.Linear(in_features=h1, out_features=h0, bias=bias, device=device),
            activation,
            nn.Linear(in_features=h0, out_features=1, bias=bias, device=device),

        )

    def forward(self, node_x, layer_x):
        node_y = self.node_embedding(node_x)
        node_y = torch.matmul(node_y.unsqueeze(dim=2), node_y.unsqueeze(dim=1))
        node_y = self.node_embedding_encoder(node_y)
        node_y = torch.mean(node_y, dim=1)

        layer_y = self.layer_embedding(layer_x)
        layer_y = torch.matmul(layer_y.unsqueeze(dim=2), layer_y.unsqueeze(dim=1))
        layer_y = self.layer_embedding_encoder(layer_y)
        layer_y = torch.mean(layer_y, dim=1)

        y = torch.matmul(layer_y.unsqueeze(dim=2), node_y.unsqueeze(dim=1))
        y = self.encoder(y)
        y = torch.mean(y, dim=1)
        y = self.regression(y)
        return y