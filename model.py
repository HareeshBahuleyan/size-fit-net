import torch
import utils
import torch.nn as nn
import torch.nn.functional as F


class SFNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding_dim = config["sfnet"]["embedding_dim"]

        self.item_embedding = nn.Embedding(num_embeddings=config["sfnet"]["num_item_emb"], embedding_dim=self.embedding_dim)
        self.category_embedding = nn.Embedding(num_embeddings=config["sfnet"]["num_category_emb"], embedding_dim=self.embedding_dim)
        self.cup_size_embedding = nn.Embedding(num_embeddings=config["sfnet"]["num_cup_size_emb"], embedding_dim=self.embedding_dim)
        self.user_embedding = nn.Embedding(num_embeddings=config["sfnet"]["num_user_id_emb"], embedding_dim=self.embedding_dim)

        self.dropout = nn.Dropout(p=config["sfnet"]["dropout"])

        assert config["sfnet"]["activation"] in ["relu", "tanh"], "Please specify a valid activation funciton: relu or tanh"
        if config["sfnet"]["activation"] == "relu":
            self.activation = F.relu
        elif config["sfnet"]["activation"] == "tanh":
            self.activation = F.tanh        


    def forward(self, batch_input, target=None):
        pass


class SkipBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation):
        """
        Skip Connection for feed-forward block based on ResNet idea:
        Refer: 
        - Youtube: https://www.youtube.com/watch?v=ZILIbUvp5lk
        - Medium: https://medium.com/@14prakash/understanding-and-implementing-architectures-of-resnet-and-resnext-for-state-of-the-art-image-cf51669e1624 
                 Residual block function when the input and output dimensions are not same.

        """
        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = F.tanh        

        self.inp_transform = nn.Linear(input_dim, output_dim)
        self.out_transform = nn.Linear(output_dim, output_dim)
        self.inp_projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        y = x --> T1(x) --> ReLU(T1(x))
        z = ReLU(T2(y) + Projection(x))
        """
        y = self.activation(self.inp_transform(x))
        z = self.activation(self.out_transform(y) + self.inp_projection(x))
        return z
