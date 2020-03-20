import torch
import utils
import torch.nn as nn
import torch.nn.functional as F


class SFNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.user_pathway = config["user_pathway"][:]
        self.item_pathway = config["item_pathway"][:]
        self.combined_pathway = config["combined_pathway"][:]
        self.embedding_dim = config["embedding_dim"]

        self.user_embedding = nn.Embedding(
            num_embeddings=config["num_user_emb"],
            embedding_dim=self.embedding_dim,
            max_norm=1.0,
        )
        self.cup_size_embedding = nn.Embedding(
            num_embeddings=config["num_cup_size_emb"],
            embedding_dim=self.embedding_dim,
            max_norm=1.0,
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=config["num_item_emb"],
            embedding_dim=self.embedding_dim,
            max_norm=1.0,
        )
        self.category_embedding = nn.Embedding(
            num_embeddings=config["num_category_emb"],
            embedding_dim=self.embedding_dim,
            max_norm=1.0,
        )

        # Customer pathway transformation
        # user_embedding_dim + cup_size_embedding_dim + num_user_numeric_features
        user_features_input_size = 2 * self.embedding_dim + config["num_user_numeric"]
        self.user_pathway.insert(0, user_features_input_size)
        self.user_transform_blocks = []
        for i in range(1, len(self.user_pathway)):
            self.user_transform_blocks.append(
                SkipBlock(
                    self.user_pathway[i - 1],
                    self.user_pathway[i],
                    config["activation"],
                )
            )
            self.user_transform_blocks.append(nn.Dropout(p=config["dropout"]))
        self.user_transform_blocks = nn.Sequential(*self.user_transform_blocks)

        # Article pathway transformation
        # item_embedding_dim + category_embedding_dim + num_item_numeric_features
        item_features_input_size = 2 * self.embedding_dim + config["num_item_numeric"]
        self.item_pathway.insert(0, item_features_input_size)
        self.item_transform_blocks = []
        for i in range(1, len(self.item_pathway)):
            self.item_transform_blocks.append(
                SkipBlock(
                    self.item_pathway[i - 1],
                    self.item_pathway[i],
                    config["activation"],
                )
            )
            self.item_transform_blocks.append(nn.Dropout(p=config["dropout"]))
        self.item_transform_blocks = nn.Sequential(*self.item_transform_blocks)

        # Combined top layer pathway
        # u = output dim of user_transform_blocks
        # t = output dim of item_transform_blocks
        # Pathway combination through [u, t, |u-t|, u*t]
        # Hence, input dimension will be 4*dim(u)
        combined_layer_input_size = 4 * self.user_pathway[-1]
        self.combined_pathway.insert(0, combined_layer_input_size)
        self.combined_blocks = []
        for i in range(1, len(self.combined_pathway)):
            self.combined_blocks.append(
                SkipBlock(
                    self.combined_pathway[i - 1],
                    self.combined_pathway[i],
                    config["activation"],
                )
            )
            self.combined_blocks.append(nn.Dropout(p=config["dropout"]))
        self.combined_blocks = nn.Sequential(*self.combined_blocks)

        # Linear transformation from last hidden layer to output
        self.hidden2output = nn.Linear(self.combined_pathway[-1], config["num_targets"])

    def forward(self, batch_input):

        # Customer Pathway
        user_emb = self.user_embedding(batch_input["user_id"])
        cup_size_emb = self.cup_size_embedding(batch_input["cup_size"])
        user_representation = torch.cat(
            [user_emb, cup_size_emb, batch_input["user_numeric"]], dim=-1
        )
        user_representation = self.user_transform_blocks(user_representation)

        # Article Pathway
        item_emb = self.item_embedding(batch_input["item_id"])
        category_emb = self.category_embedding(batch_input["category"])
        item_representation = torch.cat(
            [item_emb, category_emb, batch_input["item_numeric"]], dim=-1
        )
        item_representation = self.item_transform_blocks(item_representation)

        # Combine the pathways
        combined_representation = self.merge_representations(
            user_representation, item_representation
        )
        combined_representation = self.combined_blocks(combined_representation)

        # Output layer of logits
        logits = self.hidden2output(combined_representation)
        pred_probs = F.softmax(logits, dim=-1)

        return logits, pred_probs

    def merge_representations(self, u, v):
        """
        Combining two different representations via:
        - concatenation of the two representations
        - element-wise product u ∗ v
        - absolute element-wise difference |u-v|
        Link: https://arxiv.org/pdf/1705.02364.pdf
        """
        return torch.cat([u, v, torch.abs(u - v), u * v], dim=-1)


class SkipBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation):
        """
        Skip Connection for feed-forward block based on ResNet idea:
        Refer: 
        - Youtube: https://www.youtube.com/watch?v=ZILIbUvp5lk
        - Medium: https://medium.com/@14prakash/understanding-and-implementing-architectures-of-resnet-and-resnext-for-state-of-the-art-image-cf51669e1624 
                 Residual block function when the input and output dimensions are not same.

        """
        super().__init__()
        assert activation in [
            "relu",
            "tanh",
        ], "Please specify a valid activation funciton: relu or tanh"
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
