import torch.nn as nn


class ActionEmbeddingEncoder(nn.Module):
    """Encodes discrete actions via embedding lookup.

    Input: actions [B, T] int64 (discrete action indices)
    Output: [B, D, T] float32 (embedded actions matching predictor input format)
    """

    def __init__(self, num_actions=17, embedding_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_actions, embedding_dim)
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim

    def forward(self, actions):
        # actions: [B, T] int64
        embedded = self.embedding(actions)  # [B, T, D]
        return embedded.permute(0, 2, 1)  # [B, D, T]
