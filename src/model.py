from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
import torch

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_expert, x_team, edge_index_team_experts):
        # Convert node embeddings to edge-level representations:
        edge_feat_expert = x_expert[edge_index_team_experts[0]]
        edge_feat_team = x_team[edge_index_team_experts[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_expert * edge_feat_team).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(self, in_skills, in_experts, in_teams, meta, hidden_channels):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        self.movie_lin = torch.nn.Linear(20, hidden_channels)
        self.skill_emb = torch.nn.Embedding(data['skills'].num_nodes, hidden_channels)
        self.expert_emb = torch.nn.Embedding(data['experts'].num_nodes, hidden_channels)
        self.team_emb = torch.nn.Embedding(data['teams'].num_nodes, hidden_channels)
        # self.skill_emb = torch.nn.Embedding(in_skills, hidden_channels)
        # self.expert_emb = torch.nn.Embedding(in_experts, hidden_channels)
        # self.team_emb = torch.nn.Embedding(in_teams, hidden_channels)
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)
        # Convert GNN model into a heterogeneous variant:
        # self.gnn = to_hetero(self.gnn, metadata=meta)
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()

    def forward(self, data):
        x_dict = {
            "user": self.user_emb(data["user"].node_id),
            "movie": self.movie_lin(data["movie"].x) + self.movie_emb(data["movie"].node_id),
        }
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["user"],
            x_dict["movie"],
            data["user", "rates", "movie"].edge_label_index,
        )
        return pred


model = Model(hidden_channels=64)