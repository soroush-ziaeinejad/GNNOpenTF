from torch_geometric.nn import SAGEConv, to_hetero
import torch
import tqdm
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import roc_auc_score
import traintestsplit as tts
def main(data):
    train_data, val_data, test_data = tts.split(data)
    # transform = T.RandomLinkSplit(
    #     num_val=0.2,
    #     num_test=0.0,
    #     disjoint_train_ratio=0.3,
    #     neg_sampling_ratio=2.0,
    #     add_negative_train_samples=False,
    #     edge_types=('team', 'includes', 'expert'),
    #     rev_edge_types=('expert', 'rev_includes', 'team'),
    # )
    # train_data, val_data, test_data = transform(data)
    model = Model(hidden_channels=64, data=train_data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    # print(f"Device: '{device}'")
    model = model.to(device)
    edge_label_index = train_data['team', 'includes', 'expert'].edge_label_index
    edge_label = train_data['team', 'includes', 'expert'].edge_label
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[4, 2],
        neg_sampling_ratio=2.0,
        edge_label_index=(('team', 'includes', 'expert'), edge_label_index),
        edge_label=edge_label,
        batch_size=64,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1, 10):
        total_loss = total_examples = 0
        for sampled_data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            sampled_data.to(device)
            pred = model(sampled_data)
            ground_truth = sampled_data['team', 'includes', 'expert'].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
    # Define the validation seed edges:
    edge_label_index = val_data['team', 'includes', 'expert'].edge_label_index
    edge_label = val_data['team', 'includes', 'expert'].edge_label
    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=[4, 2],
        edge_label_index=(('team', 'includes', 'expert'), edge_label_index),
        edge_label=edge_label,
        batch_size=64,
        shuffle=True,
    )
    preds = []
    ground_truths = []
    for sampled_data in tqdm.tqdm(val_loader):
        with torch.no_grad():
            sampled_data.to(device)
            preds.append(model(sampled_data))
            ground_truths.append(sampled_data['team', 'includes', 'expert'].edge_label)
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    auc = roc_auc_score(ground_truth, pred)
    print(f"Validation AUC: {auc:.4f}")
    return model

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
        edge_feat_team = x_team[edge_index_team_experts[0]]
        edge_feat_expert = x_expert[edge_index_team_experts[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_expert * edge_feat_team).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, data):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        self.skill_emb = torch.nn.Embedding(data['skill'].num_nodes, hidden_channels)
        self.expert_emb = torch.nn.Embedding(data['expert'].num_nodes, hidden_channels)
        self.team_emb = torch.nn.Embedding(data['team'].num_nodes, hidden_channels)
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)
        # Convert GNN model into a heterogeneous variant:
        # self.gnn = to_hetero(self.gnn, metadata=meta)
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()

    def forward(self, data):
        x_dict = {
            "expert": self.expert_emb(data["expert"].node_id),
            "skill": self.skill_emb(data["skill"].node_id),
            "team": self.team_emb(data["team"].node_id)
        }
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(x_dict["expert"], x_dict["team"], data["team", "includes", "expert"].edge_label_index)
        return pred


