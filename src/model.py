from torch_geometric.nn import SAGEConv, to_hetero
import torch
import tqdm
import numpy as np
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import roc_auc_score
import traintestsplit as tts
def main(data, dataset_name):
    # try:
    #     train_data = torch.load(f'../output/NewSplitMethod/{dataset_name}/train.pt')
    #     val_data = torch.load(f'../output/NewSplitMethod/{dataset_name}/val.pt')
    #     # test_data = torch.load(f'../output/NewSplitMethod/{dataset_name}/test.pt')
    #     print('splitted data loaded')
    # except:
    #     print('splitting data')
    #     train_data, val_data, test_data = tts.split(data, dataset_name,valid_ratio=0.3,test_ratio=0.0)
    transform = T.RandomLinkSplit(
        num_val=0.2,
        num_test=0.1,
        is_undirected=True,
        disjoint_train_ratio=0.0,
        neg_sampling_ratio=1.0,
        add_negative_train_samples=False,
        edge_types=('team', 'includes', 'expert'),
        rev_edge_types=('expert', 'rev_includes', 'team'),
    )
    train_data, val_data, test_data = transform(data)
    model = Model(hidden_channels=4, data=train_data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    # print(f"Device: '{device}'")
    model = model.to(device)
    edge_label_index = train_data['team', 'includes', 'expert'].edge_label_index
    edge_label = train_data['team', 'includes', 'expert'].edge_label
    train_loader = LinkNeighborLoader(
        data=train_data,
        # num_neighbors=[4, 2],
        num_neighbors={key: [-1] for key in train_data.edge_types},
        neg_sampling_ratio=0.0,
        edge_label_index=(('team', 'includes', 'expert'), edge_label_index),
        edge_label=edge_label,
        batch_size=64,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    edge_label_index = val_data['team', 'includes', 'expert'].edge_label_index
    edge_label = val_data['team', 'includes', 'expert'].edge_label
    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors={key: [-1] for key in train_data.edge_types},
        neg_sampling_ratio=1.0,
        edge_label_index=(('team', 'includes', 'expert'), edge_label_index),
        edge_label=edge_label,
        batch_size=64,
        shuffle=True,
    )


    for epoch in range(1, 5):
        total_loss = total_examples = 0
        # train_preds = []
        # train_ground_truths = []
        for sampled_data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            sampled_data.to(device)
            pred = model(sampled_data)
            # train_preds.append(pred)
            ground_truth = sampled_data['team', 'includes', 'expert'].edge_label
            # train_ground_truths.append(ground_truth)
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        print(f"Epoch: {epoch:03d}, Training Loss: {total_loss / total_examples:.4f}")
        # pred = torch.cat(train_preds, dim=0).cpu().detach().numpy()
        # ground_truth = torch.cat(train_ground_truths, dim=0).cpu().numpy()
        # auc = roc_auc_score(ground_truth, pred)
        # print(f"Train AUC Epoch {epoch}: {auc:.4f}")
        val_preds = []
        val_ground_truths = []
        val_total_loss = val_total_examples = 0
        for sampled_data in tqdm.tqdm(val_loader):
            with torch.no_grad():
                sampled_data.to(device)
                pred = model(sampled_data)
                val_preds.append(pred)
                ground_truth = sampled_data['team', 'includes', 'expert'].edge_label
                val_ground_truths.append(ground_truth)
                loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
                val_total_loss += float(loss) * pred.numel()
                val_total_examples += pred.numel()
        print(f"Epoch: {epoch:03d}, Validation Loss: {val_total_loss / val_total_examples:.4f}")
        pred = torch.cat(val_preds, dim=0).cpu().numpy()
        ground_truth = torch.cat(val_ground_truths, dim=0).cpu().numpy()
        np.save(f'../output/NewSplitMethod/{dataset_name}/pred_{epoch}.npy', pred)
        np.save(f'../output/NewSplitMethod/{dataset_name}/ground_truth_{epoch}.npy', ground_truth)
        auc = roc_auc_score(ground_truth, pred)
        print(f"Validation AUC Epoch {epoch}: {auc:.4f}")
    # Define the validation seed edges:



    # auc = roc_auc_score(ground_truth, pred)
    # print(f"Validation AUC: {auc:.4f}")
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
    def forward(self, x_expert, x_team, edge_label_index_team_experts):
        # Convert node embeddings to edge-level representations:
        edge_feat_team = x_team[edge_label_index_team_experts[0]]
        edge_feat_expert = x_expert[edge_label_index_team_experts[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_expert * edge_feat_team).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, data):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices:
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


