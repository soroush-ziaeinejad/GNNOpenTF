import dataPreparation
import model
import torch_geometric.transforms as T

data = dataPreparation.main()


# For this, we first split the set of edges into
# training (80%), validation (10%), and testing edges (10%).
# Across the training edges, we use 70% of edges for message passing,
# and 30% of edges for supervision.
# We further want to generate fixed negative edges for evaluation with a ratio of 2:1.
# Negative edges during training will be generated on-the-fly.
# We can leverage the `RandomLinkSplit()` transform for this from PyG:
# transform = T.RandomLinkSplit(
#     num_val=0.1,
#     num_test=0.1,
#     disjoint_train_ratio=0.3,
#     neg_sampling_ratio=2.0,
#     add_negative_train_samples=False,
#     edge_types=('team', 'includes', 'expert'),
#     rev_edge_types=('expert', 'rev_includes', 'team'),
# )
# train_data, val_data, test_data = transform(data)

model.main(data)