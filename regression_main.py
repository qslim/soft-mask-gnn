import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from models.regression import SMG_JK_R, SMG_R

from utils.config import process_config
from utils.utils import get_args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MyTransform(object):
    def __call__(self, data):
        data.y = data.y[:, int(config.target)]
        return data


def train():
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(data), data.y)
        loss.backward()
        loss_all += loss * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def test(loader):
    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        error += ((model(data) * std[config.target].cuda()) -
                  (data.y * std[config.target].cuda())).abs().sum().item()
    return error / len(loader.dataset)


args = get_args()
config = process_config(args)
print(config)

if config.get('seed') is not None:
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

dicts = {'SMG_JK_R': SMG_JK_R,
         'SMG_R': SMG_R}
net = dicts[config.get('model', 'SMG_JK_R')]

path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'QM9')
dataset = QM9(path, transform=T.Compose([MyTransform(), T.Distance()]))
dataset = dataset.shuffle()

# Normalize targets to mean = 0 and std = 1.
tenpercent = int(len(dataset) * 0.1)
mean = dataset.data.y[tenpercent * 2:].mean(dim=0)
std = dataset.data.y[tenpercent * 2:].std(dim=0)
dataset.data.y = (dataset.data.y - mean) / std

test_dataset = dataset[:tenpercent]
val_dataset = dataset[tenpercent:tenpercent * 2]
train_dataset = dataset[tenpercent * 2:]
test_loader = DataLoader(test_dataset, batch_size=config.hyperparams.batch_size)
val_loader = DataLoader(val_dataset, batch_size=config.hyperparams.batch_size)
train_loader = DataLoader(train_dataset, batch_size=config.hyperparams.batch_size, shuffle=True)

model = net(dataset,
            num_layers=config.hyperparams.layers,
            hidden=config.hyperparams.hidden,
            weight_conv=config.get('weight_conv', 'WeightConv1'),
            multi_channel=config.get('multi_channel', 'False'))
model.to(device).reset_parameters()
optimizer = torch.optim.Adam(model.parameters(), lr=config.hyperparams.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=config.hyperparams.step_size,
                                            gamma=config.hyperparams.decay_rate)

print('--------')
best_val_error = None
for epoch in range(1, config.hyperparams.epochs):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train()
    val_error = test(val_loader)
    scheduler.step()

    if best_val_error is None:
        best_val_error = val_error
    test_error = test(test_loader)
    if val_error <= best_val_error:
        best_val_error = val_error
        print('Epoch: {:03d}, Loss: {:.7f}, Validation MAE: {:.7f}, '
              'Test MAE: {:.7f}'.format(epoch, loss, val_error, test_error))
    else:
        print('Epoch: {:03d}'.format(epoch))

