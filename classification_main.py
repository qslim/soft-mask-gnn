import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from models.classification import SMG, SMG_JK, SMG_2h, SMG_2h_JK
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os.path as osp
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T

from utils.config import process_config
from utils.utils import get_args


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def get_dataset(name):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', name)
    dataset = TUDataset(path, name)
    dataset.data.edge_attr = None

    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    return dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(loader, model, optimizer):
    model.train()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(loader.dataset)


def test(loader, model):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


def run_given_fold(net,
                   dataset,
                   train_loader,
                   val_loader,
                   config):
    model = net(dataset,
                num_layers=config.hyperparams.layers,
                hidden=config.hyperparams.hidden,
                weight_conv=config.get('weight_conv', 'WeightConv1'),
                multi_channel=config.get('multi_channel', 'False'))
    model.to(device).reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.hyperparams.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.hyperparams.step_size,
                                                gamma=config.hyperparams.decay_rate)

    train_losses = []
    train_accs = []
    test_accs = []
    for epoch in range(1, config.hyperparams.epochs):
        scheduler.step()
        train_loss = train(train_loader, model, optimizer)
        train_acc = test(train_loader, model)
        test_acc = test(val_loader, model)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print('Epoch: {:03d}, Train Loss: {:.7f}, '
              'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                           train_acc, test_acc))

    return test_accs, train_losses, train_accs


def k_fold(dataset, folds, seed):
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)

    val_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        val_indices.append(torch.from_numpy(idx))

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, val_indices


def run_model(net, dataset, config):
    folds_test_accs = []
    folds_train_losses = []
    folds_train_accs = []

    def k_folds_average(avg_folds):
        avg_folds = np.vstack(avg_folds)
        return np.mean(avg_folds, axis=0), np.std(avg_folds, axis=0)

    for fold, (train_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, 10, config.get('seed', 12345)))):

        if fold >= config.get('folds_cut', 10):
            break

        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]

        train_loader = DataLoader(train_dataset, config.hyperparams.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, config.hyperparams.batch_size, shuffle=False)

        print('------------- Split {} --------------'.format(fold))
        test_accs, train_losses, train_accs = run_given_fold(
            net,
            dataset,
            train_loader,
            val_loader,
            config=config
        )

        folds_test_accs.append(np.array(test_accs))
        folds_train_losses.append(np.array(train_losses))
        folds_train_accs.append(np.array(train_accs))

    avg_test_accs, std_test_accs = k_folds_average(folds_test_accs)
    sel_epoch = np.argmax(avg_test_accs)
    sel_test_acc = np.max(avg_test_accs)
    sel_test_acc_std = std_test_accs[sel_epoch]

    print('------------- Final results --------------')
    print("Mean accuracy: {:.7f}, std: {:.7f}".format(sel_test_acc, sel_test_acc_std))


def main():
    args = get_args()
    config = process_config(args)
    print(config)

    if config.get('seed') is not None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    dicts = {'SMG': SMG,
             'SMG_JK': SMG_JK,
             'SMG_2h': SMG_2h,
             'SMG_2h_JK': SMG_2h_JK}
    dataset = get_dataset(config.dataset_name).shuffle()
    run_model(dicts[config.get('model', 'SMG')], dataset, config=config)


if __name__ == "__main__":
    main()
