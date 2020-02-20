import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, Sigmoid
from torch_geometric.nn import (global_add_pool, JumpingKnowledge)
from conv.sparse_conv import SparseConvR
from conv.weight_conv import WeightConv1, WeightConv2


class SMG_R(torch.nn.Module):
    def __init__(self,
                 dataset,
                 num_layers=3,
                 hidden=64,
                 weight_conv='WeightConv1',
                 multi_channel='False'):
        super(SMG_R, self).__init__()
        self.lin0 = Linear(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(SparseConvR(hidden, hidden,
                                          Sequential(Linear(5, 128),
                                                     ReLU(),
                                                     Linear(128, hidden * hidden)),
                                          bias=False))

        self.masks = torch.nn.ModuleList()
        if multi_channel == 'True':
            out_channel = hidden
        else:
            out_channel = 1
        if weight_conv != 'WeightConv2':
            for i in range(num_layers):
                self.masks.append(WeightConv1(hidden, hidden, out_channel))
        else:
            for i in range(num_layers):
                self.masks.append(WeightConv2(Sequential(
                    Linear(hidden * 2, hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    ReLU(),
                    Linear(hidden, out_channel),
                    Sigmoid()
                )))

        self.lin1 = torch.nn.Linear(hidden, (num_layers + 1) // 2 * hidden)
        self.lin2 = torch.nn.Linear((num_layers + 1) // 2 * hidden, hidden)
        self.lin3 = torch.nn.Linear(hidden, 1)

    def reset_parameters(self):
        self.lin0.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for mask in self.masks:
            mask.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.lin0(x)
        mask_val = None

        for i, conv in enumerate(self.convs):
            mask = self.masks[i]
            mask_val = mask(x, edge_index, mask_val)
            x = F.elu(conv(x, edge_index, edge_attr, mask_val))

        x = F.elu(self.lin1(global_add_pool(x, batch)))
        x = F.elu(self.lin2(x))
        x = self.lin3(x)
        return x.view(-1)

    def __repr__(self):
        return self.__class__.__name__


class SMG_JK_R(torch.nn.Module):
    def __init__(self,
                 dataset,
                 num_layers=3,
                 hidden=64,
                 weight_conv='WeightConv1',
                 multi_channel='False'):
        super(SMG_JK_R, self).__init__()
        self.lin0 = Linear(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(SparseConvR(hidden, hidden,
                                          Sequential(Linear(5, 128),
                                                     ReLU(),
                                                     Linear(128, hidden * hidden)),
                                          bias=False))

        self.masks = torch.nn.ModuleList()
        if multi_channel == 'True':
            out_channel = hidden
        else:
            out_channel = 1
        if weight_conv != 'WeightConv2':
            for i in range(num_layers):
                self.masks.append(WeightConv1(hidden, hidden, out_channel))
        else:
            for i in range(num_layers):
                self.masks.append(WeightConv2(Sequential(
                    Linear(hidden * 2, hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    ReLU(),
                    Linear(hidden, out_channel),
                    Sigmoid()
                )))

        self.lin1 = torch.nn.Linear(num_layers * hidden, (num_layers + 1) // 2 * hidden)
        self.lin2 = torch.nn.Linear((num_layers + 1) // 2 * hidden, hidden)
        self.lin3 = torch.nn.Linear(hidden, 1)

        self.jump = JumpingKnowledge(mode='cat')

    def reset_parameters(self):
        self.lin0.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for mask in self.masks:
            mask.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.lin0(x)
        mask_val = None
        xs = []

        for i, conv in enumerate(self.convs):
            mask = self.masks[i]
            mask_val = mask(x, edge_index, mask_val)
            x = F.elu(conv(x, edge_index, edge_attr, mask_val))
            xs += [global_add_pool(x, batch)]

        x = self.jump(xs)
        x = F.elu(self.lin1(x))
        x = F.elu(self.lin2(x))
        x = self.lin3(x)
        return x.view(-1)

    def __repr__(self):
        return self.__class__.__name__
