import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, Sigmoid
from torch_geometric.nn import (global_add_pool, JumpingKnowledge)
from conv.sparse_conv import SparseConv
from conv.weight_conv import WeightConv1, WeightConv2


class SMG(torch.nn.Module):
    def __init__(self,
                 dataset,
                 num_layers,
                 hidden,
                 weight_conv='WeightConv1',
                 multi_channel='False'):
        super(SMG, self).__init__()
        self.lin0 = Linear(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(SparseConv(hidden, hidden))

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

        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.lin0.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for mask in self.masks:
            mask.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.lin0(x)
        mask_val = None
        for i, conv in enumerate(self.convs):
            mask = self.masks[i]
            mask_val = mask(x, edge_index, mask_val)
            x = F.relu(conv(x, edge_index, mask_val))
        x = F.relu(self.lin1(global_add_pool(x, batch)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class SMG_JK(torch.nn.Module):
    def __init__(self,
                 dataset,
                 num_layers,
                 hidden,
                 weight_conv='WeightConv1',
                 multi_channel='False'):
        super(SMG_JK, self).__init__()
        self.lin0 = Linear(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(SparseConv(hidden, hidden))

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

        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.lin0.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for mask in self.masks:
            mask.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.lin0(x)
        mask_val = None
        xs = []
        for i, conv in enumerate(self.convs):
            mask = self.masks[i]
            mask_val = mask(x, edge_index, mask_val)
            x = F.relu(conv(x, edge_index, mask_val))
            xs += [global_add_pool(x, batch)]
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class SMG_2h(torch.nn.Module):
    def __init__(self,
                 dataset,
                 num_layers,
                 hidden,
                 weight_conv='WeightConv1',
                 multi_channel='False'):
        super(SMG_2h, self).__init__()
        self.lin0 = Linear(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(SparseConv(hidden, hidden))

        self.ma1hs = torch.nn.ModuleList()
        self.ma2hs = torch.nn.ModuleList()
        if multi_channel == 'True':
            out_channel = hidden
        else:
            out_channel = 1
        if weight_conv != 'WeightConv2':
            for i in range(num_layers):
                self.ma1hs.append(WeightConv1(hidden, hidden, hidden, aggr='mean'))
                self.ma2hs.append(WeightConv1(hidden, hidden, out_channel, aggr='mean'))
        else:
            for i in range(num_layers):
                self.ma1hs.append(WeightConv2(Sequential(
                    Linear(hidden * 2, hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    Sigmoid()
                ), aggr='mean'))
                self.ma2hs.append(WeightConv2(Sequential(
                    Linear(hidden * 2, hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    ReLU(),
                    Linear(hidden, out_channel),
                    Sigmoid()
                ), aggr='mean'))

        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.lin0.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for ma1hs in self.ma1hs:
            ma1hs.reset_parameters()
        for ma2hs in self.ma2hs:
            ma2hs.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.lin0(x)
        for i, conv in enumerate(self.convs):
            ma1h = self.ma1hs[i]
            ma2h = self.ma2hs[i]
            mask_1ho = ma1h(x, edge_index)
            mask_val = ma2h(mask_1ho, edge_index)
            x = F.relu(conv(x, edge_index, mask_val))
        x = F.relu(self.lin1(global_add_pool(x, batch)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class SMG_2h_JK(torch.nn.Module):
    def __init__(self,
                 dataset,
                 num_layers,
                 hidden,
                 weight_conv='WeightConv1',
                 multi_channel='False'):
        super(SMG_2h_JK, self).__init__()
        self.lin0 = Linear(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(SparseConv(hidden, hidden))

        self.ma1hs = torch.nn.ModuleList()
        self.ma2hs = torch.nn.ModuleList()
        if multi_channel == 'True':
            out_channel = hidden
        else:
            out_channel = 1
        if weight_conv != 'WeightConv2':
            for i in range(num_layers):
                self.ma1hs.append(WeightConv1(hidden, hidden, hidden, aggr='mean'))
                self.ma2hs.append(WeightConv1(hidden, hidden, out_channel, aggr='mean'))
        else:
            for i in range(num_layers):
                self.ma1hs.append(WeightConv2(Sequential(
                    Linear(hidden * 2, hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    Sigmoid()
                ), aggr='mean'))
                self.ma2hs.append(WeightConv2(Sequential(
                    Linear(hidden * 2, hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    ReLU(),
                    Linear(hidden, out_channel),
                    Sigmoid()
                ), aggr='mean'))

        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.lin0.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for ma1hs in self.ma1hs:
            ma1hs.reset_parameters()
        for ma2hs in self.ma2hs:
            ma2hs.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.lin0(x)
        xs = []
        for i, conv in enumerate(self.convs):
            ma1h = self.ma1hs[i]
            ma2h = self.ma2hs[i]
            mask_1ho = ma1h(x, edge_index)
            mask_val = ma2h(mask_1ho, edge_index)
            x = F.relu(conv(x, edge_index, mask_val))
            xs += [global_add_pool(x, batch)]
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
