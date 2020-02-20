import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops
import torch.nn.functional as F

from torch_geometric.nn.inits import uniform, reset


class WeightConv1(MessagePassing):

    def __init__(self, in_channels, hid_channels, out_channels=1, aggr='add', bias=True,
                 **kwargs):
        super(WeightConv1, self).__init__(aggr=aggr, **kwargs)

        self.lin1 = torch.nn.Linear(in_channels, hid_channels, bias=bias)
        self.lin2 = torch.nn.Linear(in_channels, hid_channels, bias=bias)
        self.lin3 = torch.nn.Linear(hid_channels * 2, hid_channels, bias=bias)
        self.lin4 = torch.nn.Linear(hid_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        self.lin4.reset_parameters()

    def forward(self, x, edge_index, mask=None, edge_weight=None, size=None):
        edge_index, _ = remove_self_loops(edge_index)
        if mask is not None:
            x = x * mask
        h = self.lin1(x)
        return self.propagate(edge_index, size=size, x=x, h=h,
                              edge_weight=edge_weight)

    def message(self, h_j, edge_weight):
        return h_j if edge_weight is None else edge_weight.view(-1, 1) * h_j

    def update(self, aggr_out, x):
        weight = torch.cat([aggr_out, self.lin2(x)], dim=-1)
        weight = F.relu(weight)
        weight = self.lin3(weight)
        weight = F.relu(weight)
        weight = self.lin4(weight)
        weight = torch.sigmoid(weight)
        return weight

    def __repr__(self):
        return self.__class__.__name__


class WeightConv2(MessagePassing):

    def __init__(self, nn, aggr='add', **kwargs):
        super(WeightConv2, self).__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x, edge_index, mask=None, edge_weight=None, size=None):
        edge_index, _ = remove_self_loops(edge_index)
        if mask is not None:
            x = x * mask
        h = x
        return self.propagate(edge_index, size=size, x=x, h=h,
                              edge_weight=edge_weight)

    def message(self, h_j, edge_weight):
        return h_j if edge_weight is None else edge_weight.view(-1, 1) * h_j

    def update(self, aggr_out, x):
        weight = torch.cat([aggr_out, x], dim=-1)
        weight = self.nn(weight)
        return weight

    def __repr__(self):
        return self.__class__.__name__
