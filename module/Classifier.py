
import torch
import torch.nn as nn



class GroupedLinear(nn.Module):
    def __init__(self, in_features, out_features, num_groups):
        super(GroupedLinear, self).__init__()
        assert out_features % num_groups == 0, "The out_features should be divisible by num_groups."

        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups

        group_in_features = in_features // num_groups
        group_out_features = out_features // num_groups

        self.grouped_linear = nn.ModuleList([
            nn.Linear(group_in_features, group_out_features) for _ in range(num_groups)
        ])

    def forward(self, x):
        x_split = torch.split(x, self.in_features // self.num_groups, dim=1)
        x_out = torch.cat([linear(xi) for xi, linear in zip(x_split, self.grouped_linear)], dim=1)
        return x_out
