from torch import nn
import copy

def clones(layer, num_layers):
    "Produce num_layers identical layers."
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])