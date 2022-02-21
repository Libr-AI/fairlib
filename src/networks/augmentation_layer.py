import torch.nn as nn
import torch
import copy

class Augmentation_layer(nn.Module):
    def __init__(self, mapping, num_component, device, sample_component) -> None:
        super().__init__()

        self.mapping = mapping
        self.num_component = num_component
        self.device = device

        # Init the augmentation layer
        self.augmentation_components = nn.ModuleList()
        for _ in range(num_component):
            self.augmentation_components.append(copy.deepcopy(sample_component))
    
    def forward(self, input_data, group_label):
        specific_output = []
        # Get group-specific representations
        # number_classes * batch_size * adv_units
        for _group_id in range(self.num_component):
            _group_output = input_data
            for layer in self.augmentation_components[_group_id]:
                _group_output = layer(_group_output)
            specific_output.append(_group_output) # batch_size * adv_units
            
        # Reshape the out_g to batch*num_classes*adv_units
        specific_output = [i.unsqueeze(dim=1) for i in specific_output] # Each element has the shape: batch_size * 1 * adv_units
        specific_output = torch.cat(specific_output, dim=1)

        # Mapping the group label to one-hot representation
        group_label = self.mapping[group_label.long()] # batch_size * num_classes
        group_label = group_label.unsqueeze(dim=1) # batch_size * 1 * num_classes

        # (batch_size * 1 * num_classes) * (batch*num_classes*adv_units)
        specific_output = torch.matmul(group_label.to(self.device), specific_output) # (batch_size * 1 * adv_units)
        specific_output = specific_output.squeeze(dim=1) # (batch_size * adv_units)

        return specific_output