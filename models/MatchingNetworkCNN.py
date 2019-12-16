import torch
import torch.nn as nn
import util
from collections import OrderedDict
import functional_layers as L
import math

class MatchingNetworkCNN(nn.Module):

    def __init__(self, num_classes):
        super(MatchingNetworkCNN, self).__init__()
        self.layers = []
        for block_id in range(3):
            self.layers += util.ConvBlock(str(block_id), input_channels=(1 if block_id == 0 else 64))
        self.features = nn.Sequential(OrderedDict(self.layers))
        self.add_module('fc', nn.Linear(64, num_classes))

        self._init_weights()

    def forward(self, input, weights=None, cuda=True):
        input = input.double()
        if weights == None:
            output = self.features(input)
            output = output.view(output.size(0), 64)
            output = self.fc(output)
        else:
            # Manually use weights for forward pass to make derivative through inner loop possible
            output = input
            for block_id in range(3):
                output = L.conv2d(output, weights[f'features.conv{block_id}.weight'], weights[f'features.conv{block_id}.bias'], cuda=cuda)
                output = L.batch_norm(output, weights[f'features.bn{block_id}.weight'], bias=weights[f'features.bn{block_id}.bias'], momentum=1, cuda=cuda)
                output = L.relu(output)
                output = L.max_pool(output, kernel_size=2, stride=2)
            output = output.view(output.size(0), 64)
            output = L.linear(output, weights['fc.weight'], weights['fc.bias'], cuda=cuda)
        return output

    def _init_weights(self):
        torch.manual_seed(11)
        torch.cuda.manual_seed_all(11)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                flattened_size = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2. / flattened_size))
                if module.bias is not None:
                    module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()
                elif isinstance(module, nn.Linear):
                    module.weight.data.normal_(0, 0.01)
                    module.bias.data = torch.ones(module.bias.data.size())

    def set_weights(self, copied_matching_network):
        for module_from, module_to in zip(copied_matching_network.modules(), self.modules()):
            if isinstance(module_to, nn.Linear) or isinstance(module_to, nn.Conv2d) or isinstance(module_to, nn.BatchNorm2d):
                module_to.weight.data = module_from.weight.data.clone()
                if module_to.bias is not None:
                    module_to.bias.data = module_from.bias.data.clone()