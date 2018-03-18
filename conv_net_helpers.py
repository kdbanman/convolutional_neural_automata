# Convolutional Neural Automata
# Neural networks trained as cellular automata.
# Copyright (C) 2018  Kirby Banman (kirby.banman@gmail.com)

# Redistributable and modifiable under the terms of GNU GPLv3.
# See LICENSE.txt or <http://www.gnu.org/licenses/>


import math

from torch import nn

def initialize_convolution_weights(modules):
    for module in modules:
        if isinstance(module, nn.Conv2d):
            number_of_parameters = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            module.weight.data.normal_(0, math.sqrt(2. / number_of_parameters))
            if module.bias is not None:
                module.bias.data.zero_()