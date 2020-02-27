#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F

### YOUR CODE HERE for part 1d
class Highway(nn.Module):
    """
    Highway network
    Maps x_conv_out to x_highway
    """
    def __init__(self, word_embed_size):
        """
        Args:
            word_embed_size: embedding size for input and output
        """
        super(Highway, self).__init__()
        self.word_embed_size = word_embed_size
        self.proj = nn.Linear(self.word_embed_size, self.word_embed_size)
        self.gate = nn.Linear(self.word_embed_size, self.word_embed_size)

    def forward(self, x_conv):
        '''
        Compute x_highway on minibatch of convolution output
        Args:
            x_conv: batch_size x word_embed_size

        Returns:
            x_highway: batch_size x word_embed_size

        '''
        x_proj = F.relu(self.proj(x_conv))
        x_gate = torch.sigmoid(self.gate(x_conv))
        x_highway = x_gate * x_proj + (1-x_gate)*x_conv
        return x_highway

### END YOUR CODE 

