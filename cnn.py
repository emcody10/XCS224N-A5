#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch

### YOUR CODE HERE for part 1e
class CNN(nn.Module):
    """
    Convolutional network
    """
    def __init__(self, char_embedding_size, num_filters, max_word_length, kernel_size=5):
        """
        Initialize CNN
        Args:
            char_embedding_size: embedding size of characters in a word
            num_filters: number of output channels
            max_word_length: maximum word length
            kernel_size: convolution window size
        """
        super(CNN, self).__init__()
        self.char_embedding_size = char_embedding_size # input channels
        self.kernel_size = kernel_size
        self.num_filters = num_filters # output channels
        self.max_word_length = max_word_length
        self.conv = nn.Conv1d(char_embedding_size, num_filters, kernel_size)
        self.maxpool = nn.MaxPool1d(max_word_length-kernel_size+1)

    def forward(self, input):
        """
        Take batch of character embeddings, compute word embedding
        Args:
            input: (batch_size, char_embed_size, max_word_length)

        Returns:
            x_out: (batch_size, word_embed_size)

        """
        x_conv = self.conv(input)
        #x_out = self.maxpool(F.relu(x_conv)).squeeze() # (batch_size, word_embed_size)
        x_out = torch.max(F.relu(x_conv), dim=2)[0]
        return x_out
### END YOUR CODE
