#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
from cnn import CNN
from highway import Highway
import torch

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

# from cnn import CNN
# from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        self.embed_size = embed_size
        self.vocab = vocab
        self.char_embed_size = 50
        self.max_word_len = 21
        self.dropout_rate = 0.3
        self.char_embedding = nn.Embedding(
            num_embeddings=len(vocab.char2id),
            embedding_dim=self.char_embed_size,
            padding_idx=vocab.char2id['<pad>']
        )
        self.CNN = CNN(
            char_embedding_size=self.char_embed_size,
            num_filters=embed_size,
            max_word_length=self.max_word_len,
            kernel_size=5
        )
        self.Highway = Highway(embed_size)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        ### END YOUR CODE

    def forward(self, input_tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input_tensor: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        x_emb = self.char_embedding(input_tensor) # sent_len x batch_size x max_word x char_embed_size
        sent_len, batch_size, max_word, _ = x_emb.shape # need to reshape to 3 dimensions
        x_reshape = x_emb.view(sent_len * batch_size, max_word, self.char_embed_size).transpose(1, 2)
        x_cnn = self.CNN(x_reshape)
        x_high = self.Highway(x_cnn)
        x_out = self.dropout(x_high)
        x_out = x_out.view(sent_len, batch_size, self.embed_size) # from 2 dims to 3 dims
        return x_out
        ### END YOUR CODE
