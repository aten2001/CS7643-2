import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(CNN, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        #pass
        self.conv = nn.Conv2d(in_channels=im_size[0],out_channels=hidden_dim,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.l = nn.Linear(hidden_dim*(im_size[1]//2)*(im_size[2]//2),n_classes)
        #self.softmax = nn.Softmax()
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the CNN to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = None
        #############################################################################
        # TODO: Implement the forward pass. This should take few lines of code.
        #############################################################################
        #pass
        scores = self.conv(images)
        scores = self.relu(scores)
        #print(scores.shape)
        scores = self.pool(scores)
        scores = scores.view(-1,scores.shape[1]*scores.shape[2]*scores.shape[3])
        scores = self.l(scores)
        #scores = self.softmax(scores)
        #print(scores.shape)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

