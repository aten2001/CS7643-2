import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Extra credit model

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        #pass
        C,H,W = im_size
        
        self.convnet = nn.Sequential(nn.Conv2d(3,64,3,1,1),
                                 nn.Dropout2d(),
                                 nn.Conv2d(64,64,3,1,1),
                                 nn.Dropout2d(),
                                 nn.MaxPool2d(2,2),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(),
                                 nn.Conv2d(64,128,3,1,1),
                                 nn.Dropout2d(),
                                 nn.Conv2d(128,128,3,1,1),
                                 nn.Dropout2d(),
                                 nn.MaxPool2d(2,2,1),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(),
                                 nn.Conv2d(128,256,3,1,1),
                                 nn.Dropout2d(),
                                 nn.Conv2d(256,256,3,1,1),
                                 nn.Dropout2d(),
                                 nn.MaxPool2d(2,2,1),
                                 nn.BatchNorm2d(256),
                                 nn.ReLU())#,
        '''
                                 nn.Conv2d(256,512,3,1,1),
                                 nn.Dropout2d(),
                                 nn.Conv2d(512,512,3,1,1),
                                 nn.Dropout2d(),
                                 nn.MaxPool2d(2,2,1),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU(),
                )
        '''
        '''
        self.convnet = nn.Sequential(
                nn.Conv2d(3,16,3,padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2,stride=2),
                nn.Conv2d(16,32,3,padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2,stride=2),
                nn.Conv2d(32,32,3,padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2,stride=2))
        self.fcnet = nn.Sequential(
                nn.Linear(32*4*4,32*4*4),
                nn.Linear(32*4*4,32*2*2),
                nn.Linear(32*2*2,10))
        '''
        self.fcnet = nn.Sequential(nn.Linear(256*5*5,1024), #shallow
                #nn.Linear(512*3*3,1024),#deep
                                 nn.Dropout2d(),
                                 nn.Linear(1024,1024),
                                 nn.Dropout2d(),
                                 nn.Linear(1024,10)
                                 #nn.Softmax()
                                 )
        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the model to
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
        # TODO: Implement the forward pass.
        #############################################################################
        #pass
        #scores = self.net(images)
        
        scores = self.convnet(images)
        #print(scores.shape)
        scores = scores.view(-1,256*5*5) #shallow
        #scores = scores.view(-1,32*4*4) #simple
        #scores = scores.view(-1,512*3*3) #deep
        scores = self.fcnet(scores)
        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

