#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model softmax \
    --epochs 5 \
    --weight-decay 5e-4 \
    --momentum 0.8 \
    --batch-size 512 \
    --lr 5e-4 | tee softmax.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
