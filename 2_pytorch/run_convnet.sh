#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model convnet \
    --kernel-size 7 \
    --hidden-dim 32 \
    --epochs 2 \
    --weight-decay 5e-4 \
    --momentum 0.9 \
    --batch-size 512 \
    --lr 5e-4 | tee convnet.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
