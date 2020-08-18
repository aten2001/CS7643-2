#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model mymodel \
    --kernel-size 1 \
    --hidden-dim 10 \
    --epochs 60 \
    --weight-decay 1e-6 \
    --momentum 0.9 \
    --batch-size 512 \
    --lr 1e-4 | tee mymodel.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
