# Q8.8 Experiment

# Intro
**
In my implementation for the EvalAI chanllenge, I
(1) started experimenting with a very deep neural network with a large number of parameters to learn and then reached 77% validation accuracy after training,
(2) removed the last one and two blocks of the network to simplify it. I call these two new networks shallow and simple respectively.
(3) added Batch Normalization layers and dropout layers to the three networks
(4) applied Adam optimizer
**
# Structure

The entire network consists of two parts: the convolutional net and the fully-connect net.

The convolutional net consists of four blocks, each of which is constructed with two convolution layers a max-pooling layer and a ReLU activation layer. The size of the block is increasing externally, however the numbers of channels of the layers within each channels remain unchanged. To be more specifically, the numbers of filters within each block are, respectively, 64, 128, 256 and 512.

The fully-connected net is fed with the tensors after they are outputted from the convolutional net and flattened into an 1-D tensor. This net contains three linear layers. The first one is of the size of the 1-D tensor, the second one 1024 and the third one mapped the tensor into 10 classes.

After experiments, I added dropout layers after each of the convolutional layers and the linear layers, and batch normalization layers after each four blocks in the convolutional net.

To reduce the complexity of the network, (since I realized that the dataset is actually a small one that I don't need a large network like the deep one), I removed the last block first and then last two together in the convolutional net, adjusted the input of the fully-connect net and named the two new structures the shallow and the simple, respectively.

#Results
Batchsize = 512

||***deep***|***shallow***|***simple***|
| --------   | -----:  | :----:  |
| val accuracy | 78%   |   76%     | 67% |
| train loss | 0.218020 | 0.437188 | 0.894410 |
| val loss | 0.683865 | 0.710023 | 0.981931 |
| converging epoch|    33    |  19  | 6 |
I used the shallow net on EvalAI chanllenge, and reached 76% on the test.