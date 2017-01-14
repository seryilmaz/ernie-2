# ernie-2
CNN training for Ernie architecture using TensorFlow.
Requires TensorFlow 0.10.0 or above. For previous versions, some changes in code are needed.
Tested in Python 2.7.
Training is performed with CIFAR10 dataset. Current architecture has 6 conv layers with max-pooling after every 2 layers; 2 FC layers, and 1 softmax layer.
Experiments are performed by splitting the FC layers into multiple (2-10) blocks. Split is done either locally or randomly. 
To train, use train.py. Command line arguments are described below. See sweep.py for an example of how to run train.py. sweep.py can be used to automatically perform sweeps by communicating with command prompt.

Command line arguments to train.py:
Example:
python train.py 0.03 0.1 0.9 0.0004 0.03 0.03 300000 0 0 1

First argument, which is 0.03 here is the initial learning rate

Second argument, which is 0.1 here, is the learning rate decay. Learning rate decay is continuous, and it is decayed every step exponentially so that after 350 steps, it is decayed by 0.1. 350 can be changed in network.py file, but it seems to work well in our experiments.

Third argument, which is 0.9 here; is the momentum for gradient descent. Optimization algorithm can be changed in network.py. If Adam is used, other arguments are needed (this is commented out in network.py). If gradient descent is used without momentum, momentum argument is ignored. This can be done by changing the optimizer to plain gradient descent (again this options is commented out in network.py).

Fourth argument, which is 0.0004 here, is weight decay factor. Weight decay is only used for FC layers.

Fifth argument, 0.03, is the stddev of initialization of first FC layer.

Sixth argument, 0.03 here, is the stddev of initialization for second FC layer.

Seventh argument, 300000 here, is the number of steps to run the network.

Eighth argument, 0 here; decides whether or not FC layers are used as is or are splitted to multiple blocks (results in sparser FC layer). If 0, no splitting is performed. If 1, it splits FC layers so that every neuron has a maximum of 256 fanin/fanout.

Ninth argument, which is 0 here, represents whether or not FC layer splits are done locally or randomly. If 1, it is done randomly. If previous argument is 0, this one is ignored. If random split is chosen, you need to include the connections in this randomly splitted FC layer using an input file (there is already one right now so you can run as is). We use SparseTensor class in TensorFlow to implement this. The indices to the instance of SparseTensor class is read from sparsegen3.py file. Please do not edit the files that starts with 'sparsegen'; they are previously generated by running randomgen.py file. Of course, the indices in this file are fixed. You can randomly generate another set of indices using the file randomgen.py. If you run randomgen.py as is, it will overwrite sparsegen3.py, and generate another random splitting.

Tenth argument, which is 1 here; represents whether the training uses any validation set. If it is 1, network uses the 20% of trianing data as validation set and outputs on command window the validation accuracies. Note that we implement early stopping here using these validation set. Early stopping is done by training the network with a number of steps (indicated by seventh argument), saving checkpoints for the model with largest validation set accuracy so far, and using the model with largest validation accuracy to find the test accuracy (with no further training). If it is 0, the whole training set is used for training. 
