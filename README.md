# Multi-digit Version of MNIST: Hand-written Digit Recognition

Note: This *Neural Network Digit Recognition Exercise* is from https://github.com/cyberyu/uot_intern

### Model 1: Multi layer perceptron (mlp.py)

#### - Modification of code:

- Define the layers of MLP in MLP class.

- Since one image contains two digits, a shared hidden layer is used for feature extraction (**multi-task learning**), then another output linear layer (task-specific) is stacked for each of the digit respectively to predict the output.

#### - Performance

After epoch 25, the validation accuracy stabilize around 92% ~ 93%.

The test accuracy after 30 epoch is 92. 5403% and 91.7591% for two digits respectively.

### Model 2: Convolutional Neural Network (CNN) (conv.py)

#### - Modification of code:

- Define the layers of CNN in CNN class.

- Since one image contains two digits, a shared hidden layer is used for feature extraction (**multi-task learning**), then another output linear layer (task-specific) is stacked for each of the digit respectively to predict the output.

- "Sandwitch" archetecture is used: a cycle of a simple cell acting like filters and a complex cell performing pooling.

Three archetecture for CNN: 

###### Version 1 (One "sandwich" archetecture + ADAM + 10 epochs):

**Structure:**

**Input --> convolutional layer --> batch normalization --> activation (ReLU) --> max pooling layer --> dropout --> parallel fully connected layer 1 and layer 2**

The number of epoch is changed to 10.

#### Performance

##### Optimizer: ADM

##### Number of epochs: 10

##### Accuracy: 0.965474  and 0.953881

###### Version 2  (Two "sandwich" archetecture + ADAM + 10 epochs):

**Structure:**

**Input --> convolutional layer 1 --> batch normalization --> activation (ReLU) --> max pooling layer 1 --> dropout --> convolutional layer 2--> batch normalization --> activation (ReLU) --> max pooling layer2 -->parallel fully connected layer 1 and layer 2**

#### Performance

##### Optimizer: ADAM

##### Number of epochs: 10

##### Accuracy: 0.969254  and 0.959173

We can see that even after adding a sandwitch archetecture, the performance didn't improve significantly.

###### Version 3  (One "sandwich" archetecture + SGD + 30 epochs):

##### Optimizer: SGD unchanged

##### Number of epochs: 30

##### Accuracy:   0.973286  and 0.959677

In the above three versions of CNN, the last one performs best

### Model 3: CNN with ResNet (conv-resnet.py)

In this model, I added a ResNet in it (i.e. add ResidualBlock inside the CNN class, also add a function make_layer). 

###### Version 1 (One "sandwich" archetecture + ADAM + 10 epochs):

**Structure:**

**Input --> convolutional layer --> batch normalization --> activation (ReLU) --> max pooling layer --> residual block --> residual block 1 --> global pooling -- > parallel fully connected layer 1 and layer 2**

##### Optimizer: ADAM

##### Number of epochs: 10

##### Accuracy:   0.977067  and 0.970514

###### Version 2 (One "sandwich" archetecture + ADAM + 15 epochs):

The architecture is same as the above.

##### Optimizer: ADAM

##### Number of epochs: 14

##### Accuracy:  0.976058  and 0.970514

After adding a ResNet , the performance was improved a lot, but the training procee is strikingly slower than the previous models without ResNet.

## Conclusion for this task:

Some methods to improve the performance:

-  Change the optimizer (e.g. change SGD to a more sophisticated one such as ADAM, but in this task, SGD performs even better)
- Add batch nomalization ( for improving the speed, performance, and stability of neural networks)
- Add dropout layer to avoid overfitting (but in this task, adding dropout didn't give a gobetterod performance)
- Use a deeper neural network (this depends however, in this task, two convolutional layers had nearly the same performance as one convolutional layer)
- **<u>My highes accuracy: CNN with ResNet (Version 1: One "sandwich" archetecture + ADAM + 10 epochs) has *the highest accuracy: 0.977067  and 0.970514* for two digits, respectively</u>**.

To sum up, to recognize the hand-writting digits is not a complicated task for neural network, so reletavily consice and simple archtecher is even better for a sophisticated one. However, intuitively, the accuracy of this task is lower that single digit recognization, since we need two fully connected layers to predict two digits for one time. 

** Remark **

Code is partially referred from https://github.com/yunjey/pytorch-tutorial
