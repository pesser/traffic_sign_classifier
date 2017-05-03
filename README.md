## Traffic Sign Classification

As part of Udacity's Self Driving Car Nanodegree,  I have implemented a convolutional neural network in [TensorFlow](https://www.tensorflow.org/) to classify traffic signs from the [German Traffic Sign Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). [My implementation](https://github.com/pesser/traffic_sign_classifier) achieves an accuracy of 98.62% on the test set which is close to human level accuracy of 98.81%. This document was written with [StackEdit](https://stackedit.io/) and [a rendered version can be found online](https://github.com/pesser/traffic_sign_classifier).

### Dataset Exploration
The dataset is prepared into three pickled files containing images and labels for training, validation and testing splits. The splits have the following sizes:

|  Split   | Samples  |  Height  |  Width   | Channels | Classes  |
|----------|----------|----------|----------|----------|----------|
|  train   |  34799   |    32    |    32    |    3     |    43    |
|  valid   |   4410   |    32    |    32    |    3     |    43    |
|   test   |  12630   |    32    |    32    |    3     |    43    |


Furthermore a description of the labels is given and we plot the labels together with an example image of that label:

![all traffic signs](http://i.imgur.com/y71SdNN.png)

The labels are distributed quite unevenly but each split has a similiar distribution (the black line shows the fraction that would occur if each label contained the same number of samples):

![label distribution](http://i.imgur.com/qu83KyA.png)

The top-3 most and least frequent labels are

|           Fraction           |         Traffic Sign         |
|------------------------------|------------------------------|
|            5.78 %            |     Speed limit (50km/h)     |
|            5.69 %            |     Speed limit (30km/h)     |
|            5.52 %            |            Yield             |
|            0.52 %            |     Speed limit (20km/h)     |
|            0.52 %            | Dangerous curve to the left  |
|            0.52 %            |     Go straight or left      |


I decided to not equalize the distribution by prioritizing data augmentation of rare labels or any other technique but consider the uneven distribution as a natural prior on the frequency of the traffic signs. Indeed, speed limits of 20 kmh are not very common, whereas speed limits of 50 and 30 kmh are quite common in Germany.

### Data processing
Even a relatively small network was able to perfectly fit the training data in less than 100 epochs. After that the loss does not provide any gradients anymore and thus validation accuracy is stuck as well. To avoid this problem I augment the data using random rescalings and biases for augmentation of contrast and brightness as well as an affine transformation including translation, rotation, zooming and shearing for geometric augmentations. Using different magnitudes for the augmentations I can control how easy the network can fit the data. A good balance that I have found results in the following augmentations, where the first row depicts original data samples and each subsequent row shows an augmented version of it:

![data augmentation](http://i.imgur.com/XcE6tkR.png)

Besides normalization to [-1, 1], I did not apply any preprocessing to the data but let the network learn the transformations it finds most useful. Due to the highly varying brightness in the images I initially tried to apply histogram equalization but I could not observe a consistent improvement of results and finally decided against it.

### Model architecture
I started with a simple model containing two convolution-nonlinearity-downsampling blocks followed by three fully connected layers producing logits:

|           Layer              |         Details              |
|------------------------------|------------------------------|
|            Convolution             |     kernel size: 3, feature maps: 32       |
|            Activation              |     ReLU                                   |
|            Downsampling            |     MaxPooling, kernel size: 3, stride: 2  |
|            Convolution             |     kernel size: 3, feature maps: 64       |
|            Activation              |     ReLU                                   |
|            Downsampling            |     MaxPooling, kernel size: 3, stride: 2  |
|            Flatten                 |                                            |
|            FullyConnected          |     units: 128                             |
|            Activation              |     ReLU                                   |
|            FullyConnected          |     units: 64                              |
|            Activation              |     ReLU                                   |
|            FullyConnected          |     units: 43 (number of classes)          |

All weights are initialized according to [He](https://arxiv.org/abs/1502.01852). The network's cross entropy loss was optimized for 100 epochs in batches of 64 with [Adam](https://arxiv.org/abs/1412.6980) using a constant learning rate of 1e-3. No data augmentation was used initially and this resulted in a validation accuracy of 95.36%. I tried different nonlinearities but stayed with ReLU. Using strided convolution for downsampling seemed to improve the performance a bit and I kept it as the preferred method for downsampling. The learning rate is a critical parameter and ideally we would like to adjust it according to the Lipschitz constant of our objective. However, since neural networks model highly complex and nonlinear functions we do not really know about their Lipschitz constant. A commonly used technique is to decay the learning rate during training.  I adopted a linear learning rate decay which (together with strided convolution for downsampling) improved the validation accuracy to 96.94%. Looking at the plots of the training procedure

![training](http://i.imgur.com/4tln8hg.png)

we can see that the network fits the training data perfectly after less than 25000 batches of training. Afterwards, the loss does not provide useful information anymore and the validation accuracy stagnates. To circumvent this problem, I augmented the training data. This made it much more difficult for the network to fit the data and I could even increase the size of the network by adding two more blocks of convolution-activation-downsampling without saturating the loss:

![training2](http://i.imgur.com/LhInfmT.png)

The final validation accuracy of the network was 98.14%. As a final performance boost I added dropout to the network to get a better balance between training and validation loss and added one more block of convolution-activation-downsampling. A dropout ratio of 0.15 resulted in a validation accuracy of 99.52% and the following training dynamics:

![final training](http://i.imgur.com/tTKXwFq.png)

### Evaluation

My final network starts with five convolution-activation-downsampling blocks, each convolution with a receptive field of three, activations being ReLUs with dropout ratio of 0.15 during training, downsampling being stride two convolutions. The number of feature maps are increasing linearly, they are (32, 64, 96, 128, 160). These blocks are then followed by two fully connected layers with 128 and 64 units, respectively, and activated in the same way as the convolutional blocks. Finally, a fully connected layer produces class logits from which the cross entropy loss is derived. All weights are He-initialized. The loss is minimized by Adam in batches of 64 using a learning rate that decays linearly from 1e-3 to zero in 100 epochs. It achieved the following metrics:

|  Split   |  Loss      |  Accuracy    |
|----------|------------|--------------|
|  train   |  0.03369   |    99.24%    |
|  valid   |  0.02765   |    99.52%    |
|   test   |  0.09345   |    98.62%    |

According to [the German Traffic Sign Recognition Benchmark paper](https://www.ini.rub.de/upload/file/1470692848_f03494010c16c36bab9e/StallkampEtAl_GTSRB_IJCNN2011.pdf) humans achieve a testing accuracy of 98.81% and the best result achieved during the competition was 98.98% using a committee of CNNs. Considering that my implementation uses just a single network with a relatively simple architecture I am quite satisfied with the results. The only thing that I would like to investigate more carefully is the big gap between validation and testing accuracy (as well as loss) but maybe this is just due to the small size of the validation set.

I also collected some images of traffic signs in [Heidelberg, Germany](https://goo.gl/maps/bJeZvHViKK42), cropped them by hand, resized them and had them classified by my network. The plot below shows these images together with the top-five categorical probabilities as predicted by the network. We see that the network is completely confident for each image and classifies them all correctly.

![class probabilities](http://i.imgur.com/hIEBWlE.png)