# GAN - Adjustable

* Used PR Castro's repository for reference for basis of the GAN: https://github.com/prcastro/pytorch-gan
* Used Yun Chen's repository for reference for conversion to DCGAN and WGAN: https://github.com/chenyuntc/pytorch-GAN
* Used Mamy Ratsimbazafy's comment for reference for save/load of the model: https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610

I have changed the GAN with a UI (Ipython widgets) so the model can be easily changed (normalization, batchsize, no. epochs, learning rate, betas, clamp values).
Another new function is the save and load, which allows you to interrupt and resume training at any time. Finally also a start has been made with the evaluation by implementation of a nearest neighbor finder. 

## Results 
I have run the model in three settings: 

### 1. BCE, Batchnorm

Full settings: ConvolutionalGAN, BCE loss, Adam optimizer, BatchNorm, batchsize of 32, learning rate of 0.0002, beta1 of 0.5, for 25 epochs.

The gif of training and final output: 
                                            
!["BCEBatch-gif"](https://github.com/C0rine/InitialsGAN/blob/master/04_GAN-Adjustable/Images/BCEBatch-gif.gif "BCEBatch-gif")
!["BCEBatch"](https://github.com/C0rine/InitialsGAN/blob/master/04_GAN-Adjustable/Images/BCEBatch.png "BCEBatch")

And here a comparison between some output images and their nearest neighbor (euclidean distance) from the training set. On the left of each column the generated image, on the right the nn from the dataset (training set).  

![BCEBatch-nn](https://github.com/C0rine/InitialsGAN/blob/master/04_GAN-Adjustable/Images/BCEBatchNN.jpg "BCEBatch-nn")

### 2. BCE, Instancenorm

Full settings: ConvolutionalGAN, BCE loss, Adam optimizer, InstanceNorm, batchsize of 32, learning rate of 0.0002, beta1 of 0.5, for 25 epochs.

The gif of training and final output: 

!["BCEInstance-gif"](https://github.com/C0rine/InitialsGAN/blob/master/04_GAN-Adjustable/Images/BCEInstance-gif.gif "BCEInstance-gif")
!["BCEInstance"](https://github.com/C0rine/InitialsGAN/blob/master/04_GAN-Adjustable/Images/BCEInstance.png "BCEInstance")

And here a comparison between some output images and their nearest neighbor (euclidean distance) from the training set. On the left of each column the generated image, on the right the nn from the dataset (training set).  

![BCEInstance-nn](https://github.com/C0rine/InitialsGAN/blob/master/04_GAN-Adjustable/Images/BCEInstanceNN.jpg "BCEInstance-nn")

We quickly see that with the same model settings, batchnorm performs way better than instancenorm. Especially looking at the nearest neighbors for the instancenorm, we see that almost all output images have the same nearest neighbor. This suggest that with instancenorm the model tends to collapse to a single mode. 

### 3. Wasserstein, Batchnorm

Full settings: ConvolutionalGAN, Wasserstein loss, RMSprop optimizer, BatchNorm, batchsize of 32, learning rate of 0.0002, beta1 of 0.5, for 25 epochs. (Clamp value of 0.01)

The gif of training and final output: 

!["WassBatch-gif"](https://github.com/C0rine/InitialsGAN/blob/master/04_GAN-Adjustable/Images/WassBatch-gif.gif "WassBatch-gif")
!["WassBatch"](https://github.com/C0rine/InitialsGAN/blob/master/04_GAN-Adjustable/Images/WassBatch.png "WassBatch")

And here a comparison between some output images and their nearest neighbor (euclidean distance) from the training set. On the left of each column the generated image, on the right the nn from the dataset (training set).  

![WassBatch-nn](https://github.com/C0rine/InitialsGAN/blob/master/04_GAN-Adjustable/Images/WassBatchNN.jpg "WassBatch-nn")

In the Wasserstein we see that the model is not able at all to generate anything that remote looks like letters. We do see however that it is capable of finding very diverse output when we look at the nearest neighbors.
