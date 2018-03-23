# GAN convolutional
This in an implementation of a convolutional GAN. I have used the code of Yun Chen as a reference (https://github.com/chenyuntc/pytorch-GAN). The intials dataset was loaded into memory in 64x64 pixels. The output of this GAN is also 64x64. 

## Results 
I initially started with a batch size of 128. I quickly noticed that the generator performed a whole lot better than the vanilla GAN. As can be seen in the screenshot below however the generator does not seem to be fully able to distinguish between the seperate letters. I remains a general blob. It does however capture the two modes again (see report on GAN-01) and seems to have captured the elegance of the letters somehow.

![Output for GAN02](https://github.com/C0rine/InitialsGAN/blob/master/02-GAN-Convolutional/Images/gif_128batchsize.gif "Output for GAN02")
Batch size 128

I thereafter quickly tried with a lower batch size of 32. It quickly becomes clear in this case that the model is way getting way better at capturing the seperate letters. 

*gif to be implemented here*

The model was in both cases running for 25 epochs. It seems the model is still improving at the 25th epoch, but not a lot. I therefore decided model tweaking/trying to implement wasserstein might be a better to improve performance than letter the model run longer.  

## Model architecture

#### Discriminator model
```
    nn.Conv2d(1,64,4,2,1,bias=False),
    nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(64,64*2,4,2,1,bias=False),
    nn.BatchNorm2d(64*2),
    nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(64*2,64*4,4,2,1,bias=False),
    nn.BatchNorm2d(64*4),
    nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(64*4,64*8,4,2,1,bias=False),
    nn.BatchNorm2d(64*8),
    nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(64*8,1,4,1,0,bias=False),
    nn.Sigmoid()
```

#### Generator model
```
    nn.ConvTranspose2d(100,64*8,4,1,0,bias=False),
    nn.BatchNorm2d(64*8),
    nn.ReLU(True),

    nn.ConvTranspose2d(64*8,64*4,4,2,1,bias=False),
    nn.BatchNorm2d(64*4),
    nn.ReLU(True),

    nn.ConvTranspose2d(64*4,64*2,4,2,1,bias=False),
    nn.BatchNorm2d(64*2),
    nn.ReLU(True),

    nn.ConvTranspose2d(64*2,64,4,2,1,bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),

    nn.ConvTranspose2d(64,1,4,2,1,bias=False),
    nn.Tanh()
```

## Parameter settings
* Batch size: once 128, once 32 (see above)
* Optimizer function: BCE loss
* Learning rate: 0.0002, beta1 = 0.5
