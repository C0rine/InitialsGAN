# GAN-vanilla
Code was adapted from the [pytorch-gan](https://github.com/prcastro/pytorch-gan) repository by [prcastro](https://github.com/prcastro).
The images from the initials dataset were upon loading transformed to 28x28 pixels to speed up training. 

## Results 
Below a gif showing output for every 100 steps (a step being one mini batch) of each epoch (of a total of 200 epochs). The output seems to relapse quite often and does not really seem to improve any further I do not feel it is necessary to increase the number of epochs. 

![Output for GAN01](https://github.com/C0rine/InitialsGAN/blob/master/01-GAN-Vanilla/Images/GAN01_gif.gif "Output for GAN01")

*(The GIF only plays once so it is clear where is starts and finishes, GIF might also take a while to load.)*

In general the performance of this GAN is pretty bad, but it can be noted that several things were picked up by the vanilla GAN. Firstly we see how the model quickly picks up upon the black center and white border around most initials. See the image below for an image from epoch 10. 

![Epoch 10](https://github.com/C0rine/InitialsGAN/blob/master/01-GAN-Vanilla/Images/WhiteOutline.png "epoch 10")

Secondly we can see how the output images sometimes reflect the two main modes within the dataset; namely that where the initial itself is taking up all the space in the image, and those were the initial is only covering part of the image (often just the top left corner). 

![Two modes](https://github.com/C0rine/InitialsGAN/blob/master/01-GAN-Vanilla/Images/TwoModes.png "Two modes")

An example of each of these modes from the real dataset:

!["Full"](https://github.com/C0rine/InitialsGAN/blob/master/01-GAN-Vanilla/Images/Full.png "Full")
!["Part"](https://github.com/C0rine/InitialsGAN/blob/master/01-GAN-Vanilla/Images/Part.png "Part")

And finally, in some of the output we can kind of detect the lines at which the letter are most often depicted. They form a sort of generic grid along which most of the lines of various initials would lie. 

![Letter shapes](https://github.com/C0rine/InitialsGAN/blob/master/01-GAN-Vanilla/Images/LetterOutlines.png "Letter shapes")

## Model architecture

#### Discriminator model
```
    nn.Linear(784, 1024),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Dropout(0.3),
    nn.Linear(1024, 512),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Dropout(0.3),
    nn.Linear(256, 1),
    nn.Sigmoid()
```

#### Generator model
```
    nn.Linear(100, 256),
    nn.LeakyReLU(0.2, inplace=True),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(256, 512),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(512, 1024),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(1024, 784),
    nn.Tanh()
```

## Parameter settings
* Batch size = 100
* Optimizer function: BCE loss
* Learning rate: 0.0002
