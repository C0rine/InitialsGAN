# GAN-vanilla
Code was adapted from the [pytorch-gan](https://github.com/prcastro/pytorch-gan) repository by [prcastro](https://github.com/prcastro).
It is a vanilla GAN originally trained on the MNIST dataset, adaptions have only been made to accomodate the intials dataset. All images from the initials dataset were upon loading transformed to 28x28 pixels.

## Results 
Below a gif showing output for every 100 steps (out of 300) of each epoch (of a total of 200 epochs). The output seems to relapse quite often and does not really seem to improve any further I do not feel it is necessary to increase the number of epochs. 

*insert gif here*

The output is not very realistic, but it can be seen that several things were picked up by the vanilla GAN. Firstly we see how the algorithm quickly picks up upon the black center and white border around the images.

*insert image of someting*

Secondly we can see how the output images sometimes reflect the two main modes within the dataset; namely that where the initial itself is taking up all the space in the image, and those were the initial is only covering part of the image (often just the top left corner). 

*insert images here of the two modes*

And finally, in some of the output we can kind of detect the lines at which the letter are most often depicted. They form a sort of generic grid along which most of the lines of various initials would lie. 

*insert image of that stuff described above*

## Model architecture
Size of hidden layers = 100

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
