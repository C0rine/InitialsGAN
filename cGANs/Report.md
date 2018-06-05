# Conditional GANs
Below the output of several conditional GANs are shown. All the GANs were originally intended for the celebA set, but were modified to work with the initials dataset. All models are conditioned on 'A' and 'B', just to do a quick check of the performance of the GAN. 

## Elegant
GitHub repo: https://github.com/Prinsphield/ELEGANT

Output of the model is a row of images. The first two images represent the input images, the second the images in which the features are swapped, the latter two are the reconstructed images.

Results on the celebA dataset (some of the first steps):

!["celebA_ELEGANT"](https://github.com/C0rine/InitialsGAN/blob/master/cGANs/images/BangsBeard_swap_ELEGANT.gif "celebA_ELEGANT")

In each of the images either the property 'Bangs' is swapped, or the property 'Beard' is swapped. 

Results on the initials dataset (some of the last steps):

!["initials_ELEGANT"](https://github.com/C0rine/InitialsGAN/blob/master/cGANs/images/AB_swap_ELEGANT.gif "initials_ELEGANT")

In each of the images either the property 'A'  is swapped, or the property 'B' is swapped. 



## cDCGAN
GitHub repo: https://github.com/togheppi/cDCGAN

Output of this model is a square in which the two images above eachother are from the same noise vector but conditioned on a different attribute.

Results on the celebA dataset:

!["celeba_cDCGAN"](https://github.com/C0rine/InitialsGAN/blob/master/cGANs/images/CelebA_cDCGAN_epochs_20.gif "celeba_cDCGAN")

When trained on the properties 'black hair' and 'brown hair'. 

Results on the initials dataset:

!["initials_cDCGAN"](https://github.com/C0rine/InitialsGAN/blob/master/cGANs/images/initials_cDCGAN_epochs_20.gif "initials_cDCGAN")

When trained on the properties 'A' and 'B'.


## StarGAN
GitHub repo: https://github.com/yunjey/StarGAN

I would like to use this model, but I do not have enough memory.
