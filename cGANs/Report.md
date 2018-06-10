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

~~I would like to use this model, but I do not have enough memory.~~
Managed to make this work. 

Results on the celebA dataset:

!["CelebA_StarGAN"](https://github.com/C0rine/InitialsGAN/blob/master/cGANs/images/CelebA_StarGAN.gif "CelebA_StarGAN")

The first image in the row is the input image, and after that the generated images with certain attributes imposed on them. In this case these attributes are (from left to right): black hair, blonde hair, brown hair, male, young. 

Results on the initials dataset:

!["initials_StarGAN"](https://github.com/C0rine/InitialsGAN/blob/master/cGANs/images/initials_StarGAN.gif "initials_StarGAN")

The first image is again the input image, the images after that are supposed have imposed the attributes: A, B, C, D and E. We see however that it did not manage to do this at all. 
(The switch of the input images halfway is due to an interruption in the training process. When I loaded the saved model it picked different images for this output.)

## ACGAN
GitHub repo: https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/acgan

Results on 32x32 for all letter classes: 
!["ACGAN_32x32"](https://github.com/C0rine/InitialsGAN/blob/master/cGANs/images/ACGAN_32x32.png "ACGAN_32x32")

Each column is a class listing some generated examples (from left to right: 'a', 'b', 'c', ... , 'z' , '')

Results on 64x64 for 10 letter classes (due to memory constraints):
!["ACGAN_64x64"](https://github.com/C0rine/InitialsGAN/blob/master/cGANs/images/ACGAN_64x64.png "ACGAN_64x64")

Each column is a class listing some generated examples (from left to right: 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j')
