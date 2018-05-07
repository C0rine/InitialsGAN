# Results ELEGANT
The images from the initials dataset were extracted from the hdf5 file and saved seperately in a folder. The metadata of the initials dataset was then formatted the same way as the celebA list_attr_celeba.txt. The reformatted initials dataset was put into the ELEGANT model: https://github.com/Prinsphield/ELEGANT

Training was done with batch size 8 and the following parameters:
* G_lr = 2e-4
* D_lr = 2e-4
* betas = [0,5, 0.999]
* weight_decay = 1e-5
* step_size = 3000
* gamma = 0.97

Training was done on only two conditions at the time. Once on the letter A and letter B, once on two countries (DE and FR). 

### Training on A and B
Was run for 88000 steps. 
Some of the best training results:

![AB1-b](https://github.com/C0rine/InitialsGAN/blob/master/ELEGANT/images/AB-1b.jpg "AB-1b")

![AB2-b](https://github.com/C0rine/InitialsGAN/blob/master/ELEGANT/images/AB-2b.jpg "AB-2b")

![AB3-b](https://github.com/C0rine/InitialsGAN/blob/master/ELEGANT/images/AB-3b.jpg "AB-3b")

![AB4-b](https://github.com/C0rine/InitialsGAN/blob/master/ELEGANT/images/AB-4b.jpg "AB-4b")

Average training results (taken randomly):

![AB1](https://github.com/C0rine/InitialsGAN/blob/master/ELEGANT/images/AB-1.jpg "AB-1")

![AB2](https://github.com/C0rine/InitialsGAN/blob/master/ELEGANT/images/AB-2.jpg "AB-2")

![AB3](https://github.com/C0rine/InitialsGAN/blob/master/ELEGANT/images/AB-3.jpg "AB-3")

![AB4](https://github.com/C0rine/InitialsGAN/blob/master/ELEGANT/images/AB-4.jpg "AB-4")

### Training on DE and FR
Was run for 100000 steps. Results never really got good, below some examples. 

![DEFR1](https://github.com/C0rine/InitialsGAN/blob/master/ELEGANT/images/DEFR-1.jpg "DEFR-1")

![DEFR2](https://github.com/C0rine/InitialsGAN/blob/master/ELEGANT/images/DEFR-2.jpg "DEFR-2")

![DEFR3](https://github.com/C0rine/InitialsGAN/blob/master/ELEGANT/images/DEFR-3.jpg "DEFR-3")

![DEFR4](https://github.com/C0rine/InitialsGAN/blob/master/ELEGANT/images/DEFR-4.jpg "DEFR-4")


### Analysis of resuls
CelebA dataset had really well aligned images for which the features have a clear location in the image (e.g. smile, bangs, beard). The ELEGANT even customly aligns the celebA dataset image (based on pupil location). In the case of the initials dataset we see that the images are not that well aligned and that their features aren't tied to specific locations  
