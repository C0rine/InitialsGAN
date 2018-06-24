import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from numpy import array

import torch.nn as nn
import torch.nn.functional as F
import torch

import dataset
import extractor
import evaluation

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
# Model settings
parser.add_argument('--n_epochs', type=int, default=2000, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=500, help='dimensionality of the latent space')
parser.add_argument('--sample_interval', type=int, default=500, help='interval between image sampling')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--img_size', type=int, default=64, help='size of each image dimension')

# Specify training
parser.add_argument('--train', type=bool, default=False, help='Whether or not to do the training')
parser.add_argument('--training_feat', type=str, default='letters', help='feature to train on, choose from: \'letters\', \'countries\', \'cities\' or \'names\'')
parser.add_argument('--letter_restr', type=str, default='S', help='restrict training to this letter, use \'all\' to train on all letters')

# Set path to data
parser.add_argument('--csv_path', type=str, default='Z:/CGANs/PyTorch-GAN/implementations/acgan/custom_datasets/', help='path folder with the features csv (only used if letter_restr != all)')
parser.add_argument('--img_path', type=str, default='Z:/CGANs/PyTorch-GAN/implementations/acgan/custom_datasets/', help='path to the folder containing the images (only used if letter_restr != all)')

# Specify evaluation
parser.add_argument('--test', type=bool, default=True, help='Whether or not to do the testing')
parser.add_argument('--eval_dir', type=str, default='Z:/CGANs/PyTorch-GAN/implementations/acgan/evaluation/', help='Set the directory for where to save the evaluation data')
parser.add_argument('--get_incep_imgs', type=bool, default=False, help='Get inception images at the end of training')
parser.add_argument('--nn', type=int, default=0, help='Get this number of nearest neighbors, 0 for no NN evaluation')
parser.add_argument('--ssim', type=bool, default=True, help='Whether or not to perform SSIM evaluation')

opt = parser.parse_args()
print(opt)




# ----- Meta-data and preprocessing ---- #

alphabet = ['A', 'B',  'C',   'D',  'E', 'F', 'G', 'H',  'I', 'J','K', 'L', 'M',   'N', 'O',  'P',  'Q',  'R', 'S',  'T', 'U','V',  'W','X', 'Y','Z', '']
# let cnt  [2415, 580, 2141, 1968, 2090, 888, 527, 1517, 2417, 3, 83, 1142, 1326, 1543, 1100, 2180, 2198, 755, 2564, 1249, 5, 1653, 115, 48, 25, 118, 0]

country_list = ['FR', 'CH', 'DE', 'ES', 'IT', 'BE', 'GB', 'POR', 'NL', 'DK', 'CZ']

cities_list = ['Lyon', 'Paris', 'Basel', 'Zwickau', 'Valencia', 'Frankfurt', 'Brescia', 'Madrid', 'Antwerpen', 'Venezia', 'Firenze', 'Oppenheim', 'Munchen', 'Magdenburg', 'Strasbourg', 
                'London', 'Sevilla', 'Koln', 'Louvain', 'Zurich', 'Worms', 'Bologna', 'Ingoldstadt', 'Ferrara', 'Marburg', 'Geneve', 'Pavia', 'Bergamo', 'Neustadt', 'Mainz', 'Heidelberg', 
                'Hagenau', 'Leipzig', 'Jena', 'Augsburg', 'Wesel', 'Macerata', 'Como', 'Nuremberg', 'Douai', 'Vincenza', 'Alcala', 'Wittenberg', 'Dortmund', 'Morges', 'Lucca', 'Lisboa',
                'Dillingen', 'Rome', 'Alkmaar', 'Amsterdam', 'Hamburg', 'Leiden', 'Ieper', 'Kopenhagen', 'Freiburg', 'Zaragozza', 'Troyes', 'Salamanca', 'Erfurt', 'Poitiers', 'Wurzburg', 
                'Genova', 'Caen', 'Praha']

names_list = ['Simon Bevilaqua', 'Matthieu David', 'Valentin Curio', 'Gabriel Kantz', 'Diego Gumiel', 'Jean Pillehotte', 'Andreas Cratander', 'Sebastien Cramoisy', 'Seitz Heirs Peter Main', 
                'Andre Bocard', 'Comino Presegni', 'Pedro Madrigal', 'Oudin Petit', 'Willem Vorsterman', 'Gaspare 2 Bindoni', 'Matthias David', 'Pre Galliot Du', 'Philippo 2 Giunta', 
                'Jakob Kobel', 'Adam Berg', 'Philippo 1 Giunta', 'Antoine Vincent', 'Wolfgang Kirchener', 'Gabriel Buon', 'Johann Knobloch', 'Guillaume Cavellat', 'Gilles Gourbin', 
                'Vincenzo Sabbio', 'Becker Matthias Main', 'Robert Barker', 'Bonetus Locatellus', 'Jacob Cromberger', 'Bartholomaeus Vincent', 'Mylius Crato', 'Antoine Augerelle', 
                'Robert Chaudiere', 'Hero Fuchs', 'Mace Bonhomme', 'Lucantonio 1 Giunta', 'Charlotte Guillard', 'Lucantonio 2 Giunta', 'Claude Servain', 'Peter Quentell', 'Jacques Huguetan', 
                'Arnold Heirs Birckman', 'Dirck Martensz', 'Giovanni Battista Ciotti', 'Bernardino Gerrualda', 'Claude Chevallon', 'Andreas Gessner', 'Simon Beys', 'Peter Schoeffer', 
                'Johann Oporinus', 'Heinrich Petri', 'Sebastien Honorat', 'Giovanni Rosso', 'Girolamo Scoto', 'Robert Winter', 'Bartollomeo Carampello', 'Alexander Weissenhorn', 
                'Vittorio Baldini', 'Candido Benedetto', 'Simon Colines', 'Adam Petri', 'Jean Roigny', 'Jacques Sacon', 'Christian Egenolff', 'Sebastian Henricpetri', 'Giovanni Maria Bonelli', 
                'Johann Soter', 'Eustache Vignon', 'Giacomo Pocatello', 'Antoine Ry', 'Ludwig Ostenius', 'Jamet Mettayer', 'Jean Louis', 'Jean Tournes', 'Feyerabend Johann Main', 'Joannes Moylin', 
                'Etienne Gueynard', 'Gerwin Calenius', 'Francesco Rampazetto', 'Comino Ventura', 'Johann Schott', 'Pierre Marechal', 'Wilhelm Harnisch', 'Claude Fremy', 'Sebastien Griffo', 
                'Gregorio Gregoriis', 'Josse Bade', 'Lorenzo Torrentino', 'Francesco Franceschi', 'Franz Behem', 'Melchior Neuss', 'Francesco Rossi', 'Cyriacus Jacob Main', 'Andreas Cambier', 
                'Domenico Farri', 'Ivo Schoeffer', 'Heinrich Gran', 'Michael Blum', 'Heirs Thomas Rebart', 'Peter Perna', 'Melchior Lotter', 'Ludwig Alectorius', 'David Sartorius', 
                'Wechel Heirs Andreas Main', 'Paganinus Paganinis', 'Ernhard Ratdolt', 'Johann Walder', 'Heinrich Stayner', 'Porte Hugues La', 'Hans Braker', 'Johann Miller', 'G M Beringen', 
                'Johann Feyerabend', 'Thielman Kerver', 'Ulrich Gering', 'Arnold Birckman', 'Vincent Portonariis', 'Robert Estienne', 'Aldus 1 Manutius', 'Jaques Marechal', 'Braubach Peter Main', 
                'Nicolas Benedictis', 'Valentin Kobian', 'Sebastiano Martellini', 'Prez Nicolas Des', 'Jerome Olivier', 'Francois Regnault', 'Michael Sonnius', 'Bartolommeo Zanis', 'Pierre Fradin', 
                'Johann Herwagen', 'Egenolff Christian Main', 'Arnoul Angelier', 'Laurent Sonnius', 'Claude Senneton', 'Thomas Anshelm', 'Theodore Rihel', 'Andre Wechel', 'Guillaume Chaudiere', 
                'Symphorien Barbier', 'Melchior Sessa', 'Joannes Crispinus', 'Christian Wechel', 'Johan Bebel', 'Antoine Harsy', 'Eucharius Cervicorius', 'Lechler Martin Main', 'Anselmo Giacarelli', 
                'Jean le Preux', 'Michael Hillenius', 'Porte Sybille La', 'Gottardo da Ponte', 'Berthold Rembold', 'Johann von Berg', 'Blaise Guido', 'GB Bellagamba', 'Willem Sylvius', 
                'Georg Hantsch', 'Damiano Zenaro', 'Michel Vascosan', 'Jan Loe', 'Adrien Tunebre', 'Giovanni Battista Phaelli', 'Conrad Bade', 'Jeune Martin Le', 'Allesandro Benacci', 
                'Jan 1 Roelants', 'Jean Bogard', 'Nicolaus Episcopius', 'Vincent Valgrisi', 'George Bishop', 'Gilbert Villiers', 'Johann Quentell', 'Theobaldus Ancelin', 'Richter Wolfgang Main', 
                'Noir Guillaume Le', 'Michael Isengrin', 'Guillaume Rouille', 'Jean Clein', 'Joannes Platea', 'Pierre Vidoue', 'Sebastien Nivelle', 'Louis Cynaeus', 'Jean Le Rouge', 
                'Christophorus Zelle', 'Giorgio Greco', 'Arnaldo Guillen Brocar', 'Hieronymus Verdussen', 'Laurentius Annison', 'Jakob Kundig', 'Guillaume Guillemot', 'Christoph Rodinger', 
                'Symon Cock', 'Claude Davost', 'Charles Estienne', 'Jan 1 Moretus', 'Johann Schwertel', 'Francesco Bolzetta', 'Fevre Francois Le', 'Melchior Soter', 'Denys Janot', 
                'Johann Schoeffer', 'Guillaume Lairmarius', 'Venturina Rosselini', 'Eichorn Andreas Oder', 'Jacques Faure', 'Barthelemy Ancelin', 'Vincenzo Busdraghi', 'Martin LEmpereur', 
                'Johann Neuss', 'Hartmann Friedrich Oder', 'Pierre Galterus', 'Hieronymus Wallaeus', 'Rutgerus Velpius', 'Levinus Hulsius', 'de Ferrara Gabriele Giolito', 'Antonio Padovani', 
                'Jean Cavelleir', 'Bassaeus Oder', 'Johann Faber', 'Ambrosius Froben', 'Guichard Julieron', 'Hieronymus Commelin', 'Antoine Blanchard', 'Heirs Symphorien Beraud', 'Jacques Myt', 
                'Godefriedus Kempen', 'Puys Jacques Du', 'Andrea Arrivabene', 'Heinrich Quentell', 'Froschauer Christopher 1 CH', 'Miguel Eguia', 'Pierre Roussin', 'Caspar Behem', 'Luis Rodrigues', 
                'Henry Estienne', 'Samuel Konig', 'Guglielmo Fontaneto', 'Giovanni Guerigli', 'Mathieu Berjon', 'Anton Hierat', 'Sebald Mayer', 'Johann Prael', 'Widow Gabriel Buon', 
                'Heinrich Gymnicus', 'Thomas Courteau', 'Symphorien Beraud', 'Bonaventura Nugo', 'Joannes Criegher', 'Jan van Keerbergen', 'Bartolomeo Bonfadini', 'Petrus Colinaeus', 
                'Heirs Sebastien Griffo', 'Francois Fradin', 'Guillaume Morel', 'Hieronymus Froben', 'Widow Martinus Nutius', 'Johann Froben', 'Nicolaus Brylinger', 'Vivant Gautherot', 
                'Widow Hendrick Peetersen', 'Joachim Trognesius', 'Nicolaus Faber', 'Jean Petit', 'Thibaud Payen', 'Johann Setzer', 'Jean Crespin', 'Norton Johann Main', 'Jacques Androuet', 
                'Giovanni Bariletto', 'Hans Luft', 'Jean Ogerolles', 'Andrea Poleti', 'Jacob Meester', 'Christoffel Cunradus', 'van Waesberghe Joannes Janssonius', 'Maternus Cholinus', 
                'Thomas Raynald', 'Johann Ruremundensis', 'Jean Laon', 'Giorgio Angelieri', 'Andreas Angermaier', 'Jean Gerard', 'Martinus Verhasselt', 'Federic Morel', 'Elisabetta Rusconi', 
                'Georg Papen', 'Jean Dalier', 'Jean Bienne', 'Heirs Hieronymus Benedictis', 'Barezzo Barezzi', 'Thomas Wolf', 'Martinus Gymnicus', 'Etienne Dolet', 'Francesco Suzzi', 
                'Froschauer Christopher 2 CH', 'Clerc David Le', 'Konrad Mechel', 'Christoffel Guyot', 'Arnout Brakel', 'Marcus Zaltieri', 'Joannes Degaram', 'Joannes Masius', 'Henri Estienne', 
                'Horace Cardon', 'Hadrianus Perier', 'Joos Destree', 'Peter Seitz', 'Benoit Prevost', 'Johann Gruninger', 'Mats Vingaard', 'Konrad Caesarius', 'Jean Marion', 'Michel Cotier', 
                'Jacob Roussin', 'Breisgau Emmeus Johann im', 'Pierre Mettayer', 'Francisco Baba', 'Guillaume Julianus', 'Johann Birckman', 'Catharina Gerlach', 'Pedro Bernuz', 'Rouge Nicolas Le', 
                'Jacob Stoer', 'Guillaume Foquel', 'Konrad Waldkirch', 'Paul Frellon', 'Fredericus Lydius', 'Bartollomeo Alberti', 'Pasqier Le Tellier', 'Joannes Grapheus', 'Guillaume Lairmarie', 
                'Paulus Queckus', 'Hendrik Connix', 'Raben Georg Main', 'Johann Gymnicus', 'Pietro Maria Marchetti', 'Wolfgang Sthurmer', 'Philippe Tinghi', 'Pamphilius Gengenbach', 
                'Johann Schoenstenius', 'Jasper Gennep', 'Herman Moller', 'Pierre Gautier', 'Bernardino Vitalis', 'Jean Blanchet', 'Thomas Brumennius', 'Guillaume Merlin', 
                'Bartholomaeus Westheimer', 'Elzeviers', 'Heinrich von Aich', 'Georg Defner', 'Antoine Chuppin', 'Johann Gemusaeus', 'Antonio Bellona', 'Girard Angier', 'Giovanni 1 Griffio', 
                'Ottavio Scoto', 'Francois Arnoullet', 'Heirs Johann Quentell', 'Robert Field', 'Andreas2 en HJ Gesner', 'Wilhelm Lutzenkirchen', 'Bolognino Zaltieri', 'Georg Schwartz', 
                'Christoffel Plantin', 'Gilles Huguetan', 'Baldassare Constantini', 'Gerard Morrhe', 'Jacques Giunta', 'Bocchiana Nova Academia']


# Get the number of classes, the values of those classes and print the length
if opt.training_feat == 'letters':
    n_classes = len(alphabet)
    attr_list = alphabet
    print('Alphabet length: ' + str(len(alphabet)))
elif opt.training_feat == 'countries':
    n_classes = len(country_list)
    attr_list = country_list
    print('Countries length: ' + str(len(country_list)))
elif opt.training_feat == 'cities':
    n_classes = len(cities_list)
    attr_list = cities_list
    print('Cities length: ' + str(len(cities_list)))
elif opt.training_feat == 'names':
    n_classes = len(names_list)
    attr_list = names_list
    print('Names length: ' + str(len(names_list)))
else:
    raise ValueError('You cannot train on the chosen attribute, please use one of the following: \'letters\', \'countries\', \'cities\' or \'names\'.')



# -------- Model and optimizer set-up --------- #

# Check if GPU is available for training
cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
    """ Function to initialize the weights of the models """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    """ The generator model """
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, opt.latent_dim)

        self.init_size = opt.img_size // 4 # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128*self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    """ The discriminator model """
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2**4

        # Output layers
        self.adv_layer = nn.Sequential( nn.Linear(128*ds_size**2, 1),
                                        nn.Sigmoid())
        self.aux_layer = nn.Sequential( nn.Linear(128*ds_size**2, n_classes),
                                        nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label

# Loss functions
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# move to GPU is one is available
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


# -------- Creation of dataloader --------- #

# only necessary if we want to perform training
if opt.train:
    # load the initials dataset
    if opt.letter_restr == 'all':
        data = dataset.get_dataset()
    else: 
        csv = opt.csv_path + 'list_attr_' + opt.training_feat + '_' + opt.letter_restr.upper() + '.csv'
        img_dir = opt.img_path + 'Seperated_' + opt.letter_restr.upper() + '/'
        extractor.extract(opt.letter_restr, opt.training_feat, csv, img_dir)
        data = dataset.get_letterdataset(csv, img_dir)

    # print the number of images that will be trained on
    print('There are', len(data), 'images in this dataset')

    # create the dataloader
    dataloader = DataLoader(dataset=data, num_workers=opt.n_cpu, batch_size=opt.batch_size, shuffle=True, pin_memory=True)


# -------- Helper-functions --------- #

def count_class(feature):
    """Counts the number of items for each feature value in the dataset"""
    counts = []

    if feature == 'letter':
        attribute = alphabet
    elif feature == 'countries':
        attribute = country_list
    elif feature == 'cities':
        attribute = cities_list
    elif feature == 'names':
        attribute = names_list
    else: 
        raise ValueError('Feature is not a valid value.')

    # setup a list with a counter (starting at 0) for each feature
    for i in range(len(attribute)):
        counts.append(0)

    for i, (imgs, letters, countries, cities, names) in enumerate(dataloader):
        for k, initial in enumerate(feature):
            for l, letter in enumerate(attribute):
                if initial == letter:
                    counts[l] += 1

    return counts

def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row**2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    
    save_image(gen_imgs.data, 'images/%d.png' % batches_done, nrow=n_row, normalize=True)


def get_inception_images(nr_classes, attribute_list, save_dir):
    """we need about 50.000 images with uniform distribution over all classes"""
    iterations = int(50000/nr_classes) + 1
    for i in range(iterations):
        # Sample noise
        z = Variable(FloatTensor(np.random.normal(0, 1, (nr_classes, opt.latent_dim))))
        # Get labels ranging from 0 to n_classes for n rows
        labels = np.array([num for num in range(nr_classes)]) 
        labels = Variable(LongTensor(labels))
        gen_imgs = generator(z, labels)

        for j, image in enumerate(gen_imgs):
            # print(labels.cpu().numpy()[j])
            save_image(gen_imgs[j], save_dir + str(i) + '_' + attribute_list[j] + '.jpg')

     
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """ Function to save the current state of both the generator and discriminator """
    saveloc = 'models/' + filename
    torch.save(state, saveloc)
    print('-- Checkpoint saved --')


def load_checkpoint():
    """ Function to load model (either a fully trained model or in progress model) """
    cp_path = 'models/checkpoint.pth.tar'
    if os.path.isfile(cp_path):
        print('Loading checkpoint...')
        # load the data
        checkpoint = torch.load(cp_path)
        # forward it to the appropriate parts of the model
        epoch = checkpoint['epoch']
        generator.load_state_dict(checkpoint['g_state'])
        discriminator.load_state_dict(checkpoint['d_state'])
        optimizer_G.load_state_dict(checkpoint['g_optim'])
        optimizer_D.load_state_dict(checkpoint['d_optim'])
        print('Checkpoint successfully loaded')
        return epoch
    else:
        # There is no checkpoint so we can start at epoch 0
        print('No checkpoint to load.')
        return 0


# -------- Training --------- #
if opt.train: 
    # Load checkpoint if one if available
    start_epoch = load_checkpoint()

    # Start training process
    for epoch in range(opt.n_epochs):

        if epoch <= start_epoch:
            # Skip epoch numbers that were already done (in case of loading from a checkpoint)
            continue
        else: 
            for i, (imgs, letters, countries, cities, names) in enumerate(dataloader):
     
                batch_size = imgs.shape[0]

                # Adversarial ground truths
                valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

                # Get a list of flags to represent the features
                labels = []
                if opt.training_feat == 'letters':
                    for k, initial in enumerate(letters):
                        for l, letter in enumerate(alphabet):
                            if initial == letter:
                                labels.append(l)

                elif opt.training_feat == 'countries':
                    for k, country in enumerate(countries):
                        for l, l_country in enumerate(country_list):
                            if country == l_country:
                                labels.append(l)

                elif opt.training_feat == 'cities':
                    for k, city in enumerate(cities):
                        for l, l_city in enumerate(cities_list):
                            if city == l_city:
                                labels.append(l)

                elif opt.training_feat == 'names':
                    for k, name in enumerate(names):
                        for l, l_name in enumerate(names_list):
                            if name == l_name:
                                labels.append(l)

                # transform the features to a tensor
                labels = array(labels)
                labels = torch.from_numpy(labels)

                # Configure input
                real_imgs = Variable(imgs.type(FloatTensor))
                labels = Variable(labels.type(LongTensor))

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Sample noise and labels as generator input
                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
                gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))

                # Generate a batch of images
                gen_imgs = generator(z, gen_labels)

                # Loss measures generator's ability to fool the discriminator
                validity, pred_label = discriminator(gen_imgs)
                g_loss = 0.5 * (adversarial_loss(validity, valid) + \
                                auxiliary_loss(pred_label, gen_labels))

                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Loss for real images
                real_pred, real_aux = discriminator(real_imgs)
                d_real_loss =  (adversarial_loss(real_pred, valid) + \
                                auxiliary_loss(real_aux, labels)) / 2

                # Loss for fake images
                fake_pred, fake_aux = discriminator(gen_imgs.detach())
                d_fake_loss =  (adversarial_loss(fake_pred, fake) + \
                                auxiliary_loss(fake_aux, gen_labels)) / 2

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                # Calculate discriminator accuracy
                pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
                gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
                d_acc = np.mean(np.argmax(pred, axis=1) == gt)

                d_loss.backward()
                optimizer_D.step()

                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                                    d_loss.item(), 100 * d_acc,
                                                                    g_loss.item()))
                batches_done = epoch * len(dataloader) + i
               
                # Get a sample image at the indicated sample_interval
                if batches_done % opt.sample_interval == 0:
                    sample_image(n_row=10, batches_done=batches_done)

        # Save the model after each epoch
        save_checkpoint({
            'epoch': (epoch),
            'g_state': generator.state_dict(),
            'd_state': discriminator.state_dict(),
            'g_optim' : optimizer_G.state_dict(),
            'd_optim' : optimizer_D.state_dict()
        })

if opt.test: 
    # Create the evaluation directory if it is not yet present
    if not os.path.exists(opt.eval_dir):
        os.makedirs(opt.eval_dir)

    # generate 50000 images with the trained model 
    if opt.get_incep_imgs:
        get_inception_images(n_classes, attr_list, opt.eval_dir)

    # perform NN evaluation is nr of neighbors is not set to zero
    if opt.nn != 0:
        nn_dir = opt.eval_dir + 'nn/'
        if not os.path.exists(nn_dir):
            os.makedirs(nn_dir)
        evaluation.nearest_neighbor(opt.nn, 10, 20, 'Z:/CGANs/PyTorch-GAN/implementations/acgan/Z-Finished/inception_imgs_letters/', nn_dir)

    # perform SSIM evaluation
    if opt.ssim: 
        ssim_class_means = evaluation.SSIM_classmeans('Z:/CGANs/PyTorch-GAN/implementations/acgan/Z-Finished/inception_imgs_letters/', attr_list)
        print(ssim_class_means)
