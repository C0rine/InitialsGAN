from   torch.utils.data import DataLoader
from   torch.autograd import Variable
import torch, torchvision
import torchvision.transforms as tf
import torch.nn as nn
import argparse, os
import csv
import itertools 
import numpy as np
from scipy.stats import entropy

#                                                                             1    1    1    1    1    1    1    1    1    1    2    2    2    2   2
#                           0    1    2    3    4    5    6    7    8    9    0    1    2    3    4    5    6    7    8    9    0    1    2    3   4
# Order of letters:       ['M', 'Q', 'C', 'N', 'P', 'O', 'D', 'T', 'H', 'S', 'V', 'B', 'A', 'W', 'I', 'E', 'L', 'F', 'R', 'Z', 'G', 'Y', 'X', 'K', '']


print("Started loading stuff...")
parser = argparse.ArgumentParser(description='Initials Project STL Training')
parser.add_argument('--task', default='initials', type=str, help='Classify initials or countries.')
parser.add_argument('--incep', default=True, type=bool, help='Perform inception score calculation?')
parser.add_argument('--testimg', default='imgs/', type=str, help='Specify the path to the images to be evaluated')

args = parser.parse_args()

threads       = 0
use_cuda      = torch.cuda.is_available()

task          = args.task
perform_incep = args.incep
test_path     = args.testimg


# Test function
def test_model(model, test_loader, class_dict, loader_index, imgnames):

    # We need this because of the dropout and batch normalization we use. They behave differently in eval mode.
    model.eval()

    probs = []
    predictions_return = []
    predictions_csv = []

    for batch_idx, returned_values in enumerate(test_loader):

        if use_cuda:
            inputs = returned_values[0].cuda(async=True)

        # Get the classifications of the model
        inputs = Variable(inputs)
        outputs = model(inputs)
        probslist = outputs.data.cpu().numpy()[0]

        # Get the class for which the classifier feels most confident
        _, pred = torch.max(outputs.data, 1)
        # print('Final assignment for ' + imgnames[batch_idx] + ':')
        predint = pred.cpu().numpy()[0]
        predlet = class_dict[predint]

        # collect the data
        probs.append(probslist)
        predictions_return.append(predint)
        predictions_csv.append(predlet)

    # Save the classifications to file (csv)
    with open('results.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(imgnames, predictions_csv, probs))

    # Return as well
    return probs, predictions_return


# Helper function to make predictions sum to 1
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_inceptionscore(probs, splits):   

    # no. images
    N = len(filenames)
    np_probs = []

    # Softmax the probabilties
    for i, item in enumerate(probs):
        np_probs.append(softmax(np.asarray(probs[i])))
    
    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part =  np.asarray(np_probs[k * (N // splits): (k+1) * (N // splits)], dtype=np.float32)
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


# Make dataloader for the to-be-classified images
# Get the names of the images
filenames = os.listdir(test_path)

# Transform the images
transform = tf.Compose([
    tf.Grayscale(),
    tf.Resize([224, 224]),
    tf.ToTensor()
])

test_data = torchvision.datasets.ImageFolder(test_path, transform=transform)
test_loader = DataLoader(dataset=test_data, num_workers=threads, batch_size=1, shuffle=False)


# Setup the loader index, hardcoded since we only use the pretrained model
if task == 'initials':
    model = '/model_snapshots/initials.pt'
    class_dictionary = ['M', 'Q', 'C', 'N', 'P', 'O', 'D', 'T', 'H', 'S', 'V', 'B', 'A', 'W', 'I', 'E', 'L', 'F', 'R', 'Z', 'G', 'Y', 'X', 'K', '']
    class_count = 25
    loader_index = 1
elif task == 'countries':
    model = '/model_snapshots/countries.pt'
    class_dictionary = ['IT', 'DE', 'CH', 'FR', 'BE', 'POR', 'DK', 'NL', 'GB', 'ES', 'CZ']
    class_count = 11
    loader_index = 2


# Start the testing
print('Starting the classification...')
testing_model = torch.load(model) 
probs, predictions = test_model(testing_model, test_loader, class_dictionary, loader_index, filenames)
print('Classification finished.')

# Performing inception scoring
if perform_incep:
    print('Starting calculation of inception score...')
    mean, std = get_inceptionscore(probs, 5)
    print(mean)
    print(std)
    print('Inception score calculation finished.')