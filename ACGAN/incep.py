from   torch.utils.data import DataLoader
from   torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as tf
import torch.nn as nn
import argparse, os
import csv
import itertools 
import numpy as np
from scipy.stats import entropy


print("Started loading stuff...")
parser.add_argument('--task', default='letters', type=str, help='Classify letters, countries, cities or names.')
parser.add_argument('--incep', default=True, type=bool, help='Perform inception score calculation?')
parser.add_argument('--testimg', default='Z:/CGANs/PyTorch-GAN/implementations/acgan/incep/', type=str, help='Specify the path to the images to be evaluated')

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

    print('Starting to get the prediction probabilities...')

    for batch_idx, returned_values in enumerate(test_loader):
        print(batch_idx)

        if use_cuda:
            inputs = returned_values[0].cuda(async=True)
        else:
            inputs = returned_values[0]

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


def get_inceptionscore(probs):   

    # no. images
    N = len(filenames)
    np_probs = []

    # Softmax the probabilties
    for i, item in enumerate(probs):
        np_probs.append(softmax(np.asarray(probs[i])))
    
    # Now compute the mean kl-div
    splits = 10 # hardcoded to confirm to https://github.com/openai/improved-gan/blob/master/inception_score/model.py
    split_scores = []

    scores = []
    for i in range(splits):
        print('Working on split ' + str(i) + '...')
        part =  np.asarray(np_probs[i * (N // splits): (i+1) * (N // splits)], dtype=np.float32)
        for imi in part:
            for i, probi in enumerate(imi):
                if probi == 0:
                    imi[i] += 0.000000000000000001
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

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
if task == 'letters':
    model = 'Z:/Inception/model_snapshots/initials.pt'
    class_dictionary = ['M', 'Q', 'C', 'N', 'P', 'O', 'D', 'T', 'H', 'S', 'V', 'B', 'A', 'W', 'I', 'E', 'L', 'F', 'R', 'Z', 'G', 'Y', 'X', 'K', '']
    class_count = 25
    loader_index = 1
elif task == 'countries':
    model = 'Z:/Inception/model_snapshots/countries.pt'
    class_dictionary = ['IT', 'DE', 'CH', 'FR', 'BE', 'POR', 'DK', 'NL', 'GB', 'ES', 'CZ']
    class_count = 11
    loader_index = 2
elif task == 'cities':
    model = 'Z:/Inception/model_snapshots/cities.pt'
    class_dictionary = ['IT', 'DE', 'CH', 'FR', 'BE', 'POR', 'DK', 'NL', 'GB', 'ES', 'CZ']
    class_count = 11
    loader_index = 2
elif task == 'names':
    model = 'Z:/Inception/model_snapshots/names.pt'
    class_dictionary = ['IT', 'DE', 'CH', 'FR', 'BE', 'POR', 'DK', 'NL', 'GB', 'ES', 'CZ']
    class_count = 11
    loader_index = 2
else:
    raise ValueError('Not a valid feature to classify, please pick from: \'letters\', \'countries\', \'cities\', \'names\'.')


# Start the testing
print('Starting the classification...')
print(torch.__version__)
testing_model = torch.load(model) 
probs, predictions = test_model(testing_model, test_loader, class_dictionary, loader_index, filenames)
print('Classification finished.')

# Performing inception scoring
if perform_incep:
    print('Starting calculation of inception score...')
    mean, std = get_inceptionscore(probs)
    print(mean)
    print(std)
    print('Inception score calculation finished.')