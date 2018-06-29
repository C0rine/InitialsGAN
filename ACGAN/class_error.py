import csv

counter = 0

# Open csv file
with open('Z:/CGANs/PyTorch-GAN/implementations/acgan/Z-Finished/results-names-S.csv') as csvfile:
    csv = csv.reader(csvfile, delimiter=',')
    for row in csv:
    	if row == []:
    		continue
    	else:
    		# get the part of the image name which represents the feature 
    		underscoreidx = row[0][:-4].rfind('_')
    		imagelabel = row[0][underscoreidx+1:-4]
    		# get the classification
    		classification = row[1]
    		
    		# if image label and classification are the same, increase the counter
    		if(imagelabel == classification):
    			counter += 1
	
# calculate the percentage of the full dataset which was correct classified
accuracy = counter * float(100) / float(50000) # and make sure this will be an accuracy float (thus probably cast all of the values to float)
print('==================================')
print('Accuracy: ' + str(accuracy))
print('Counter: ' + str(counter))