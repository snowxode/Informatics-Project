import distutils.util
import argparse
import csv
import os
import Dummy
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sewar.full_ref import mse
import numpy as np
import cv2
from tqdm import tqdm


classification_scheme = ['Female', 'Male', 'Primate', 'Rodent', 'Food']


def validateDataFormat(data, predicted):
    # Check if there are any entries in the data
    if not data:
        return False
    
    required_columns = ["Path", "ActualClass"]
    
    # If predicted is True, add PredictedClass to required columns
    if predicted:
        required_columns.append("PredictedClass")
    
    # Check if the required columns are present
    if data[0] != required_columns:
        return False
    
    # Check if the values in the columns are valid
    for row in data[1:]:

        # Check if the path is valid
        if not os.path.isfile(row[0]):
            return False
        
        # check if the actual class is valid
        if row[1] not in classification_scheme:
            return False

        # Check if the predicted class is valid
        if predicted and row[2] not in classification_scheme:
            return False
    
    return True



def readAndResize(image_path, width=60, height=30):
    image = []

    try:
        # Read image
        image = cv2.imread(image_path)
        
        # Resize the image
        image = cv2.resize(image, (width, height))
        
        
    except FileNotFoundError:
        print(f"File not found: {image_path}")
    
    return image


def computeMeasure1(image1, image2):
    # Cosine Similarity
    if image1 is None or image2 is None:
        return float('nan')
    sim = cosine_similarity(np.array([image1.flatten()]), np.array([image2.flatten()]))
    return sim

def computeMeasure2(image1, image2):
    # Euclidean Distance
    if image1 is None or image2 is None:
        return float('nan')
    dist = euclidean_distances(np.array([image1.flatten()]), np.array([image2.flatten()]))
    return dist


def computeMeasure3(image1, image2):
    # Mean Squared Error
    if image1 is None or image2 is None:
        return float('nan')
    dist = mse(image1, image2)
    return dist


def selfComputeMeasure1(image1, image2):
    # Cosine Similarity
    # Dot product helper function to replace numpy.dot()
    def dp(image1, image2):
        dot_product = float(0)
        for i in range(len(image1)):
            dot_product += image1[i] * image2[i]
        return dot_product
    
    # Check if the images are empty
    if image1 is None or image2 is None:
        return float('nan')
    
    # Flatten the images from 3D to 1D
    image1 = np.array(image1.flatten()).tolist() 
    image2 = np.array(image2.flatten()).tolist() 
    
    # Compute the dot product of the two images
    dot_product = dp(image1, image2)
    
    # Compute the norms for the two images
    norm_1 = np.sqrt(dp(image1, image1))
    norm_2 = np.sqrt(dp(image2, image2))

    # Compute the cosine similarity equation and return the result
    cosine_sim = dot_product / (norm_1 * norm_2)
    return cosine_sim


def selfComputeMeasure2(image1, image2):
    # Euclidean Distance
    # Check if the images are empty
    if image1 is None or image2 is None:
        return float('nan')
    
    # Compute the euclidean distance using the equation and return the result
    euclidean_dist = np.sqrt(np.sum(np.square(image1 - image2)))
    return euclidean_dist

def getClassesOfKNearestNeighbours(measures_classes, k, similarity_flag):
    nearest_neighbours_classes = {}
    
    # Sort the measures_classes list based on the distance/similarity values, in descending order if similarity_flag is True
    measures_classes = sorted(measures_classes, key=lambda x: x[0], reverse=similarity_flag)

    # loop over the k nearest neighbours and count the occurrences of each class
    for i in range(k):
        # break early if we have reached the end of the list (doubtful, but just in case)
        if i >= len(measures_classes):
            break
        
        # Get the class label of the current neighbour
        class_label = measures_classes[i][1]
        
        # Add the class to the dictionary if it is not there yet, otherwise increment the counter
        if class_label in nearest_neighbours_classes:
            nearest_neighbours_classes[class_label] += 1
        else:
            nearest_neighbours_classes[class_label] = 1

    return nearest_neighbours_classes

def getMostCommonClass(nearest_neighbours_classes):
    winner = ''
    # If the dictionary is empty, return empty string
    if bool(nearest_neighbours_classes) == False:
        return winner
    
    # Get the list of class labels in the dictionary
    nn_class_scheme = list(nearest_neighbours_classes.keys())
    
    # Check if it does not contain any classes from the scheme
    if set(nn_class_scheme).isdisjoint(set(classification_scheme)):
        return winner
    
    # Check that all classes have occurrence of 0
    if all(value == 0 for value in nearest_neighbours_classes.values()):
        return winner
    
    # loop over the class labels in the scheme and find the most common one
    max_occurrence = -1 #-1 is an integer that cannot be reached by the number of occurrences of a class
    
    for key in nn_class_scheme:
        # If the label is not in the classification scheme, skip it
        if key not in classification_scheme:
            continue
        
        # Get the occurence value from dictionary
        occurrence = nearest_neighbours_classes[key]

        # If the occurrence is 0, skip it (its not going to be a winner)
        if occurrence == 0:
            continue
        
        # If the occurrence is greater than the current max, update the max and the winner
        elif occurrence > max_occurrence:
            max_occurrence = occurrence
            winner = key
            
        # If the occurrence is equal to the current max, tie-breaker
        elif occurrence == max_occurrence:
            # Get the index of the current class and the winner in the scheme order
            winner_index = classification_scheme.index(winner)
            current_index = classification_scheme.index(key)
            
            # If the current class index is before the winner in the scheme order, update the winner
            if current_index < winner_index:
                winner = key
    
    return winner

def kNN(training_data, k, measure_func, similarity_flag, data_to_classify,
        most_common_class_func=getMostCommonClass, get_neighbour_classes_func=getClassesOfKNearestNeighbours,
        read_func=readAndResize):
    #print(most_common_class_func({"a": 0, "b": 0, "Food": 0, "Primate": 12, "d": 0})) #debugging 
    #exit(1)
    
    # This sets the header list
    classified_data = [['Path', 'ActualClass', 'PredictedClass']]
    
    # Data validation
    if(len(data_to_classify[0])) == 2:
        if not validateDataFormat(data_to_classify, False) or not validateDataFormat(training_data, False):
            return classified_data
    elif (len(data_to_classify[0])) > 2:
        if not validateDataFormat(data_to_classify, True) or not validateDataFormat(training_data, True):
            return classified_data
    
    # Loop through each image to classify
    for i in tqdm(range(1, len(data_to_classify)), desc='Classifying Images', position=0):
        # Read the image and resize it for comparisons
        img_data = read_func(data_to_classify[i][0])

        # Check that the image was read successfully if not skip, it
        if not img_data.any():
            continue
        
        # Create a list to store the distances/similarities and classes for each training image
        measures_classes = []
        
        # Loop through each training image and calculate the distance/similarity to the image to classify
        for j in range(1, len(training_data)):
            train_data = read_func(training_data[j][0])
            
            # Check that the training image was read successfully if not, skip it
            if not train_data.any():
                continue

            # Calculate the distance/similarity between the training image and the image to classify
            measure = measure_func(train_data, img_data)
            
            # Add the distance/similarity and the class of the training image to the list
            measures_classes.append([measure, training_data[j][1]])
        
        # Get the classes of the k nearest neighbours
        nn_classes = get_neighbour_classes_func(measures_classes, k, similarity_flag)
        
        # Get the most common class among the k nearest neighbours
        predicted_class = most_common_class_func(nn_classes)
        
        # Add the image path, actual class, and predicted class to the classified_data list
        classified_data.append([data_to_classify[i][0], data_to_classify[i][1], predicted_class])
    
    return classified_data


def main():
    opts = parseArguments()
    if not opts:
        exit(1)
    print(f'Reading data from {opts["training_data"]} and {opts["data_to_classify"]}')
    training_data = readCSVFile(opts['training_data'])
    data_to_classify = readCSVFile(opts['data_to_classify'])
    unseen = opts['mode']
    print('Running kNN')
    print(opts['simflag'])
    result = kNN(training_data, opts['k'], eval(opts['measure']), opts['simflag'], data_to_classify,
                 eval(opts['mcc']), eval(opts['gnc']), eval(opts['rrf']))
    if unseen:
        path = os.path.dirname(os.path.realpath(opts['data_to_classify']))
        out = f'{path}/210_classified_data.csv'
        print(f'Writing data to {out}')
        writeCSVFile(out, result)


# Straightforward function to read the data contained in the file "filename"
def readCSVFile(filename):
    lines = []
    with open(filename, newline='') as infile:
        reader = csv.reader(infile)
        for line in reader:
            lines.append(line)
    return lines


# Straightforward function to write the data contained in "lines" to a file "filename"
def writeCSVFile(filename, lines):
    with open(filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(lines)


# This function simply parses the arguments passed to main. It looks for the following:
#       -k              : the value of k neighbours
#                         (needed in Tasks 1, 2, 3 and 5)
#       -f              : the number of folds to be used for cross-validation
#                         (needed in Task 3)
#       -measure        : function to compute a given similarity/distance measure
#       -simflag        : flag telling us whether the above measure is a distance (False) or similarity (True)
#       -u              : flag for how to understand the data. If -u is used, it means data is "unseen" and
#                         the classification will be written to the file. If -u is not used, it means the data is
#                         for training purposes and no writing to files will happen.
#                         (needed in Tasks 1, 3 and 5)
#       training_data   : csv file to be used for training the classifier, contains two columns: "Path" that denotes
#                         the path to a given image file, and "Class" that gives the true class of the image
#                         according to the classification scheme defined at the start of this file.
#                         (needed in Tasks 1, 2, 3 and 5)
#       data_to_classify: csv file formatted the same way as training_data; it will NOT be used for training
#                         the classifier, but for running and testing it
#                         (needed in Tasks 1, 2, 3 and 5)

def parseArguments():
    parser = argparse.ArgumentParser(description='Processes files ')
    parser.add_argument('-k', type=int)
    parser.add_argument('-f', type=int)
    parser.add_argument('-m', '--measure')
    parser.add_argument('-s', '--simflag', type=lambda x:bool(distutils.util.strtobool(x)))
    parser.add_argument('-u', '--unseen', action='store_true')
    parser.add_argument('-train', type=str)
    parser.add_argument('-test', type=str)
    parser.add_argument('-classified', type=str)
    parser.add_argument('-mcc', default="getMostCommonClass")
    parser.add_argument('-gnc', default="getClassesOfKNearestNeighbours")
    parser.add_argument('-rrf', default="readAndResize")
    parser.add_argument('-cf', default="confusionMatrix")
    parser.add_argument('-sf', default="splitDataForCrossValidation")
    parser.add_argument('-al', default="Task_1_5.kNN")
    params = parser.parse_args()

    opt = {'k': params.k,
           'f': params.f,
           'measure': params.measure,
           'simflag': params.simflag,
           'training_data': params.train,
           'data_to_classify': params.test,
           'classified_data': params.classified,
           'mode': params.unseen,
           'mcc': params.mcc,
           'gnc': params.gnc,
           'rrf': params.rrf,
           'cf': params.cf,
           'sf': params.sf,
           'al': params.al
           }
    return opt


if __name__ == '__main__':
    main()
