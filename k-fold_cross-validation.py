# In this file please complete the following task:
#
# Task 3 [6] Cross validation
#
# Evaluate your classifiers using the k-fold cross-validation technique covered in the lectures (use the training
# data only). Output their average precisions, recalls, F-measures and accuracies. You need to implement the
# validation yourself. Remember that folds need to be of roughly equal size. The template contains a range of
# functions you need to implement for this task.

import os
import classifier
import basic_eval
import Dummy
from tqdm import tqdm
import numpy as np
import random
from classifier import computeMeasure1,computeMeasure2,computeMeasure3,selfComputeMeasure1,selfComputeMeasure2


# This function takes the data for cross evaluation and returns training_data a list of lists s.t. the first element
# is the round number, second is the training data for that round, and third is the testing data for that round
#
# INPUT: training_data      : a list of lists that was read from the training data csv (see parse_arguments function)
#        f                  : the number of folds to split the data into (which is also same as # of rounds)
# OUTPUT: folds             : a list of lists s.t. the first element is the round number, second is the training data
#                             for that round, and third is the testing data for that round

def splitDataForCrossValidation(training_data, f):
    # Pop off the header:
    header = training_data.pop(0)
    
    # Randomise the data
    random.shuffle(training_data)
    
    # Get the size of each fold 
    fold_size = len(training_data) // f
    
    # Find out how many folds need to be one element larger to cover the remaining data
    remaining_data = len(training_data) % f
    
    # Initialize the list of folds
    folds = []
    
    start_index = 0
    for fold_num in range(f):
        
        # Get the end index for this fold
        if fold_num < remaining_data:
            # For the first remaining_data folds, make them one element larger
            # until all the remaining data is used up
            end_index = start_index + fold_size + 1
        else:
            end_index = start_index + fold_size
            
        # Get the training and testing data for this fold and add the header
        testing_data_fold = training_data[start_index:end_index]
        testing_data_fold.insert(0, header)
        
        training_data_fold = training_data[:start_index] + training_data[end_index:]
        training_data_fold.insert(0, header)
        # Add this fold to the list of folds
        folds.append([fold_num, training_data_fold, testing_data_fold])
        
        # Update start index
        start_index = end_index
    
    # Return the new data sectioned into folds
    return folds


# In this function, please implement validation of the data that is produced by the cross evaluation function PRIOR to
# the addition of rows with the average meaasures.
#
# INPUT:  data              : a list of lists that was produced by the crossEvaluateKNN function
#         f                 : number of folds to validate against
#
# OUTPUT: boolean value     : True if the data contains the header ["Path", "ActualClass", "PredictedClass","FoldNumber"]
#                             (there can be more column names, but at least these four at the start must be present)
#                             AND the values in the "Path" column (if there are any) are file paths
#                             AND the values in the "ActualClass" and "PredictedClass" columns
#                             (if there are any) are classes from the scheme
#                             AND the values in the "FoldNumber" column are integers in [0,f) range
#                             AND there are as many Path entries as ActualClass and PredictedClass and FoldNumber entries
#                             AND the number of entries per each integer in [0,f) range for FoldNumber are approximately
#                             the same (they can differ by at most 1)
#
#                             False otherwise

def validateDataFormat(data, f):
    # Initialize the formatCorrect variable
    formatCorrect = True
    
    # Check if the header is correct by checking if the first row contains the correct headers
    headers = ["Path", "ActualClass", "PredictedClass","FoldNumber"]
    if not all(header in data[0] for header in headers):
        return False
    
    fold_list = np.zeros(f, dtype=int)
    for row in range(1, len(data)):
        # Check that all the values in the "Path" column are file paths
        if not os.path.isfile(data[row][0]):
            return False
        
        # Check that all the values in the "ActualClass" and "PredictedClass" columns are classes from the scheme
        if data[row][1] not in classifier.classification_scheme or data[row][2] not in classifier.classification_scheme:
            return False  

        # Check that all the values in the "FoldNumber" column are integers in [0,f) range
        # Accepts strings in the form of integers or integers/floats
        try:
            if data[row][3].isnumeric():
                if int(data[row][3]) not in range(0, f+1):
                    return False
            else:
                return False
        except AttributeError:
            if int(data[row][3]) not in range(0, f+1):
                return False
            
        # Add 1 to the list of fold sizes indexed by the fold number
        fold_list[int(data[row][3])] += 1
        
    # Make sure that the fold numbers differ by at most 1
    asc_order_fold_list = sorted(fold_list)
    if (asc_order_fold_list[f-1] - asc_order_fold_list[0]) not in [0, 1]:
        return False
    
    return formatCorrect


# This function takes the classified data from each cross validation round and calculates the average precision, recall,
# accuracy and f-measure for them.
# Invoke either the Task 2 evaluation function or the dummy function here, do not code from scratch!
#
# INPUT: classified_data_list
#                           : a tuple consisting of the classified data computed for each cross validation round
#        evaluation_func    : the function to be invoked for the evaluation (by default, it is the one from
#                             basic_eval, but you can use dummy)
# OUTPUT: avg_precision, avg_recall, avg_f_measure, avg_accuracy
#                           : average evaluation measures. You are expected to evaluate every classified data in the
#                             tuple and average out these values in the usual way.

def evaluateCrossValidation(*classified_data_list, evaluation_func=basic_eval.evaluateKNN):
    # There are multiple ways to count average measures during cross-validation. For the purpose of this portfolio,
    # it's fine to just compute the values for each round and average them out in the usual way.
    
    # Is this just? ... 
    #avg_precision, avg_recall, avg_f_measure, avg_accuracy = evaluation_func(classified_data_list[0])
    avg_precision, avg_recall, avg_f_measure, avg_accuracy = float(0), float(0), float(0), float(0)
    
    for i in range(1, len(classified_data_list[0])):
        precision, recall, f_measure, accuracy = evaluation_func(classified_data_list[0])
        avg_precision += precision
        avg_recall += recall
        avg_f_measure += f_measure
        avg_accuracy += accuracy

    
    avg_precision = avg_precision/len(classified_data_list[0])
    avg_recall = avg_recall/len(classified_data_list[0])
    avg_f_measure = avg_f_measure/len(classified_data_list[0])
    avg_accuracy = avg_accuracy/len(classified_data_list[0])
    
    return avg_precision, avg_recall, avg_f_measure, avg_accuracy


# In this task you are expected to perform cross-validation where f defines the number of folds to consider.
# "processed" holds the information from training data along with the following information: for each image,
# stated the id of the fold it landed in, and the predicted class it was assigned once it was chosen for testing data.
# After everything is done, we add the average measures at the end. The writing to csv is done in a different function.
# You are expected to invoke the Task 1 kNN classifier or the dummy classifier here, do not implement these things
# from scratch!
#
# INPUT: training_data      : a list of lists that was read from the training data csv (see parse_arguments function)
#        k                  : the value of k neighbours, to be passed to the kNN classifier
#        measure_func       : the function to be invoked to calculate similarity/distance
#        similarity_flag    : a boolean value stating that the measure above used to produce the values is a distance
#                             (False) or a similarity (True)
#        knn_func           : the function to be invoked for the classification (by default, it is the one from
#                             classifier, but you can use dummy)
#        split_func         : the function used to split data for cross validation (by default, it is the one above)
#        f                  : number of folds to use in cross validation
# OUTPUT: processed+r       : a list of lists which expands the training_data with columns stating the fold number to
#                             which a given image was assigned and the predicted class for that image; and with rows
#                             that contain the average evaluation measures
# Again, please remember to have a look at the Dummy file!
def crossEvaluateKNN(training_data, k, measure_func, similarity_flag, f, knn_func=classifier.kNN,
                     split_func=splitDataForCrossValidation):
    #validateDataFormat(training_data, f)
    #exit(1)
    # This adds the header
    processed = [['Path', 'ActualClass', 'PredictedClass', 'FoldNumber']]
    avg_precision = -1.0;
    avg_recall = -1.0;
    avg_fMeasure = -1.0;
    avg_accuracy = -1.0;

    # Have fun with the computations!
    data_folds = split_func(training_data, f)
    for i in range(len(data_folds)):
        fold_num = data_folds[i][0]

        classified_data = knn_func(data_folds[i][1], k, measure_func, similarity_flag, data_folds[i][2])
        
        for j in range(1, len(classified_data)):
            processed.append([classified_data[j][0], classified_data[j][1], classified_data[j][2], fold_num])
        
    if validateDataFormat(processed, f):
        avg_precision, avg_recall, avg_fMeasure, avg_accuracy = evaluateCrossValidation(processed)
    else:
        print("Data not validated")
        
        
    # The measures are now added to the end:
    h = ['avg_precision', 'avg_recall', 'avg_f_measure', 'avg_accuracy']
    v = [avg_precision, avg_recall, avg_fMeasure, avg_accuracy]
    r = [[h[i], v[i]] for i in range(len(h))]

    return processed + r


# This function reads the necessary arguments (see parse_arguments function in classifier),
# and based on them evaluates the kNN classifier using the cross-validation technique. The results
# are written into an appropriate csv file.
def main():
    opts = classifier.parseArguments()
    if not opts:
        exit(1)
    print(f'Reading data from {opts["training_data"]}')
    training_data = classifier.readCSVFile(opts['training_data'])
    print('Evaluating kNN')
    result = crossEvaluateKNN(training_data, opts['k'], eval(opts['measure']), opts['simflag'], opts['f'],
                              eval(opts['al']), eval(opts['sf']))
    path = os.path.dirname(os.path.realpath(opts['training_data']))
    out = f'{path}/{classifier.student_id}_cross_validation.csv'
    print(f'Writing data to {out}')
    classifier.writeCSVFile(out, result)


if __name__ == '__main__':
    main()
