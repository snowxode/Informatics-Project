
#  Basic evaluation


import Task_1_5
import Dummy
import numpy as np

classification_scheme = Task_1_5.classification_scheme

def confusionMatrix(classified_data):
    # Initialize the confusion matrix to all zeros
    confusion_matrix = np.zeros((len(classification_scheme), len(classification_scheme)))

    # Iterate through the classified data and update the confusion matrix
    for row in range(1, len(classified_data)):
        # Some simple error checking
        try:
            # Get the indices of the actual and predicted classes
            actual_index = classification_scheme.index(classified_data[row][1])
            predicted_index = classification_scheme.index(classified_data[row][2])
            
            # Add 1 to the appropriate cell in the confusion matrix
            confusion_matrix[predicted_index][actual_index] += 1
            
        except ValueError:
            print("ValueError at: ", row)
    return confusion_matrix


def computeTPs(confusion_matrix):
    # Initialize the list of true positives for each category
    tps = []
    
    for x in range(0, len(classification_scheme)):
        # Append to the list each number of true positives (diagonal of the confusion matrix )
        tps.append(confusion_matrix[x][x])
        
    return tps

def computeFPs(confusion_matrix):
    # Initialize the list of false positives for each category
    fps = []
    
    for x in range(0, len(classification_scheme)):
        # Append to the list each number of false positives (sum of the column - diagonal of the confusion matrix )
        fp = np.sum(confusion_matrix, axis=0)
        fps.append(fp[x] - confusion_matrix[x][x])
    return fps

def computeFNs(confusion_matrix):
    # Initialize the list of false negatives for each category
    fns = []
    
    for x in range(0, len(classification_scheme)):
        # Append to the list each number of false negatives (sum of the row - diagonal of the confusion matrix )
        fn = np.sum(confusion_matrix, axis=1)
        fns.append(fn[x] - confusion_matrix[x][x])
    return fns


def computeMacroPrecision(tps, fps, fns, data_size):
    # Initialize the precision
    precision = float(0)

    for i in range(len(tps)):
        # Compute the precision for each class and add it to the total precision
        microprecision = tps[i] / (tps[i] + fps[i])
        precision += microprecision
        
    # Compute the average precision by dividing the total precision by the number of classes
    precision = precision/len(tps)
    return precision

def computeMacroRecall(tps, fps, fns, data_size):
    # Initialize the recall
    recall = float(0)
    
    for i in range(len(tps)):
        # Compute the recall for each class and add it to the total recall
        microrecall = tps[i] / (tps[i] + fns[i])
        recall += microrecall

    # Compute the average recall by dividing the total recall by the number of classes
    recall = recall/len(tps)
    return recall

def computeMacroFMeasure(tps, fps, fns, data_size):
        # Initialize the f-measure
    f_measure = float(0)
    
    for i in range(len(tps)):
        # Compute the f-measure for each class and add it to the total f-measure
        f_measure += tps[i] / (tps[i] + 0.5 * (fps[i] + fns[i]))
    
    # Find the average f-measure by dividing the total f-measure by the number of classes
    f_measure = f_measure / len(tps)
    return f_measure
    
    # The equation (2 x p x r / p + r) does not work for some reason,
    # so i used the equation (tp / (tp + 0.5 * (fp + fn))) instead
    # The code below is the original code that using the first equation
'''
def computeMacroFMeasure(tps, fps, fns, data_size):
    # Initialize the f-measure
    f_measure = float(0)
    
    for i in range(len(tps)):
    
        # Compute the precision and recall for each class
        p = float(tps[i] / (tps[i] + fps[i]))
        r = float(tps[i] / (tps[i] + fns[i]))
        try:
            # Compute the f-measure for each class and add it to the total f-measure
            f_measure += float(( 2 * p * r)/(p + r))
            
        # If either the precision and recall are 0, the f-measure is 0 (as division by 0 is not possible)
        except ZeroDivisionError:
            f_measure = float(0)
        
    # Compute the average f-measure by dividing the total f-measure by the number of classes
    f_measure = f_measure/len(tps)
    return f_measure
'''


def computeAccuracy(tps, fps, fns, data_size):
    # Initialize the list of true negatives
    tns = []
    # Compute the true negatives for each class
    for i in range(len(fps)):
        # True negatives = total data size - (false positives + false negatives)
        fn_and_fp = fps[i] + fns[i]
        tns.append(data_size - fn_and_fp)
        
    # Compute the accuracy 
    accuracy = float((sum(tps) + sum(tns)) / (sum(tps) + sum(fps) + sum(fns) + sum(tns)))
    return accuracy

def evaluateKNN(classified_data, confusion_func=confusionMatrix):
    # First, we compute the confusion matrix
    cm = confusion_func(classified_data)
    
    # Find the data size
    data_size = len(classified_data)- 1 # -1 for header
    
    # Compute the true positives, false positives and false negatives
    tps, fps, fns, = computeTPs(cm), computeFPs(cm), computeFNs(cm)
    
    # Compute the evaluation measures
    precision = computeMacroPrecision(tps, fps, fns, data_size)
    recall = computeMacroRecall(tps, fps, fns, data_size)
    f_measure = computeMacroFMeasure(tps, fps, fns, data_size)
    accuracy = computeAccuracy(tps, fps, fns, data_size)

    # Return the computed measures
    return precision, recall, f_measure, accuracy


def main():
    opts = Task_1_5.parseArguments()
    if not opts:
        exit(1)
    print(f'Reading data from {opts["classified_data"]}')
    classified_data = Task_1_5.readCSVFile(opts['classified_data'])
    print('Evaluating kNN')
    result = evaluateKNN(classified_data, eval(opts['cf']))
    print('Result: precision {}; recall {}; f-measure {}; accuracy {}'.format(*result))


if __name__ == '__main__':
    main()
