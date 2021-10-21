import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''

    true_args_pred = np.argwhere(prediction==True)
    false_args_pred = np.argwhere(prediction==False)

    TP = (prediction[true_args_pred] == ground_truth[true_args_pred]).sum()
    TN = (prediction[false_args_pred] == ground_truth[false_args_pred]).sum()
    FP = (prediction[true_args_pred] != ground_truth[true_args_pred]).sum()
    FN = (prediction[false_args_pred] != ground_truth[false_args_pred]).sum()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (len(prediction))
    f1 = precision * recall / (precision + recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy

    T = (prediction == ground_truth).sum()

    accuracy = T / len(prediction)

    return accuracy
