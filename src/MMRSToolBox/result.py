import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def cal_results(matrix):
    """
    calc overall accuracy, means of average accuracy, kappa, average accuracy

    :matrix: confusion matrix

    return: overall accuracy, means of average accuracy, kappa
    """
    shape = np.shape(matrix)
    number = 0
    line_sum = 0
    AA = np.zeros([shape[0]], dtype=float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        line_sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = line_sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)

    return OA, AA_mean, Kappa


def classification_results(target, prediction):
    """
    calc overall accuracy, means of average accuracy, kappa, average accuracy

    :tar: target, 1-D array-like of ground truth
    :pre: prediction, 1-D array-like of prediction

    return: overall accuracy, means of average accuracy, kappa quantity_disagreement,
    allocation_disagreement, f1_macro, f1_micro, f1_weighted, classification_report
    """
    matrix = confusion_matrix(target, prediction)
    OA, AA_mean, Kappa = cal_results(matrix)

    # calculate f1 scores
    f1_macro = f1_score(target, prediction, average='macro')
    f1_micro = f1_score(target, prediction, average='micro')
    f1_weighted = f1_score(target, prediction, average='weighted')

    # Calculate sums of rows and columns
    row_sums = matrix.sum(axis=1)  # Actual
    col_sums = matrix.sum(axis=0)  # Predicted
    total = matrix.sum()

    # Calculate quantity disagreement
    quantity_disagreement = 0.5 * np.sum(np.abs(row_sums - col_sums)) / total

    # Calculate allocation disagreement
    allocation_disagreement = (np.sum(np.abs(matrix - np.outer(row_sums, col_sums) / total)) / (2 * total))

    # get classification report
    classification = classification_report(target, prediction, digits=4)

    return OA, AA_mean, Kappa, quantity_disagreement, allocation_disagreement, f1_macro, f1_micro, f1_weighted, classification
