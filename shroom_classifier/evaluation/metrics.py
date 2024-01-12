from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from numpy import ndarray

def get_metrics(y_true: ndarray, y_hat: ndarray) -> (float, float, float, float, float):
    ''' Computes classification metrics:
         - Accuracy
        - Precision
        - Recall
        - F1-score
        - Support

        Args:
            y_true: True labels (N_CLASSES)
            y_hat: Predicted labels (N_CLASSES)

        Returns:
            accuracy: Accuracy score
            precision: Precision score
            recall: Recall score
            f1: F1 score
    '''
    
    accuracy = accuracy_score(y_true, y_hat)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_hat, average="macro", zero_division=0)

    return accuracy, precision, recall, f1, support