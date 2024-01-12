from shroom_classifier.evaluation.metrics import get_metrics
import numpy as np



def test_get_metrics():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_hat = np.array([0, 2, 1, 0, 0, 1])
    
    true_accuracy = np.array([1, 0, 0, 1, 0, 0]).mean()
    true_precision = np.array([2/3, 0 ,0]).mean()
    true_recall = np.array([1, 0, 0]).mean()
    true_f1 = 2 * true_precision * true_recall / (true_precision + true_recall)
    
    accuracy, precision, recall, f1 = get_metrics(y_true, y_hat)
    
    print("accuracy:", accuracy)
    print("precision:", precision)
    print("recall:", recall)
    print("f1:", f1)
    
    assert np.isclose(accuracy, true_accuracy), f"accuracy: {accuracy}, true_accuracy: {true_accuracy}"
    assert np.isclose(precision, true_precision), f"precision: {precision}, true_precision: {true_precision}"
    assert np.isclose(recall, true_recall), f"recall: {recall}, true_recall: {true_recall}"
    assert np.isclose(f1, true_f1), f"f1: {f1}, true_f1: {true_f1}"


if __name__ == "__main__":
    test_get_metrics()