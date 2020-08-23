from data_loader import rand_test_data
from data_loader import rand_train_data
from model import sequential_model
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
# ------------------------------------------------
# TODO: Neural Network Predictions with TensorFlow's Keras API
print('----------------------------------------------------------------------')
print('[INFO] Predicting in probability for labels using test data....')
print('----------------------------------------------------------------------')
predictions = sequential_model.NeuNetmodel.predict(x=rand_test_data.scaled_test_samples, batch_size=10, verbose=0)
print('[INFO] Rounding off the prediction set into `YES-1` or `NO-0` labels.....')
print('----------------------------------------------------------------------')
rounded_predictions = np.argmax(predictions, axis=-1)
print('First 10 instance of label from predicted dataset....')
print('----------------------------------------------------------------------')
print(rounded_predictions[:20])

# TODO: Create a Confusion Matrix for Neural Network Predictions
cm = confusion_matrix(y_true=rand_test_data.test_labels, y_pred=rounded_predictions)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm_plot_labels = ['no_side_effects','had_side_effects']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
plt.show()