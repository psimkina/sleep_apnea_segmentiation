import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import auc, roc_curve, confusion_matrix

def draw_metrics(ypr, yvalid):
    """
    Draw ROC curves and confusion matrix. 
    Args: 
        ypr (np.array): predicted values
        yvalid (np.array): true values
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # plot the roc curve for the model
    fpr, tpr, _ = roc_curve(yvalid.ravel(), ypr.ravel())
    roc_auc = auc(fpr, tpr)

    ax[0].plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax[0].plot([0, 1], [0, 1], 'k--', color='red')
    ax[0].legend() 
    ax[0].grid()
    ax[0].set_xlabel('False positive rate')
    ax[0].set_ylabel('True positive rate')

    # plot the confusion matrix
    cf_matrix = confusion_matrix(yvalid.flatten(), ypr.flatten() > 0.5)

    cf_matrix = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cf_matrix, annot=True, ax=ax[1], cmap='Blues', fmt='.2%')
    ax[1].set_xlabel('Predicted label')
    ax[1].set_ylabel('True label')