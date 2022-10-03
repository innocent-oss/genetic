"""
Receiver operating characteristic curve and area under it
"""

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import sys
sys.path.append('./genetic/code/util/numpy_utils.py')
from numpy_utils import flatten_tensor

sys.path.append('./genetic/code/metrics')
def get_roc_curve(preds, targets):
    """
    Get ROC curve

    Arguments:
        preds: raw probability outputs
        targets: ground truth

    Returns:
        fpr: false positive rate
        tpr: true positive rate
        thresholds: thresholds
    """
    preds, targets = list(map(flatten_tensor, [preds, targets]))

    fpr, tpr, thresholds = roc_curve(y_true=targets,
                                     y_score=preds,
                                     pos_label=None,
                                     sample_weight=None,
                                     drop_intermediate=True)

    return fpr, tpr, thresholds

sys.path.append('./genetic/code/metrics')

def get_auroc(preds, targets):
    """
    Get Area under ROC curve

    Arguments:
        preds: raw probability outputs
        targets: ground truth

    Returns:
        auroc: the area under ROC curve
    """
    preds, targets = list(map(flatten_tensor, [preds, targets]))

    auroc = roc_auc_score(y_true=targets,
                          y_score=preds,
                          average='macro',
                          sample_weight=None,
                          max_fpr=None)
    return auroc
