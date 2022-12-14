import sys
sys.path.append(r'./genetic/code/metrics')
from binary_confusion_matrix import get_binary_confusion_matrix, get_threshold_binary_confusion_matrix
from binary_statistical_metrics import get_accuracy, get_true_positive_rate, get_true_negative_rate, \
    get_precision, get_f1_socre, get_iou
from dice_coefficient import hard_dice
sys.path.append(r'./genetic/code/metrics/pr_curve.py')
from pr_curve import get_pr_curve
sys.path.append('./genetic/code/metrics/calculate_metrics.py')
from roc_curve import get_auroc, get_roc_curve
sys.path.append('./genetic/code/util')
from numpy_utils import tensor2numpy
import numpy as np
from copy import deepcopy

# np.seterr(divide='ignore', invalid='ignore')
import sys
sys.path.append('./genetic/code/train/train_model.py')
def calculate_metrics( preds, targets, device,config=None):
    curr_TP, curr_FP, curr_TN, curr_FN = get_binary_confusion_matrix(
        input_=preds, target=targets, device=device, pixel=0,
        threshold=0.5,
        reduction='sum')

    curr_acc = get_accuracy(true_positive=curr_TP,
                            false_positive=curr_FP,
                            true_negative=curr_TN,
                            false_negative=curr_FN)

    curr_recall = get_true_positive_rate(true_positive=curr_TP,
                                         false_negative=curr_FN)

    curr_specificity = get_true_negative_rate(false_positive=curr_FP,
                                              true_negative=curr_TN)

    curr_precision = get_precision(true_positive=curr_TP,
                                   false_positive=curr_FP)

    curr_f1_score = get_f1_socre(true_positive=curr_TP,
                                 false_positive=curr_FP,
                                 false_negative=curr_FN)

    curr_iou = get_iou(true_positive=curr_TP,
                       false_positive=curr_FP,
                       false_negative=curr_FN)

    curr_auroc = get_auroc(preds, targets)

    return (curr_acc, curr_recall, curr_specificity, curr_precision,
            curr_f1_score, curr_iou, curr_auroc)

