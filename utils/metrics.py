"""
Classification metrics and training utility functions.

Provides rich evaluation metrics (accuracy, precision, recall, F1, AUC,
per-class sensitivity/specificity, confusion matrix formatting)
and training infrastructure (AverageMeter, EarlyStopping, checkpointing).
"""

import torch
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix)


def calculate_metrics(y_true, y_pred, y_prob=None, num_classes=4):
    """
    Calculate comprehensive classification metrics (supports multi-class).

    Args:
        y_true: Ground truth labels (list or array)
        y_pred: Predicted labels (list or array)
        y_prob: Predicted probabilities (for AUC), shape [N, num_classes]
        num_classes: Number of classes

    Returns:
        dict with keys: accuracy, precision, recall, f1, f1_macro, auc, auc_macro,
                        confusion_matrix, plus per-class metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }

    # Per-class metrics
    for i in range(num_classes):
        y_true_bin = (y_true == i).astype(int)
        y_pred_bin = (y_pred == i).astype(int)
        tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
        tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
        fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
        fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        metrics[f'sensitivity_class_{i}'] = sensitivity
        metrics[f'specificity_class_{i}'] = specificity
        metrics[f'precision_class_{i}'] = prec

    # Macro averages of per-class metrics
    metrics['sensitivity_macro'] = np.mean([
        metrics.get(f'sensitivity_class_{i}', 0) for i in range(num_classes)
    ])
    metrics['specificity_macro'] = np.mean([
        metrics.get(f'specificity_class_{i}', 0) for i in range(num_classes)
    ])

    # Binary: expose tumor (class 1) as the positive class.  For binary the
    # *_macro values are both equal to balanced accuracy (sensitivity_macro ==
    # specificity_macro identically), so a clinically meaningful sensitivity /
    # specificity must be reported with tumor=1 as positive.
    if num_classes == 2:
        metrics['sensitivity'] = metrics.get('sensitivity_class_1', 0.0)
        metrics['specificity'] = metrics.get('specificity_class_1', 0.0)

    # AUC
    if y_prob is not None:
        y_prob = np.array(y_prob)
        try:
            if num_classes == 2:
                metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
                metrics['auc_macro'] = metrics['auc']
            else:
                from sklearn.preprocessing import label_binarize
                y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
                metrics['auc'] = roc_auc_score(
                    y_true_bin, y_prob, multi_class='ovr', average='macro'
                )
                metrics['auc_macro'] = metrics['auc']
        except Exception:
            metrics['auc'] = 0.5
            metrics['auc_macro'] = 0.5

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(
        y_true, y_pred, labels=list(range(num_classes))
    )

    return metrics


def format_confusion_matrix(cm, class_names=None):
    """
    Format a confusion matrix as a human-readable string.

    Args:
        cm: 2D confusion matrix (numpy array)
        class_names: Optional list of class name strings

    Returns:
        Formatted string with aligned columns
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]

    # Find max width needed
    max_name_len = max(len(n) for n in class_names)
    cell_width = max(max_name_len + 2, 8)

    # Header
    header = " " * (max_name_len + 3)
    for name in class_names:
        header += f"{name:>{cell_width}}"
    lines = [header]

    # Rows
    for i, (name, row) in enumerate(zip(class_names, cm)):
        line = f"{name:>{max_name_len + 2}}"
        for j, val in enumerate(row):
            marker = " *" if i == j else "  "
            line += f"{marker}{val:>{cell_width - 2}d}"
        lines.append(line)

    return "\n".join(lines)


def five_scores(bag_labels, bag_predictions, use_auc=True):
    """
    Legacy: Calculate accuracy, precision, recall, f1, auc for binary classification.

    Args:
        bag_labels: Ground truth labels
        bag_predictions: Predicted probabilities/scores
        use_auc: Whether to compute AUC

    Returns:
        (accuracy, auc, precision, recall, fscore)
    """
    bag_labels = np.array(bag_labels)
    bag_predictions = np.array(bag_predictions)

    bag_predictions_binary = (bag_predictions >= 0.5).astype(int)

    accuracy = accuracy_score(bag_labels, bag_predictions_binary)
    precision = precision_score(bag_labels, bag_predictions_binary, zero_division=0)
    recall = recall_score(bag_labels, bag_predictions_binary, zero_division=0)
    fscore = f1_score(bag_labels, bag_predictions_binary, zero_division=0)

    if use_auc:
        try:
            auc = roc_auc_score(bag_labels, bag_predictions)
        except Exception:
            auc = 0.5
    else:
        auc = 0.0

    return accuracy, auc, precision, recall, fscore


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping based on validation metric improvement."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience: Number of epochs with no improvement before stopping
            stop_epoch: Earliest epoch at which stopping is allowed
            verbose: Whether to print messages
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch >= self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def state_dict(self):
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
            'val_loss_min': self.val_loss_min
        }

    def load_state_dict(self, state_dict):
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.early_stop = state_dict['early_stop']
        self.val_loss_min = state_dict['val_loss_min']


def save_checkpoint(model, optimizer, epoch, save_path, **kwargs):
    """Save a model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    checkpoint.update(kwargs)

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, checkpoint_path, optimizer=None, device='cuda'):
    """Load a model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)

    return model, optimizer, epoch
