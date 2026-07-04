import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

def calculate_metrics(y_true, y_pred, y_probs=None):
    """Calculate multi-label classification metrics"""
    
    metrics = {
        'accuracy': np.mean(y_true == y_pred) * 100,
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1': f1_score(y_true, y_pred, average='macro')
    }
    
    if y_probs is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_probs, average='macro')
        except:
            metrics['auc'] = 0
    
    return metrics

def clean_report(report):
    """Clean generated report text"""
    # Capitalize first letter
    if report:
        report = report[0].upper() + report[1:]
    
    # Add period if missing
    if report and report[-1] not in ['.', '!', '?']:
        report += '.'
    
    return report