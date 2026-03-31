import numpy as np
from scipy.optimize import minimize_scalar


#----- Temperature scaling for binary classification calibration -----

def fit_temperature(logits_true, logits_false, labels):
    """Fit temperature T on a validation set.
    
    logits_true, logits_false: arrays of class-level logits (pre-softmax)
    labels: binary array (1 = true, 0 = false)
    """
    def nll(T):
        scaled_true = logits_true / T
        scaled_false = logits_false / T
        log_p_true = scaled_true - np.logaddexp(scaled_true, scaled_false)
        log_p_false = scaled_false - np.logaddexp(scaled_true, scaled_false)
        # negative log-likelihood
        return -np.mean(labels * log_p_true + (1 - labels) * log_p_false)
    
    result = minimize_scalar(nll, bounds=(0.01, 20.0), method='bounded')
    return result.x


def apply_temperature(logits_true, logits_false, T):
    """Apply learned temperature and return calibrated probabilities."""
    scaled_true = logits_true / T
    scaled_false = logits_false / T
    p_true = np.exp(scaled_true - np.logaddexp(scaled_true, scaled_false))
    p_false = 1 - p_true
    return p_true, p_false


def fit_temperature_from_probs(p_true, p_false, labels, eps=1e-12):
    """Fit temperature T from pre-computed probabilities.
    
    Converts to logits first, then optimizes.
    """
    logits_true = np.log(np.clip(p_true, eps, 1 - eps))
    logits_false = np.log(np.clip(p_false, eps, 1 - eps))
    return fit_temperature(logits_true, logits_false, labels)


def apply_temperature_from_probs(p_true, p_false, T, eps=1e-12):
    """Apply learned temperature starting from probabilities."""
    logits_true = np.log(np.clip(p_true, eps, 1 - eps))
    logits_false = np.log(np.clip(p_false, eps, 1 - eps))
    return apply_temperature(logits_true, logits_false, T)