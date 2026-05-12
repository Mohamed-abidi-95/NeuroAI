"""
losses.py
Focal Loss et Weighted Cross-Entropy pour la gestion du déséquilibre des classes.
"""

import tensorflow as tf
import numpy as np


class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss : FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Pénalise davantage les erreurs sur les classes difficiles / minoritaires.

    Args:
        gamma (float): Facteur de focalisation. gamma=0 → Cross-Entropy classique.
        alpha (float|list): Pondération par classe. None = uniforme.
    """

    def __init__(self, gamma=2.0, alpha=None, name="focal_loss"):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # Stabilisation numérique
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Cross-entropy de base
        ce = -y_true * tf.math.log(y_pred)

        # Terme focal
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)

        # Pondération alpha
        if self.alpha is not None:
            alpha_t = tf.constant(self.alpha, dtype=tf.float32)
            alpha_t = tf.reduce_sum(y_true * alpha_t, axis=-1, keepdims=True)
            ce = alpha_t * ce

        loss = focal_weight * ce
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    def get_config(self):
        return {"gamma": self.gamma, "alpha": self.alpha}


def weighted_categorical_crossentropy(class_weights: dict):
    """
    Retourne une fonction de perte Weighted Cross-Entropy.
    class_weights : dict {0: w0, 1: w1, ..., 4: w4}
    """
    weights = np.array([class_weights[i] for i in sorted(class_weights.keys())],
                       dtype=np.float32)

    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        # Poids par sample selon la vraie classe
        sample_weights = tf.reduce_sum(
            y_true * tf.constant(weights), axis=-1
        )
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        return tf.reduce_mean(sample_weights * ce)

    loss_fn.__name__ = "weighted_categorical_crossentropy"
    return loss_fn


def compute_focal_alpha(class_counts: dict, num_classes=None):
    """
    Calcule les alpha proportionnellement à 1/freq pour chaque classe.
    class_counts : dict {0: n0, 1: n1, ..., (num_classes-1): n_(num_classes-1)}
    num_classes : None pour détection automatique, ou int
    """
    if num_classes is None:
        num_classes = len(class_counts)
    
    total = sum(class_counts.values())
    freqs = np.array([class_counts.get(i, 1) / total for i in range(num_classes)])
    alpha = (1.0 / freqs) / np.sum(1.0 / freqs)
    return alpha.tolist()


if __name__ == "__main__":
    # Test rapide
    import numpy as np
    y_true = tf.one_hot([0, 1, 2, 3, 4], depth=5)
    y_pred = tf.constant([[0.8, 0.05, 0.05, 0.05, 0.05],
                          [0.1, 0.6, 0.1, 0.1, 0.1],
                          [0.1, 0.1, 0.6, 0.1, 0.1],
                          [0.1, 0.1, 0.1, 0.6, 0.1],
                          [0.1, 0.1, 0.1, 0.1, 0.6]])
    fl = FocalLoss(gamma=2.0)
    print(f"Focal Loss = {fl(y_true, y_pred).numpy():.4f}")

