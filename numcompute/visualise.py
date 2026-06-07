"""
visualise.py — Lightweight plotting utilities for NumCompute streaming logs.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def plot_metric_over_time(metric_values, title="Metric over Time", ylabel="Metric", save_path=None, show=True):
    values = np.asarray(metric_values, dtype=float)

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(values) + 1), values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def compare_models(metric1, metric2, labels=("Model 1", "Model 2"), title="Model Comparison", ylabel="Metric", save_path=None, show=True):
    m1 = np.asarray(metric1, dtype=float)
    m2 = np.asarray(metric2, dtype=float)

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(m1) + 1), m1, marker="o", label=labels[0])
    ax.plot(np.arange(1, len(m2) + 1), m2, marker="s", label=labels[1])
    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_predictions_vs_ground_truth(y_true, y_pred, title="Predictions vs Ground Truth", save_path=None, show=True):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    x = np.arange(len(y_true))

    fig, ax = plt.subplots()
    ax.scatter(x, y_true, marker="o", label="Ground Truth")
    ax.scatter(x, y_pred, marker="x", label="Prediction")
    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Class")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax