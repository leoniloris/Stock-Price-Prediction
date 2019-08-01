from abc import ABC, abstractmethod
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import torch


def reconstruct_close_prices_from_log_returns(previous_close, log_returns, normalization_factor):
    close_prices = []
    def invert_log_returns(previous_close, current_log_return):
        close_prices.append(previous_close)
        return np.exp(current_log_return * normalization_factor) * previous_close

    reduce(invert_log_returns, log_returns, previous_close)
    return close_prices


def flatten_batch_of_samples(samples):
    samples = [samples] if not isinstance(samples, list) else samples
    flattened_samples =\
        map(lambda sample:\
            np.concatenate([sample.windowed_features.ravel(),
                            sample.scalar_features]).astype('float32'), samples)
    return torch.FloatTensor(list(flattened_samples))


def plot_metrics_factory():
    figure_handle = 0
    def plot_metrics(metrics, y_range=[0, 20], title=''):
        nonlocal figure_handle
        plt.figure(figsize=(13, 6))
        plt.plot(metrics['train_loss'], '-*', label='train loss', )
        plt.plot(metrics['validation_loss'], '-*', label='validation loss')
        plt.legend()
        plt.ylim(y_range)
        plt.title(('Figure %d\nMSE regression Loss. ' % figure_handle) + title)
        plt.xlabel('epoch')
        plt.grid()
        plt.show()
        figure_handle += 1
    return plot_metrics
