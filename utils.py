from abc import ABC, abstractmethod
from functools import reduce
import numpy as np
import torch


def reconstruct_close_prices_from_log_returns(previous_close, log_returns, normalization_factor):
    close_prices = []
    def invert_log_returns(previous_close, current_log_return):
        close_prices.append(previous_close)
        return np.exp(current_log_return * normalization_factor) * previous_close

    reduce(invert_log_returns, log_returns, previous_close)
    return close_prices


def backtest(model, samples):
    predictions = list(map(
        lambda sample: model.predict(sample),
        samples
    ))


def flatten_batch_of_samples(samples):
    samples = [samples] if not isinstance(samples, list) else samples
    flattened_samples =\
        map(lambda sample:\
            np.concatenate([sample.windowed_features.ravel(),
                            sample.scalar_features]).astype('float32'), samples)
    return torch.FloatTensor(list(flattened_samples))
