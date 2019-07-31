from abc import ABC, abstractmethod
from collections import namedtuple
from functools import reduce

import numpy as np

Sample = namedtuple('Sample', ['windowed_features', 'scalar_features', 'target'])


class Model(ABC):
    @abstractmethod
    def predict(self, sample: Sample):
        return 0


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


