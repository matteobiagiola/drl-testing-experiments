"""
The MIT License

Copyright (c) 2019 Antonin Raffin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from stable_baselines3.common.results_plotter import X_EPISODES, X_TIMESTEPS, X_WALLTIME

X_SUCCESS = "success_rate"


def ts2xy(data_frame: pd.DataFrame, x_axis: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose a data frame variable to x ans ys

    :param data_frame: the input data
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: the x and y output
    """
    if x_axis == X_TIMESTEPS:
        x_var = np.arange(len(data_frame))
        y_var = data_frame.r.values
    elif x_axis == X_EPISODES:
        x_var = np.arange(len(data_frame))
        y_var = data_frame.l.values
    elif x_axis == X_WALLTIME:
        # Convert to hours
        x_var = data_frame.t.values / 3600.0
        y_var = data_frame.r.values
    elif x_axis == X_SUCCESS:
        x_var = np.arange(len(data_frame))
        y_var = data_frame.is_success.values
    else:
        raise NotImplementedError
    return x_var, y_var
