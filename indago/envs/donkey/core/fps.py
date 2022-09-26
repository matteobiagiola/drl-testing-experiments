"""
MIT License

Copyright (c) 2018 Roma Sokolkov
Copyright (c) 2018 Antonin Raffin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# Original author: Tawn Kramer
import time

from log import Log


class FPSTimer(object):
    """
    Helper function to monitor the speed of the control.
    :param verbose: (int)
    """

    def __init__(self, timer_name: str, verbose: int = 0):
        self.start_time = time.perf_counter()
        self.iter = 0
        self.verbose = verbose
        self.logger = Log("FPSTimer-{}".format(timer_name))

    def reset(self):
        self.start_time = time.perf_counter()
        self.iter = 0

    def on_frame(self):
        self.iter += 1
        if self.iter == 100:
            end_time = time.perf_counter()
            if self.verbose >= 1:
                self.logger.debug("{:.5f} fps".format(100.0 / (end_time - self.start_time)))
            self.start_time = time.perf_counter()
            self.iter = 0
