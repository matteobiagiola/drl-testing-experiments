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
# Original author: Felix Yu
import glob
import os
import platform
import subprocess
import time

from log import Log


class DonkeyUnityProcess(object):
    """
    Utility class to start unity process if needed.
    """

    def __init__(self):
        self.process = None
        self.logger = Log("DonkeyUnityProcess")

    def start(self, sim_path: str, headless: bool, port: int, simulation_mul: int = 1):
        """
        :param sim_path: (str) Path to the executable
        :param headless: (bool)
        :param port: (int)
        :param simulation_mul: (int)
        """
        if not os.path.exists(sim_path):
            self.logger.info("{} does not exist".format(sim_path))
            return

        cwd = os.getcwd()
        file_name = (
            sim_path.strip()
            .replace(".app", "")
            .replace(".exe", "")
            .replace(".x86_64", "")
            .replace(".x86", "")
        )
        true_filename = os.path.basename(os.path.normpath(file_name))
        launch_string = None
        self.logger.info("Simulation multiplier: {}".format(simulation_mul))
        port_args = [
            "--port",
            str(port),
            "-logFile",
            "unitylog.txt",
            "--simulation-mul",
            str(simulation_mul),
        ]
        platform_ = platform.system()

        if platform_.lower() == "linux" and sim_path:
            candidates = glob.glob(os.path.join(cwd, file_name) + ".x86_64")
            if len(candidates) == 0:
                candidates = glob.glob(os.path.join(cwd, file_name) + ".x86")
            if len(candidates) == 0:
                candidates = glob.glob(file_name + ".x86_64")
            if len(candidates) == 0:
                candidates = glob.glob(file_name + ".x86")
            if len(candidates) > 0:
                launch_string = candidates[0]

        elif platform_.lower() == "darwin" and sim_path:
            candidates = glob.glob(
                os.path.join(
                    cwd, file_name + ".app", "Contents", "MacOS", true_filename
                )
            )
            if len(candidates) == 0:
                candidates = glob.glob(
                    os.path.join(file_name + ".app", "Contents", "MacOS", true_filename)
                )
            if len(candidates) == 0:
                candidates = glob.glob(
                    os.path.join(cwd, file_name + ".app", "Contents", "MacOS", "*")
                )
            if len(candidates) == 0:
                candidates = glob.glob(
                    os.path.join(file_name + ".app", "Contents", "MacOS", "*")
                )
            if len(candidates) > 0:
                launch_string = candidates[0]

        elif platform_.lower() == "windows" and sim_path:
            candidates = glob.glob(os.path.join(cwd, file_name) + ".exe")
            if len(candidates) > 0:
                launch_string = candidates[0]

        if launch_string is None:
            self.logger.debug("Launch string is Null")
        else:
            self.logger.debug("This is the launch string {}".format(launch_string))

            # Launch Unity environment
            if headless:
                self.process = subprocess.Popen(
                    [launch_string, "-batchmode"] + port_args
                )
            else:
                self.process = subprocess.Popen([launch_string] + port_args)

            if sim_path:
                # hack to wait for the simulator to start
                time.sleep(20)

        self.logger.info("Donkey subprocess started")

    def quit(self):
        """
        Shutdown unity environment
        """
        if self.process is not None:
            self.logger.info("Closing donkey sim subprocess")
            self.process.kill()
            self.process = None
