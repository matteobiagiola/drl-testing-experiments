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

import base64
import io
import json
import os
import pathlib
import warnings
import zipfile
from typing import Any, Dict, Optional, Tuple, Union

import torch as th
from cloudpickle import cloudpickle
from stable_baselines3.common.save_util import open_path
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device

from indago.utils.rename_unpickler import renamed_loads


def load_from_zip_file(
    load_path: Union[str, pathlib.Path, io.BufferedIOBase],
    load_data: bool = True,
    custom_objects: Optional[Dict[str, Any]] = None,
    device: Union[th.device, str] = "auto",
    verbose: int = 0,
    algo_name: str = None,
) -> (Tuple[Optional[Dict[str, Any]], Optional[TensorDict], Optional[TensorDict]]):
    """
    Load model data from a .zip archive

    :param load_path: Where to load the model from
    :param load_data: Whether we should load and return data
        (class parameters). Mainly used by 'load_parameters' to only load model parameters (weights)
    :param custom_objects: Dictionary of objects to replace
        upon loading. If a variable is present in this dictionary as a
        key, it will not be deserialized and the corresponding item
        will be used instead. Similar to custom_objects in
        ``keras.models.load_model``. Useful when you have an object in
        file that can not be deserialized.
    :param device: Device on which the code should run.
    :return: Class parameters, model state_dicts (aka "params", dict of state_dict)
        and dict of pytorch variables
    """
    load_path = open_path(load_path, "r", verbose=verbose, suffix="zip")

    # set device to cpu if cuda is not available
    device = get_device(device=device)

    # Open the zip archive and load data
    try:
        with zipfile.ZipFile(load_path) as archive:
            namelist = archive.namelist()
            # If data or parameters is not in the
            # zip archive, assume they were stored
            # as None (_save_to_file_zip allows this).
            data = None
            pytorch_variables = None
            params = {}

            if "data" in namelist and load_data:
                # Load class parameters that are stored
                # with either JSON or pickle (not PyTorch variables).
                json_data = archive.read("data").decode()
                data = json_to_data(json_data, custom_objects=custom_objects, algo_name=algo_name)

            # Check for all .pth files and load them using th.load.
            # "pytorch_variables.pth" stores PyTorch variables, and any other .pth
            # files store state_dicts of variables with custom names (e.g. policy, policy.optimizer)
            pth_files = [file_name for file_name in namelist if os.path.splitext(file_name)[1] == ".pth"]
            for file_path in pth_files:
                with archive.open(file_path, mode="r") as param_file:
                    # File has to be seekable, but param_file is not, so load in BytesIO first
                    # fixed in python >= 3.7
                    file_content = io.BytesIO()
                    file_content.write(param_file.read())
                    # go to start of file
                    file_content.seek(0)
                    # Load the parameters with the right ``map_location``.
                    # Remove ".pth" ending with splitext
                    th_object = th.load(file_content, map_location=device)
                    # "tensors.pth" was renamed "pytorch_variables.pth" in v0.9.0, see PR #138
                    if file_path == "pytorch_variables.pth" or file_path == "tensors.pth":
                        # PyTorch variables (not state_dicts)
                        pytorch_variables = th_object
                    else:
                        # State dicts. Store into params dictionary
                        # with same name as in .zip file (without .pth)
                        params[os.path.splitext(file_path)[0]] = th_object
    except zipfile.BadZipFile:
        # load_path wasn't a zip file
        raise ValueError(f"Error: the file {load_path} wasn't a zip-file")
    return data, params, pytorch_variables


def json_to_data(json_string: str, custom_objects: Optional[Dict[str, Any]] = None, algo_name: str = None) -> Dict[str, Any]:
    """
    Turn JSON serialization of class-parameters back into dictionary.

    :param json_string: JSON serialization of the class-parameters
        that should be loaded.
    :param custom_objects: Dictionary of objects to replace
        upon loading. If a variable is present in this dictionary as a
        key, it will not be deserialized and the corresponding item
        will be used instead. Similar to custom_objects in
        ``keras.models.load_model``. Useful when you have an object in
        file that can not be deserialized.
    :return: Loaded class parameters.
    """
    if custom_objects is not None and not isinstance(custom_objects, dict):
        raise ValueError("custom_objects argument must be a dict or None")

    json_dict = json.loads(json_string)
    # This will be filled with deserialized data
    return_data = {}
    for data_key, data_item in json_dict.items():
        # print(data_key, data_item)
        if custom_objects is not None and data_key in custom_objects.keys():
            # If item is provided in custom_objects, replace
            # the one from JSON with the one in custom_objects
            return_data[data_key] = custom_objects[data_key]
        elif isinstance(data_item, dict) and ":serialized:" in data_item.keys():
            # If item is dictionary with ":serialized:"
            # key, this means it is serialized with cloudpickle.
            serialization = data_item[":serialized:"]
            # Try-except deserialization in case we run into
            # errors. If so, we can tell bit more information to
            # user.
            try:
                base64_object = base64.b64decode(serialization.encode())
                deserialized_object = cloudpickle.loads(base64_object)
            except (RuntimeError, TypeError, ModuleNotFoundError) as ex:
                if isinstance(ex, RuntimeError) or isinstance(ex, TypeError):
                    warnings.warn(
                        f"Could not deserialize object {data_key}. "
                        + "Consider using `custom_objects` argument to replace "
                        + "this object."
                    )
                elif isinstance(ex, ModuleNotFoundError):
                    # FIXME: needed because some agents were trained with an old version of the project.
                    #  After refactoring model loading does not work anymore because the models have the
                    #  the structure of the packages saved in them. For this reason, after refactoring a
                    #  module renaming is needed.
                    assert algo_name is not None, "Algo name must be instantiated"
                    assert (
                        algo_name == "her" or algo_name == "sac" or algo_name == "tqc"
                    ), "Not supported algo name: {}".format(algo_name)
                    if algo_name == "her" or algo_name == "tqc":
                        deserialized_object = renamed_loads(
                            pickled_bytes=base64_object,
                            old_module="park.algos.tqc_wrapper",
                            renamed_module="indago.algos.tqc_wrapper",
                        )
                    elif algo_name == "sac":
                        deserialized_object = renamed_loads(
                            pickled_bytes=base64_object,
                            old_module="pt.algos.custom_sac",
                            renamed_module="indago.algos.sac_wrapper",
                        )
            return_data[data_key] = deserialized_object
        else:
            # Read as it is
            return_data[data_key] = data_item
    return return_data
