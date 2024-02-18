import io
import pickle
from typing import Any
from typing.io import IO


class RenameUnpickler(pickle.Unpickler):
    def __init__(self, file: IO[bytes], old_module: str, renamed_module: str):
        super(RenameUnpickler, self).__init__(file=file)
        self.old_module = old_module
        self.renamed_module = renamed_module

    def find_class(self, __module_name: str, __global_name: str) -> Any:
        renamed_module = __module_name
        if __module_name == self.old_module:
            renamed_module = self.renamed_module
        elif __global_name == "TQCPolicy":
            renamed_module = "indago.algos.tqc_policy"

        return super(RenameUnpickler, self).find_class(renamed_module, __global_name)


def renamed_load(file_obj, old_module: str, renamed_module: str):
    return RenameUnpickler(
        file=file_obj, old_module=old_module, renamed_module=renamed_module
    ).load()


def renamed_loads(pickled_bytes, old_module: str, renamed_module: str):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(
        file_obj=file_obj, old_module=old_module, renamed_module=renamed_module
    )
