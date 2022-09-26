from typing import List

from indago.avf.env_configuration import EnvConfiguration


class TestingLogs:
    def __init__(self, config: EnvConfiguration, dynamic_info: List):
        self.config = config
        self.dynamic_info = dynamic_info

    def to_dict(self):
        return {"env_config": self.config.get_str(), "dynamic_info": self.dynamic_info}
