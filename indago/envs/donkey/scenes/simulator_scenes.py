SIMULATOR_SCENES = ["generated_track"]


class SimulatorScene:
    def __init__(self, scene_name: str):
        self.scene_name = scene_name

    def get_scene_name(self) -> str:
        return self.scene_name


class GeneratedTrack(SimulatorScene):
    def __init__(self):
        super(GeneratedTrack, self).__init__(scene_name=SIMULATOR_SCENES[0])


SIMULATOR_SCENES_DICT = {"generated_track": GeneratedTrack()}
