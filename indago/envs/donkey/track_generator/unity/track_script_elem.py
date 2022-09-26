from indago.envs.donkey.track_generator.unity.state import State


class TrackScriptElem:
    def __init__(self, state: State = State.Straight, si: float = 1.0, num: int = 1):
        self.state = state
        self.value = si
        self.num_to_set = num
