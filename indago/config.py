PARK_ENV_ID = "parking-v0"
HUMANOID_ENV_ID = "Humanoid-v0"
DONKEY_ENV_ID = "DonkeyVAE-v0"
CARTPOLE_ENV_ID = "CartPole-v1"
ENV_IDS = [PARK_ENV_ID, HUMANOID_ENV_ID, DONKEY_ENV_ID, CARTPOLE_ENV_ID]

PARK_ENV_NAME = "park"
HUMANOID_ENV_NAME = "humanoid"
DONKEY_ENV_NAME = "donkey"
CARTPOLE_ENV_NAME = "cartpole"
ENV_NAMES = [PARK_ENV_NAME, HUMANOID_ENV_NAME, DONKEY_ENV_NAME, CARTPOLE_ENV_NAME]

# Other environments
PARAM_SEPARATOR = "#"
# DonkeyCar
COMMAND_SEPARATOR = "@"

# Humanoid coefficient
# C = 0.03
C = 0.01  # original value

# DonkeyCar params

MAX_SPEED = 15
SPEED_LIMIT = 10
MIN_SPEED = 5

COMMAND_NAME_VALUE_DICT = {"S": 0, "DY": 1, "R": 2, "L": 3}

VALUE_COMMAND_NAME_DICT = {0: "S", 1: "DY", 2: "R", 3: "L"}

# Raw camera input
CAMERA_HEIGHT = 120
CAMERA_WIDTH = 160
CAMERA_RESOLUTION = (CAMERA_WIDTH, CAMERA_HEIGHT)
MARGIN_TOP = CAMERA_HEIGHT // 3
# Region Of Interest
# r = [margin_left, margin_top, width, height]
ROI = [0, MARGIN_TOP, CAMERA_WIDTH, CAMERA_HEIGHT - MARGIN_TOP]

# Input dimension for VAE
IMAGE_WIDTH = ROI[2]
IMAGE_HEIGHT = ROI[3]
N_CHANNELS = 3
INPUT_DIM = (IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS)

# very smooth control: 10% -> 0.2 diff in steering allowed (requires more training)
# smooth control: 15% -> 0.3 diff in steering allowed
MAX_STEERING_DIFF = 0.2
# Negative reward for getting off the road
REWARD_CRASH = -10
# Penalize the agent even more when being fast
CRASH_SPEED_WEIGHT = 5

# Symmetric command
MAX_STEERING = 1
MIN_STEERING = -MAX_STEERING

# Simulation config
MIN_THROTTLE = 0.4
# max_throttle: 0.6 for level 0 and 0.5 for level 1
MAX_THROTTLE = 0.5
# Number of past commands to concatenate with the input
N_COMMAND_HISTORY = 5
# Max cross track error (used in normal mode to reset the car)
MAX_CTE_ERROR = 10.0

BASE_PORT = 9091
BASE_SOCKET_LOCAL_ADDRESS = 52804

# DonkeyCar track generator params

# exp_id 64
MIN_VALUE_COMMANDS = 3
MAX_DY_VALUE = 40

# exp_id 63 and below
# MIN_VALUE_COMMANDS = 1
# MAX_DY_VALUE = 50

NUM_COMMANDS_TRACK = 12
TRACK_WIDTH = 5
MAX_VALUE_COMMANDS = 20


FRAME_SKIP = 1

# Params that are logged
SIM_PARAMS = [
    "MIN_THROTTLE",
    "MAX_THROTTLE",
    "MAX_CTE_ERROR",
    "N_COMMAND_HISTORY",
    "MAX_STEERING_DIFF",
    "NUM_COMMANDS_TRACK",
    "MAX_DY_VALUE",
]
