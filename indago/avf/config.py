AVF_DNN_POLICIES = ["mlp"]

CLASSIFIER_LAYERS = [1, 2, 3, 4]
AVF_TRAIN_POLICIES = [*AVF_DNN_POLICIES]
AVF_TEST_POLICIES_WITH_DNN = [
    "hc",
    "hc_failure",
    "hc_saliency_rnd",
    "hc_saliency_failure",
    "ga_rnd",
    "ga_failure",
    "ga_saliency_rnd",
    "ga_saliency_failure",
    "nn",
]
AVF_TEST_POLICIES = ["prioritized_replay", "test", "random", "replay_test_failure", *AVF_TEST_POLICIES_WITH_DNN]

DNN_SAMPLING_POLICIES = ["original"]
# NN parameters
SAMPLING_SIZE = 1000

# GA parameters
CROSSOVER_RATE = 0.75
MUTATION_RATE = 0.25
POPULATION_SIZE = 50
ELITISM_PERCENTAGE = 5
NUM_GENERATIONS = 100
NUM_TRIALS_CROSSOVER_MUTATION = 20

FILTER_FAILURE_BASED_APPROACHES = 30
