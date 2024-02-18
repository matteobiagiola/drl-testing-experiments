from indago.avf.ga.chromosome import Chromosome
from indago.utils import randomness


def single_point_fixed_crossover(
    c1: Chromosome, c2: Chromosome, trials: int = 20
) -> bool:
    assert (
        c1.length == c2.length
    ), "The length of the two chromosome must be the same: {} != {}".format(
        c1.length, c2.length
    )

    if c1.length < 2:
        return False

    for _ in range(trials):
        point = randomness.get_random_int(low=0, high=c1.length)
        mixed_chromosome_1 = c1.crossover(c=c2, pos1=point, pos2=point)
        mixed_chromosome_2 = c2.crossover(c=c1, pos1=point, pos2=point)
        if mixed_chromosome_1 is not None and mixed_chromosome_2 is not None:
            c1.set_env_config(env_config=mixed_chromosome_1.env_config)
            c2.set_env_config(env_config=mixed_chromosome_2.env_config)
            return True

    return False
