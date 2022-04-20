# Todo find a way yo generate group for GRs
# use PCC function to generate group
import numpy as np


def generate_group(user_ids):
    group_scale = 10
    group_numbers = int(len(user_ids) / 10)
    print(len(user_ids))
    groups = []
    for gn in range(group_numbers):
        one_group = []
        for mn in range(np.random.randint(1, group_scale)):
            one_group.append(np.random.randint(1, len(user_ids)))
        groups.append(one_group)
    return groups
