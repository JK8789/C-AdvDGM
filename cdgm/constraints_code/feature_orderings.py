import json
import random
from typing import List
from cdgm.constraints_code.classes import Variable

random.seed(0)


def set_random_ordering(ordering: List[Variable], to_be_frozen_ids:[]):
    random.shuffle(ordering) # in-place shuffling
    frozen_vars = []
    for var in ordering:
        if var.id in to_be_frozen_ids:
            ordering.remove(var)
            frozen_vars.append(var)
    return frozen_vars + ordering
    return ordering


def set_ordering(use_case, ordering: List[Variable], label_ordering_choice: str, model_type: str, data_partition='test'):
    json_filename = f'cdgm/feature_ordering/feature_orderings.json'
    with open(json_filename, "r") as f:
        data = json.load(f)

    if label_ordering_choice == 'random':
        if use_case == "heloc":
            ordering = set_random_ordering(ordering, [1,12])
        else:
            ordering = set_random_ordering(ordering, [])

    elif label_ordering_choice == 'causal':
        ordering = data["feature_orderings"][use_case]['general'][label_ordering_choice]['train']
        ordering = list(map(lambda x: Variable(x), ordering.split()))
    else:
        ordering = data["feature_orderings"][use_case][model_type][label_ordering_choice][data_partition]
        ordering = list(map(lambda x: Variable(x), ordering.split()))

    readable_ordering = [e.readable() for e in ordering]
    print(f'Using *{label_ordering_choice}* feature ordering:\n', readable_ordering, 'len:', len(readable_ordering))
    return ordering