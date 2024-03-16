import os
import pickle
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import logging as log
import numpy as np
import random


def shuffle_np(x, y):
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    return x, y


def shuffle_and_undersample(x, y):
    x, y = shuffle_np(x, y)

    min_len = np.minimum(np.sum(y == 1), np.sum(y == 0))

    x_zeros = x[(y == 0).reshape(-1)][:min_len]
    x_ones = x[(y == 1).reshape(-1)][:min_len]

    x = np.concatenate([x_zeros, x_ones])
    y = np.concatenate([np.zeros(min_len), np.ones(min_len)])
    x, y = shuffle_np(x, y)

    return x, y


def load_pickle_file(data_path: str, file_name: str = "dump.pickle"):
    with open(os.path.join(data_path, file_name), "rb") as file:
        df = pickle.load(file)
    return df


def write_pickle_file(df, data_path: str, file_name: str = "dump.pickle") -> None:
    os.makedirs(data_path, exist_ok=True)
    with open(os.path.join(data_path, file_name), "wb") as file:
        pickle.dump(df, file)
    log.info(f"Saved data to {data_path}/{file_name}")


def get_val_test_ids():
    return {
        "test_ids": (
            (3, 32),
            (3, 18),
            (1, 27),
            (3, 19),
            (3, 17),
            (2, 21),
            (1, 20),
            (1, 11),
        ),
        "val_ids": (
            (3, 3),
            (2, 10),
            (1, 24),
            (3, 24),
            (1, 32),
            (2, 1),
            (1, 10),
            (1, 16),
        ),
    }


def get_full_set_ids():
    return {
        "test_ids": ((1, 11),),
        "val_ids": ((1, 10),),
    }


def get_vs_val_test_ids() -> dict:
    return {
        "test_ids": ((1, 16), (1, 20), (1, 21), (2, 16), (2, 20), (2, 21)),
        "val_ids": (
            (1, 11),
            (1, 14),
            (1, 15),
            (1, 19),
            (2, 11),
            (2, 14),
            (2, 15),
            (2, 19),
        )
        + get_experiment_ids(3),
    }


def get_inv_vs_val_test_ids() -> dict:
    return {
        "test_ids": (
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 12),
            (1, 17),
            (2, 7),
            (2, 8),
            (2, 9),
            (2, 12),
            (2, 17),
        ),
        "val_ids": (
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 22),
            (1, 26),
            (1, 30),
            (1, 4),
            (1, 10),
            (1, 23),
            (1, 27),
            (1, 31),
            (1, 5),
            (1, 24),
            (1, 28),
            (1, 32),
            (1, 6),
            (1, 13),
            (1, 25),
            (1, 29),
            (1, 33),
            (1, 18),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 10),
            (2, 5),
            (2, 6),
            (2, 13),
            (2, 18),
        )
        + get_experiment_ids(3),
    }


def get_vd_val_test_ids() -> dict:
    return {
        "test_ids": (
            (1, 17),
            (1, 18),
            (1, 19),
            (1, 20),
            (1, 21),
            (2, 18),
            (2, 19),
            (2, 20),
            (2, 21),
            (2, 22),
        ),
        "val_ids": (
            (1, 5),
            (1, 24),
            (1, 28),
            (1, 32),
            (1, 12),
            (1, 6),
            (1, 13),
            (1, 25),
            (1, 29),
            (1, 33),
            (1, 14),
            (1, 15),
            (1, 16),
            (2, 5),
            (2, 12),
            (2, 6),
            (2, 13),
            (2, 14),
            (2, 15),
            (2, 16),
        )
        + get_experiment_ids(3),
    }


def get_inv_vd_val_test_ids() -> dict:
    return {
        "test_ids": (
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 22),
            (1, 26),
            (1, 30),
            (2, 1),
            (2, 2),
            (2, 3),
        ),
        "val_ids": (
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 4),
            (1, 10),
            (1, 23),
            (1, 27),
            (1, 31),
            (1, 11),
            (1, 5),
            (1, 24),
            (1, 28),
            (1, 32),
            (2, 7),
            (2, 8),
            (2, 9),
            (2, 4),
            (2, 10),
            (2, 11),
            (2, 5),
        )
        + get_experiment_ids(3),
    }


def get_experiment_ids(experiment: int) -> tuple:
    if experiment == 1:
        runs = [
            2,
            3,
            4,
            5,
            7,
            8,
            9,
            10,
            11,
            13,
            14,
            15,
            16,
            20,
            21,
            22,
            23,
            24,
            26,
            27,
            28,
            30,
            31,
            32,
        ]
    elif experiment == 2:
        runs = [1, 2, 3, 4, 5, 8, 9, 10, 11, 15, 16, 20, 21]
    elif experiment == 3:
        runs = list(range(1, 22)) + list(range(23, 33))
    else:
        raise ValueError("Select from experimets [1, 2, 3]")
    return tuple([(experiment, w) for w in runs])


def get_val_test_experiments(
    experiments: list[int], num_val: int = 5, num_test: int = 5
) -> dict:
    """
    Returns a dict with validation and test IDs randomly selected from the given experiments.
    """
    if len(experiments) not in [1, 2, 3]:
        raise ValueError("Pick 1 to 3 experiments")
    if not set(experiments).issubset([1, 2, 3]):
        raise ValueError("Select from experiments [1, 2,]")

    ids = ()
    for e in experiments:
        ids += get_experiment_ids(e)
    print(ids)
    val = random.sample(ids, num_val)
    for run in val:
        ids_list = list(ids)
        ids_list.remove(run)
        ids = tuple(ids_list)
    test = random.sample(ids, num_test)
    for run in test:
        ids_list = list(ids)
        ids_list.remove(run)
        ids = tuple(ids_list)
    return {"val_ids": tuple(val), "test_ids": tuple(test)}


def plot_single_CV(x, y):
    fig, ax1 = plt.subplots()
    ax1.plot(x[:, 0])
    ax_2 = ax1.twinx()
    ax_2.plot(x[:, 1], color="red")
    title = "good" if y == 1 else "bad"
    plt.title(title)
    fig.tight_layout()
    plt.show()


class MyScaler:

    def __init__(self) -> None:
        self.scaler = StandardScaler()

    def fit(self, x):
        s_0, s_1, s_2 = x.shape
        self.scaler.fit(x.reshape(-1, s_2))

    def transform(self, x):
        s_0, s_1, s_2 = x.shape
        x = self.scaler.transform(x.reshape(-1, s_2))
        return x.reshape(s_0, s_1, s_2)

    def inverse_transform(self, x):
        s_0, s_1, s_2 = x.shape
        x = self.scaler.inverse_transform(x.reshape(-1, s_2))
        return x.reshape(s_0, s_1, s_2)


def select_random_val_test_ids():
    mixed = [2, 16]
    good_exmples = [2, 3, 22, 24, 26, 27, 28]
    bad_examples = [16, 5, 7, 8, 9, 10, 11, 13, 14, 15, 20, 21, 23, 30, 31, 32]

    good_val_id, good_test_id = np.random.choice(good_exmples, 2, replace=False)
    bad_val_id, bad_test_id = np.random.choice(bad_examples, 2, replace=False)
    return good_val_id, bad_val_id, good_test_id, bad_test_id


def get_data_path():
    return "data"
