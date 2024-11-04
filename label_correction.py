import os

import pandas as pd

from configs import DATA_DIR
from main import main
from utils import set_seed


def correct_label_errors(data: pd.DataFrame):
    # TODO: Do Something
    corrected_data = data
    return corrected_data


if __name__ == "__main__":
    set_seed()
    data = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    corrected_data = correct_label_errors(data)
    main(corrected_data)
