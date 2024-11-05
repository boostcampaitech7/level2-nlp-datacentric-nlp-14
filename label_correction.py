import os

import pandas as pd

from configs import DATA_DIR
from main import main
from noise_data_filter import noise_labeling
from utils import set_seed


def correct_label_errors(data: pd.DataFrame):
    # TODO: Do Something
    corrected_data = data.copy()
    return corrected_data


if __name__ == "__main__":
    set_seed()
    data = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

    labeled_data = noise_labeling(data)

    # TODO: restored_sentence 만드는 함수로 교체 필요
    restored = pd.read_csv(os.path.join(DATA_DIR, "restored_sentences.csv"))
    restored_data = pd.merge(labeled_data, restored[["ID", "restored"]], on="ID")
    # TODO End

    corrected_data = correct_label_errors(restored_data)
    corrected_data.to_csv(os.path.join(DATA_DIR, "label_corrected_train.csv"), index=False)
    main(corrected_data, do_predict=False)
