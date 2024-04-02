import json
from typing import Protocol

import yaml


class ImageFolder(Protocol):
    @property
    def class_to_idx() -> dict[str, int]:
        """"""


def create_artist_json(dataset: ImageFolder, artist_mapping_file: str):
    artist_mapping = {val: key for key, val in dataset.class_to_idx.items()}
    with open(artist_mapping_file, "w") as fp:
        json.dump(artist_mapping, fp)


def get_num_classes(model_params_fp: str) -> int:
    """When the data pipeline runs and gets all of the card images, the last
    step outputs the number of artists to classify to model_params. This function
    retrieves that for the purposes of building the classifier

    Returns:
        int: The number of artists to classify
    """
    with open(model_params_fp, "r") as fp:
        params_dict = yaml.safe_load(fp)
    num_classes = params_dict["num_artists"]

    return int(num_classes)
