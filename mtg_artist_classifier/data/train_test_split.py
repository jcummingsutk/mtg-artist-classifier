import glob
import os
import random
import shutil
from dataclasses import dataclass

import numpy as np
import yaml


@dataclass
class TrainEvalTestIndices:
    first_train_example_idx: int
    last_train_example_idx: int
    first_eval_example_idx: int
    last_eval_example_idx: int
    first_test_example_idx: int
    last_test_example_idx: int


@dataclass
class ArtistImageInfo:
    artist_name: str
    image_directory: str


def get_shuffled_multiverse_ids(files: list[str]) -> list[int]:
    basename_files = [os.path.basename(f) for f in files]
    multiverse_ids = sorted([int(file_[:-4]) for file_ in basename_files])
    seed = 42
    random.Random(seed).shuffle(multiverse_ids)
    return multiverse_ids


def get_train_eval_test_indices(multiverse_ids: list[int]) -> TrainEvalTestIndices:
    if len(multiverse_ids) < 10:
        raise ValueError("Not enough indices to do a split")

    with open("dvc_params.yaml", "r") as dvc_config_file:
        dvc_params = yaml.safe_load(dvc_config_file)
    train_fraction = dvc_params["train_fraction"]
    eval_fraction = dvc_params["eval_fraction"]
    test_fraction = dvc_params["test_fraction"]
    if not np.isclose(test_fraction, 1.0 - train_fraction - eval_fraction):
        raise ValueError("train eval test should add to one")

    num_examples = len(multiverse_ids)
    first_train_example_idx = 0
    last_train_example_idx = int(train_fraction * num_examples)
    first_eval_example_idx = last_train_example_idx + 1
    last_eval_example_idx = int((train_fraction + eval_fraction) * num_examples)
    first_test_example_idx = last_eval_example_idx + 1
    last_test_example_idx = num_examples
    train_eval_test_indices = TrainEvalTestIndices(
        first_train_example_idx=first_train_example_idx,
        last_train_example_idx=last_train_example_idx,
        first_eval_example_idx=first_eval_example_idx,
        last_eval_example_idx=last_eval_example_idx,
        first_test_example_idx=first_test_example_idx,
        last_test_example_idx=last_test_example_idx,
    )
    return train_eval_test_indices


def create_train_test_eval_split(
    artist_image_info: ArtistImageInfo, model_data_dir: str
):
    files = glob.glob(os.path.join(artist_image_info.image_directory, "*.jpg"))
    shuffled_multiverse_ids = get_shuffled_multiverse_ids(files)
    idxs = get_train_eval_test_indices(shuffled_multiverse_ids)
    train_multiverse_ids = shuffled_multiverse_ids[
        idxs.first_train_example_idx : idxs.last_train_example_idx
    ]
    eval_multiverse_ids = shuffled_multiverse_ids[
        idxs.first_eval_example_idx : idxs.last_eval_example_idx
    ]
    test_multiverse_ids = shuffled_multiverse_ids[
        idxs.first_test_example_idx : idxs.last_test_example_idx
    ]

    source_train_filenames = [
        os.path.join(
            artist_image_info.image_directory, str(train_multiverse_id) + ".jpg"
        )
        for train_multiverse_id in train_multiverse_ids
    ]
    destination_train_filenames = [
        os.path.join(
            model_data_dir,
            "train_images",
            artist_image_info.artist_name,
            str(train_multiverse_id) + ".jpg",
        )
        for train_multiverse_id in train_multiverse_ids
    ]

    source_eval_filenames = [
        os.path.join(
            artist_image_info.image_directory, str(eval_multiverse_id) + ".jpg"
        )
        for eval_multiverse_id in eval_multiverse_ids
    ]
    destination_eval_filenames = [
        os.path.join(
            model_data_dir,
            "eval_images",
            artist_image_info.artist_name,
            str(eval_multiverse_id) + ".jpg",
        )
        for eval_multiverse_id in eval_multiverse_ids
    ]

    source_test_filenames = [
        os.path.join(
            artist_image_info.image_directory, str(test_multiverse_id) + ".jpg"
        )
        for test_multiverse_id in test_multiverse_ids
    ]
    destination_test_filenames = [
        os.path.join(
            model_data_dir,
            "test_images",
            artist_image_info.artist_name,
            str(test_multiverse_id) + ".jpg",
        )
        for test_multiverse_id in test_multiverse_ids
    ]
    os.makedirs(
        os.path.join(model_data_dir, "train_images", artist_image_info.artist_name),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(model_data_dir, "eval_images", artist_image_info.artist_name),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(model_data_dir, "test_images", artist_image_info.artist_name),
        exist_ok=True,
    )
    for source, dest in zip(source_train_filenames, destination_train_filenames):
        shutil.copy(source, dest)
    for source, dest in zip(source_eval_filenames, destination_eval_filenames):
        shutil.copy(source, dest)
    for source, dest in zip(source_test_filenames, destination_test_filenames):
        shutil.copy(source, dest)


if __name__ == "__main__":
    root_image_dir = os.path.join("data", "card_images")
    root_model_data_dir = os.path.join("data", "model_training_and_eval")
    artist_dirs = os.listdir(root_image_dir)
    artist_image_info_list = [
        ArtistImageInfo(
            artist_name=os.path.basename(artist_dir),
            image_directory=os.path.join(root_image_dir, artist_dir),
        )
        for artist_dir in artist_dirs
    ]
    artist_names = [os.path.basename(dir_) for dir_ in artist_dirs]
    for image_dir in artist_image_info_list:
        print(artist_image_info_list)
        create_train_test_eval_split(
            artist_image_info=image_dir, model_data_dir=root_model_data_dir
        )
