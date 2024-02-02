import glob
import os
import random
import shutil
from dataclasses import dataclass

import numpy as np
import yaml


@dataclass
class TrainValIndices:
    first_train_example_idx: int
    last_train_example_idx: int
    first_val_example_idx: int
    last_val_example_idx: int


@dataclass
class ArtistImageInfo:
    artist_name: str
    image_directory: str


def get_shuffled_multiverse_ids(files: list[str]) -> list[int]:
    """Gets a list of files of the form some_dir/some_other_dir/number.jpg,
    shuffles the list predictably, then returns the list of the numbers, which are the
    multiverse ids of the cards

    Args:
        files (list[str]): _description_

    Returns:
        list[int]: _description_
    """
    basename_files = [os.path.basename(f) for f in files]
    multiverse_ids = sorted([int(file_[:-4]) for file_ in basename_files])
    seed = 42
    random.Random(seed).shuffle(multiverse_ids)
    return multiverse_ids


def get_train_eval_test_indices(multiverse_ids: list[int]) -> TrainValIndices:
    """Takes a list of multiverse ids and creates the indices for the training and
    validation set. Example: if train_fraction is 0.8 and val_fraction is 0.2, and there
    are 20 items, then first
    first_train_example_idx = 0
    last_train_example_idx = 16 later will use training_list = total_list[0:16]
    first_val_example_idx = 16
    last_val_example_idx = 20 later will use eval_list = total_list[16:20]

    Args:
        multiverse_ids (list[int]): _description_

    Returns:
        TrainValIndices: _description_
    """
    if len(multiverse_ids) < 10:
        raise ValueError("Not enough indices to do a split")

    with open("dvc_params.yaml", "r") as dvc_config_file:
        dvc_params = yaml.safe_load(dvc_config_file)
    train_fraction = dvc_params["train_fraction"]
    val_fraction = dvc_params["val_fraction"]
    if not np.isclose(0, 1.0 - train_fraction - val_fraction):
        raise ValueError("train eval test should add to one")

    num_examples = len(multiverse_ids)
    first_train_example_idx = 0
    last_train_example_idx = int(train_fraction * num_examples)
    first_val_example_idx = last_train_example_idx
    last_val_example_idx = int((train_fraction + val_fraction) * num_examples)
    train_eval_test_indices = TrainValIndices(
        first_train_example_idx=first_train_example_idx,
        last_train_example_idx=last_train_example_idx,
        first_val_example_idx=first_val_example_idx,
        last_val_example_idx=last_val_example_idx,
    )
    return train_eval_test_indices


def create_train_val_split(artist_image_info: ArtistImageInfo, model_data_dir: str):
    """For a given artist, gets all the jpgs of their art in the local directory,
    creates training and validation directories in the model_data_dir, then copies
    the appropriate fraction of images into the training and validation directory

    Args:
        artist_image_info (ArtistImageInfo): Artist name and where their jpgs are
        model_data_dir (str): Where the artist's training and validation directories
        will be held
    """
    # Get the list of jpgs, shuffle, decide which will be training and validation
    files = glob.glob(os.path.join(artist_image_info.image_directory, "*.jpg"))
    shuffled_multiverse_ids = get_shuffled_multiverse_ids(files)
    idxs = get_train_eval_test_indices(shuffled_multiverse_ids)
    train_multiverse_ids = shuffled_multiverse_ids[
        idxs.first_train_example_idx : idxs.last_train_example_idx
    ]
    val_multiverse_ids = shuffled_multiverse_ids[
        idxs.first_val_example_idx : idxs.last_val_example_idx
    ]

    # Create filenames that need to be copied (source) and where they will be copied to
    # (destination) for both training and validation sets
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

    source_val_filenames = [
        os.path.join(artist_image_info.image_directory, str(val_multiverse_id) + ".jpg")
        for val_multiverse_id in val_multiverse_ids
    ]
    destination_val_filenames = [
        os.path.join(
            model_data_dir,
            "val_images",
            artist_image_info.artist_name,
            str(val_multiverse_id) + ".jpg",
        )
        for val_multiverse_id in val_multiverse_ids
    ]

    # Copy the files
    os.makedirs(
        os.path.join(model_data_dir, "train_images", artist_image_info.artist_name),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(model_data_dir, "val_images", artist_image_info.artist_name),
        exist_ok=True,
    )

    for source, dest in zip(source_train_filenames, destination_train_filenames):
        shutil.copy(source, dest)
    for source, dest in zip(source_val_filenames, destination_val_filenames):
        shutil.copy(source, dest)


if __name__ == "__main__":
    # For each artist that is contained in the card_images file,
    # this script creates training and validation images in the
    # data/model_training_and_eval folder
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
        create_train_val_split(
            artist_image_info=image_dir, model_data_dir=root_model_data_dir
        )
