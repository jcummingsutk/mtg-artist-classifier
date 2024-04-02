import os

import yaml

from mtg_artist_classifier.classifier.config_utils import get_num_classes


def test_correct_path():
    """Make sure that model params file has correct key and location"""
    model_config_path = os.path.join(
        "mtg_artist_classifier", "classifier", "model_params.yaml"
    )
    with open(model_config_path, "r") as fp:
        params = yaml.safe_load(fp)
    num_artists = params["num_artists"]
    assert type(num_artists) is int


def test_config_num_artists():
    """Makes sure the get_num_classes correctly gets the number of artists with a static model config test"""
    static_test_model_config_path = os.path.join("tests", "model_config_test.yaml")
    num_artists = get_num_classes(static_test_model_config_path)
    assert num_artists == 2
