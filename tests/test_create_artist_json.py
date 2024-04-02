import json
import os

from mtg_artist_classifier.classifier.config_utils import create_artist_json


class MockImageFolder:
    def __init__(self, class_to_idx_dict: dict[str, int]):
        self.class_to_idx_dict = class_to_idx_dict

    @property
    def class_to_idx(self) -> dict[str, int]:
        return self.class_to_idx_dict


def test_correct_artist_json_creation():
    """During training a json file needs to be created with keys that are ints and
    values that are artist names"""
    mock_class_to_idx_dict = {"artist 1": 0, "artist 2": 1}
    mock_image_folder = MockImageFolder(mock_class_to_idx_dict)
    output_json_file = os.path.join("tests", "artist_mapping_file.json")
    create_artist_json(mock_image_folder, output_json_file)

    with open(output_json_file, "r") as fp:
        output_dict = json.load(fp)
    expected_output_dict = {"0": "artist 1", "1": "artist 2"}
    assert output_dict == expected_output_dict
