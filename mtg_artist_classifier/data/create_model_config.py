import os

import yaml

if __name__ == "__main__":
    model_config_file = os.path.join(
        "mtg_artist_classifier", "classifier", "model_params.yaml"
    )
    if not os.path.exists(model_config_file):
        model_params_dict = None
    else:
        with open(model_config_file, "r") as f:
            model_params_dict = yaml.safe_load(f)
    if model_params_dict is None:
        model_params_dict = {}
    print(model_params_dict)
    dvc_params_file = os.path.join("dvc_params.yaml")
    with open(dvc_params_file, "r") as dvc_f:
        dvc_params_dict = yaml.safe_load(dvc_f)
    num_artists = len(dvc_params_dict["artists"])
    model_params_dict["num_artists"] = num_artists
    print(model_params_dict)
    with open(model_config_file, "w") as f:
        yaml.safe_dump(model_params_dict, f)
