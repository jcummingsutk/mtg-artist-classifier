import os

import yaml

if __name__ == "__main__":
    # Load up the model parameters file,
    model_config_file = os.path.join(
        "mtg_artist_classifier", "classifier", "model_params.yaml"
    )
    if not os.path.exists(model_config_file):
        model_params_dict = {}
    else:
        with open(model_config_file, "r") as f:
            model_params_dict = yaml.safe_load(f)
    print(model_params_dict)

    # Load up the parameters file for dvc which contains the number of artists
    dvc_params_file = os.path.join("dvc_params.yaml")
    with open(dvc_params_file, "r") as dvc_f:
        dvc_params_dict = yaml.safe_load(dvc_f)

    # Write the number of artists in the dvc file to the model parameters file
    num_artists = len(dvc_params_dict["artists"])
    model_params_dict["num_artists"] = num_artists
    print(model_params_dict)
    with open(model_config_file, "w") as f:
        yaml.safe_dump(model_params_dict, f)
