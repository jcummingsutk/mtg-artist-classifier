import argparse

import mlflow
from main import configure_remote_tracking, main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-folder")
    parser.add_argument("--val-data-folder")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-epochs", type=int)

    # Optional azure info for when training locally
    parser.add_argument("--remote-tracking", type=bool, default=False)
    args = parser.parse_args()

    train_data_folder = args.train_data_folder
    val_data_folder = args.val_data_folder
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    remote_tracking = args.remote_tracking

    if remote_tracking:
        config_file = "config.yaml"
        config_secrets_file = "config_secret.yaml"
        configure_remote_tracking(config_file, config_secrets_file)
    mlflow.set_experiment("mtg-artist-classification")
    with mlflow.start_run():
        main(train_data_folder, val_data_folder, batch_size, num_epochs)
