import argparse
import glob
import json
import os
import time
from tempfile import TemporaryDirectory

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from azure.ai.ml import MLClient
from azure.identity import EnvironmentCredential
from custom_fc_layer import CustomFullyConnectedLayer
from prepare_data_constants import IMAGE_TRANSFORMS
from torch.optim import lr_scheduler
from torchvision import models
from torchvision.datasets import ImageFolder


def create_artist_json(dataset: ImageFolder, artist_mapping_file: str):
    artist_mapping = {val: key for key, val in dataset.class_to_idx.items()}
    with open(artist_mapping_file, "w") as fp:
        json.dump(artist_mapping, fp)


def train_model(
    model,
    loss_function,
    optimizer,
    scheduler,
    datasets: dict[str, ImageFolder],
    device: torch.device,
    batch_size: int,
    num_epochs,
):
    # Time it, create data loaders
    since = time.time()
    dataset_sizes = {x: len(datasets[x]) for x in ["train", "val"]}
    dataloaders: dict[str, torch.utils.data.DataLoader] = {
        "train": torch.utils.data.DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=True, num_workers=4
        ),
        "val": torch.utils.data.DataLoader(
            datasets["val"], batch_size=batch_size, shuffle=False, num_workers=4
        ),
    }
    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        best_loss = np.inf

        for epoch in range(num_epochs):
            epoch_string = f"Epoch {epoch+1}/{num_epochs}"
            print(f"{epoch_string:=^40}")

            # We'll do something slightly different depending if we're training or validating
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                total_loss = 0.0
                total_correct_predictions = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients and optimize
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = loss_function(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # Track loss and accuracy for this batch
                    total_loss += loss.item()
                    total_correct_predictions += torch.sum(preds == labels.data)

                epoch_loss = total_loss / dataset_sizes[phase]
                epoch_acc = total_correct_predictions.double() / dataset_sizes[phase]

                print(f"{phase} Loss: {epoch_loss:.2f} Acc: {epoch_acc*100:.2f}%")

                # Copy the model if it is the best
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), best_model_params_path)

                # Step the scheduler to adjust the learning rate if the validation loss
                # has stagnated
                if phase == "val":
                    scheduler.step(epoch_loss)
                    print(f"current learning rate: {optimizer.param_groups[0]['lr']}")

        # Print a summary and log the accuracy metric
        time_elapsed = time.time() - since
        print(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f"Best val Acc: {epoch_acc*100:.2f}%")
        mlflow.log_metric("accuracy", best_acc)
        mlflow.log_metric("loss", best_loss)

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model


def get_num_classes() -> int:
    """When the data pipeline runs and gets all of the card images, the last
    step outputs the number of artists to classify to model_params. This function
    retrieves that for the purposes of building the classifier

    Returns:
        int: The number of artists to classify
    """
    this_files_folder = os.path.dirname(os.path.realpath(__file__))
    model_params_fp = os.path.join(this_files_folder, "model_params.yaml")
    with open(model_params_fp, "r") as fp:
        params_dict = yaml.safe_load(fp)
    num_classes = params_dict["num_artists"]

    return int(num_classes)


def main(
    train_data_folder: str, val_data_folder: str, batch_size: int, num_epochs: int
):
    mlflow.log_param("batch size", batch_size)
    mlflow.log_param("num_epochs", num_epochs)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    # Load up the data sets, model
    train_dataset = ImageFolder(
        root=train_data_folder,
        transform=IMAGE_TRANSFORMS,
    )
    val_dataset = ImageFolder(
        root=val_data_folder,
        transform=IMAGE_TRANSFORMS,
    )
    datasets: dict[str, ImageFolder] = {"train": train_dataset, "val": val_dataset}

    this_files_folder = os.path.dirname(os.path.realpath(__file__))

    artist_mapping_file = os.path.join(this_files_folder, "artist_mapping.json")
    print(artist_mapping_file)
    create_artist_json(train_dataset, artist_mapping_file)
    model = models.resnet18(weights="IMAGENET1K_V1")

    # Freeze the weights of the base model, create custom fully connected layer
    num_classes = get_num_classes()
    for param in model.parameters():
        param.requires_grad = False
    fully_connected = CustomFullyConnectedLayer(model, num_classes)
    model.fc = fully_connected
    model = model.to(device)

    # Train the model
    loss_function = nn.CrossEntropyLoss(reduction="sum")
    optimizer = optim.Adam(
        model.fc.parameters(),
        lr=0.001,
    )
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", patience=5
    )
    model = train_model(
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        datasets=datasets,
        device=device,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )
    torch.save(model.state_dict(), "./saved_model")

    # log the model
    this_files_dir = os.path.dirname(os.path.realpath(__file__))
    code_paths = [
        file_
        for file_ in glob.glob(os.path.join(this_files_dir, "*"))
        if "secret" not in file_
    ]
    for file_ in glob.glob(os.path.join(this_files_dir, "*")):
        print(file_)
    mlflow.pytorch.log_model(model, "model", code_paths=code_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-folder")
    parser.add_argument("--val-data-folder")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-epochs", type=int)

    # Optional azure info
    parser.add_argument("--remote-tracking", type=bool, default=False)
    args = parser.parse_args()

    train_data_folder = args.train_data_folder
    val_data_folder = args.val_data_folder
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    remote_tracking = args.remote_tracking

    if remote_tracking:
        this_files_dir = os.path.dirname(os.path.realpath(__file__))
        azure_config_file = os.path.join(this_files_dir, "azure_config.json")
        azure_secrets_config_file = os.path.join(
            this_files_dir, "azure_config_secrets.json"
        )
        with open(azure_config_file, "r") as f:
            azure_config_dict = json.load(fp=f)
        with open(azure_secrets_config_file, "r") as f:
            azure_secrets_config_dict = json.load(fp=f)
        os.environ["AZURE_TENANT_ID"] = azure_config_dict["SERVICE_PRINCIPAL_TENANT_ID"]
        os.environ["AZURE_CLIENT_ID"] = azure_config_dict["SERVICE_PRINCIPAL_CLIENT_ID"]
        os.environ["AZURE_CLIENT_SECRET"] = azure_secrets_config_dict[
            "SERVICE_PRINCIPAL_CLIENT_SECRET"
        ]
        environment_credential = EnvironmentCredential()

        ml_client = MLClient(
            subscription_id=azure_config_dict["AZURE_SUBSCRIPTION_ID"],
            resource_group_name=azure_config_dict["RESOURCE_GROUP_NAME"],
            credential=environment_credential,
            workspace_name=azure_config_dict["WORKSPACE_NAME"],
        )
        mlflow_tracking_id = ml_client.workspaces.get(
            ml_client.workspace_name
        ).mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_id)
    mlflow.set_experiment("mtg-artist-classification")
    with mlflow.start_run():
        main(train_data_folder, val_data_folder, batch_size, num_epochs)
