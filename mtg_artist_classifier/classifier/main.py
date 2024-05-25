import glob
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
from config_utils import create_artist_json
from custom_fc_layer import CustomFullyConnectedLayer
from prepare_data_constants import IMAGE_TRANSFORMS
from torch.optim import lr_scheduler
from torchvision import models
from torchvision.datasets import ImageFolder


def take_train_step(
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    loss_function: torch.nn.Module,
) -> tuple[float, float]:
    total_loss = 0.0
    total_correct_predictions = 0
    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients and optimize
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

        # Track loss and accuracy for this batch
        total_loss += loss.item()
        total_correct_predictions += torch.sum(preds == labels.data)
    return total_loss, total_correct_predictions


def take_eval_step(
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    loss_function: torch.nn.Module,
) -> tuple[float, float]:
    total_loss = 0.0
    total_correct_predictions = 0

    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients and optimize
        optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_function(outputs, labels)

        # Track loss and accuracy for this batch
        total_loss += loss.item()
        total_correct_predictions += torch.sum(preds == labels.data)
    return total_loss, total_correct_predictions


def train_model(
    model,
    loss_function,
    optimizer,
    scheduler,
    datasets: dict[str, ImageFolder],
    device: torch.device,
    batch_size: int,
    num_epochs: int,
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
                    total_loss, total_correct_predictions = take_train_step(
                        device,
                        dataloaders["train"],
                        optimizer,
                        model,
                        loss_function,
                    )
                else:
                    model.eval()
                    total_loss, total_correct_predictions = take_eval_step(
                        device,
                        dataloaders["val"],
                        optimizer,
                        model,
                        loss_function,
                    )

                epoch_loss = total_loss / dataset_sizes[phase]
                epoch_acc = total_correct_predictions.double() / dataset_sizes[phase]

                print(f"{phase} Loss: {epoch_loss:.2f} Acc: {epoch_acc*100:.2f}%")

                # Copy the model if it is the best
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), best_model_params_path)

                # Step the scheduler to adjust the learning rate if the validation loss has stagnated
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


def create_datasets(
    train_data_folder: str, val_data_folder: str
) -> dict[str, ImageFolder]:
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
    return datasets


def configure_model(artist_mapping_file: dict[str, str], device: str):
    # Freeze the weights of the base model, create custom fully connected layer
    model = models.resnet18(weights="IMAGENET1K_V1")
    num_classes = len(artist_mapping_file)
    for param in model.parameters():
        param.requires_grad = False
    fully_connected = CustomFullyConnectedLayer(model, num_classes)
    model.fc = fully_connected
    model = model.to(device)
    return model


def main(
    train_data_folder: str, val_data_folder: str, batch_size: int, num_epochs: int
):
    mlflow.log_param("batch size", batch_size)
    mlflow.log_param("num_epochs", num_epochs)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    datasets = create_datasets(train_data_folder, val_data_folder)

    # create a mapping from class num to artist, save
    this_files_folder = os.path.dirname(os.path.realpath(__file__))
    artist_mapping_file = os.path.join(this_files_folder, "artist_mapping.json")
    create_artist_json(datasets["train"], artist_mapping_file)

    model = configure_model(artist_mapping_file, device)

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
    mlflow.pytorch.log_model(model, "model", code_paths=code_paths)

    # now that everything is logged, we can tear down the artist_mapping.json we created before
    os.remove(artist_mapping_file)


def configure_remote_tracking(config_file: str, config_secrets_file: str):
    """Sets up remote tracking environment variables, remote tracking id"""

    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
        azure_config_dict = config_dict["azure_ml"]["dev"]
    with open(config_secrets_file, "r") as f:
        config_secrets_dict = yaml.safe_load(f)
        azure_secrets_config_dict = config_secrets_dict["azure_ml"]["dev"]
    os.environ["AZURE_TENANT_ID"] = azure_config_dict["service_principal_tenant_id"]
    os.environ["AZURE_CLIENT_ID"] = azure_config_dict["service_principal_client_id"]
    os.environ["AZURE_CLIENT_SECRET"] = azure_secrets_config_dict[
        "service_principal_client_secret"
    ]
    environment_credential = EnvironmentCredential()

    ml_client = MLClient(
        subscription_id=azure_config_dict["azure_subscription_id"],
        resource_group_name=azure_config_dict["resource_group_name"],
        credential=environment_credential,
        workspace_name=azure_config_dict["workspace_name"],
    )
    mlflow_tracking_id = ml_client.workspaces.get(
        ml_client.workspace_name
    ).mlflow_tracking_uri
    mlflow.set_tracking_uri(mlflow_tracking_id)
