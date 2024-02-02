import argparse
import os
import time
from tempfile import TemporaryDirectory

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from custom_fc_layer import CustomFullyConnectedLayer
from prepare_data_constants import IMAGE_TRANSFORMS
from torch.optim import lr_scheduler
from torchvision import models
from torchvision.datasets import ImageFolder


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
        mlflow.log_metric("loss", total_loss)

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

    # log the model
    this_files_dir = os.path.dirname(os.path.realpath(__file__))
    mlflow.pytorch.log_model(
        model,
        "model",
        code_paths=[os.path.join(this_files_dir, "prepare_data_constants.py")],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-folder")
    parser.add_argument("--val-data-folder")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-epochs", type=int)
    args = parser.parse_args()

    train_data_folder = args.train_data_folder
    val_data_folder = args.val_data_folder
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    mlflow.set_experiment("mtg-artist-classification")
    with mlflow.start_run():
        main(train_data_folder, val_data_folder, batch_size, num_epochs)
