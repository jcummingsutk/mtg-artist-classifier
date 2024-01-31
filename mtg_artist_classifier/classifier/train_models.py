import argparse
import os
import time
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn
import torch.optim as optim
from models import FullyConnectedLayer
from prepare_data_constants import IMAGE_TRANSFORMS
from torch.optim import lr_scheduler
from torchvision import models
from torchvision.datasets import ImageFolder


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    datasets: dict[str, ImageFolder],
    device: torch.device,
    num_epochs=25,
):
    since = time.time()
    dataset_sizes = {x: len(datasets[x]) for x in ["train", "val"]}
    dataloaders: dict[str, torch.utils.data.DataLoader] = {
        "train": torch.utils.data.DataLoader(
            datasets["train"], batch_size=4, shuffle=True, num_workers=4
        ),
        "val": torch.utils.data.DataLoader(
            datasets["val"], batch_size=4, shuffle=True, num_workers=4
        ),
    }
    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs - 1}")
            print("-" * 10)

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                # deep copy the model
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)
                if phase == "val":
                    scheduler.step(epoch_loss)
                    print(optimizer.param_groups[0]["lr"])

            print()

        time_elapsed = time.time() - since
        print(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f"Best val Acc: {best_acc:4f}")

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-folder")
    parser.add_argument("--val-data-folder")
    args = parser.parse_args()
    train_data_folder = args.train_data_folder
    val_data_folder = args.val_data_folder
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    train_dataset = ImageFolder(
        root=train_data_folder,
        transform=IMAGE_TRANSFORMS,
    )
    val_dataset = ImageFolder(
        root=val_data_folder,
        transform=IMAGE_TRANSFORMS,
    )
    datasets: dict[str, ImageFolder] = {"train": train_dataset, "val": val_dataset}
    model_conv = models.resnet18(weights="IMAGENET1K_V1")
    num_classes = 5
    for param in model_conv.parameters():
        param.requires_grad = False
    fully_connected = FullyConnectedLayer(model_conv, num_classes)
    model_conv.fc = fully_connected
    model_conv = model_conv.to(device)
    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.

    optimizer_conv = optim.Adam(
        model_conv.fc.parameters(),
        lr=0.001,
    )

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_conv, "min")
    model_conv = train_model(
        model_conv,
        criterion,
        optimizer_conv,
        scheduler,
        datasets,
        device,
        num_epochs=25,
    )


if __name__ == "__main__":
    main()
