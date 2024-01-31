from torchvision import transforms

MEANS = [0.5, 0.5, 0.5]
STDS = [0.2, 0.2, 0.2]
IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEANS, STDS),
    ]
)

DATA_LOADER_BATCH_SIZE = 4
