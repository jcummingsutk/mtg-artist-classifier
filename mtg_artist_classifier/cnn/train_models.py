from torchvision import models

if __name__ == "__main__":
    model = models.resnet50(pretrained=True)
