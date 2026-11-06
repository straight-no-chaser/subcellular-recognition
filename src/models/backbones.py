import torch
import torch.nn as nn
import torchvision.models as tvm


class SmallCNN(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def adapt_first_conv(conv, in_channels):
    if conv.in_channels == in_channels:
        return conv

    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups if in_channels % conv.groups == 0 else 1,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )

    with torch.no_grad():
        if in_channels == 1:
            weight = conv.weight.mean(dim=1, keepdim=True)
        elif conv.in_channels == 1:
            weight = conv.weight.repeat(1, in_channels, 1, 1) / float(in_channels)
        elif in_channels < conv.in_channels:
            weight = conv.weight[:, :in_channels]
        else:
            repeats = (in_channels + conv.in_channels - 1) // conv.in_channels
            weight = conv.weight.repeat(1, repeats, 1, 1)[:, :in_channels]
            weight = weight / float(repeats)
        new_conv.weight.copy_(weight)
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


def _weights(name, pretrained):
    if not pretrained:
        return None
    if name == "resnet18":
        return tvm.ResNet18_Weights.DEFAULT
    if name == "densenet121":
        return tvm.DenseNet121_Weights.DEFAULT
    if name == "efficientnet_b0":
        return tvm.EfficientNet_B0_Weights.DEFAULT
    if name == "vgg16":
        return tvm.VGG16_Weights.DEFAULT
    return None


def build_model(name, in_channels=1, pretrained=False):
    name = str(name).lower()

    if name == "smallcnn":
        return SmallCNN(in_channels=in_channels)

    if name == "resnet18":
        model = tvm.resnet18(weights=_weights(name, pretrained))
        model.fc = nn.Linear(model.fc.in_features, 1)
        model.conv1 = adapt_first_conv(model.conv1, in_channels)
        return model

    if name == "densenet121":
        model = tvm.densenet121(weights=_weights(name, pretrained))
        model.classifier = nn.Linear(model.classifier.in_features, 1)
        model.features.conv0 = adapt_first_conv(model.features.conv0, in_channels)
        return model

    if name == "efficientnet_b0":
        model = tvm.efficientnet_b0(weights=_weights(name, pretrained))
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        model.features[0][0] = adapt_first_conv(model.features[0][0], in_channels)
        return model

    if name == "vgg16":
        model = tvm.vgg16(weights=_weights(name, pretrained))
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1)
        model.features[0] = adapt_first_conv(model.features[0], in_channels)
        return model

    raise ValueError(f"Unknown model architecture: {name}")
