import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py

# Pretrained VGG19 feature extractor to compute content loss.
# Content loss is the L2 loss between the features of the rendered image and the content image.

# Weights available at the following links:
### -  https://www.dropbox.com/s/hv7b4eajrj7isyq/vgg_weights.pickle?dl=1
### -  https://github.com/ftokarev/tf-vgg-weights/raw/master/vgg19_weights_normalized.h5

class Normalization(nn.Module):
    """VGG preprocessing: RGB [0,1] -> BGR, subtract ImageNet means."""
    def __init__(self):
        super().__init__()
        # RGB -> BGR conversion kernel
        self.register_buffer(
            "kern",
            torch.tensor(
                [[[0], [0], [255]], [[0], [255], [0]], [[255], [0], [0]]],
                dtype=torch.float32,
            ).unsqueeze(-1),  # [3, 3, 1, 1]
        )
        self.register_buffer(
            "bias",
            torch.tensor([-103.939, -116.779, -123.68], dtype=torch.float32),
        )

    def forward(self, x):
        return F.conv2d(x, self.kern, bias=self.bias)


class ConvRelu(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=0)
        self.relu = nn.ReLU()


class VGG19(nn.Sequential):
    """VGG19 with normalized weights like Gatys et al. for feature extraction."""

    def __init__(self, weight_file="vgg_weights/vgg19_weights_normalized.h5"):
        super().__init__()
        self.norm = Normalization()

        self.conv1_1 = ConvRelu(3, 64)
        self.conv1_2 = ConvRelu(64, 64)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2_1 = ConvRelu(64, 128)
        self.conv2_2 = ConvRelu(128, 128)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.conv3_1 = ConvRelu(128, 256)
        self.conv3_2 = ConvRelu(256, 256)
        self.conv3_3 = ConvRelu(256, 256)
        self.conv3_4 = ConvRelu(256, 256)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.conv4_1 = ConvRelu(256, 512)
        self.conv4_2 = ConvRelu(512, 512)
        self.conv4_3 = ConvRelu(512, 512)
        self.conv4_4 = ConvRelu(512, 512)
        self.pool4 = nn.MaxPool2d(2, stride=2)

        self.conv5_1 = ConvRelu(512, 512)
        self.conv5_2 = ConvRelu(512, 512)
        self.conv5_3 = ConvRelu(512, 512)
        self.conv5_4 = ConvRelu(512, 512)
        self.pool5 = nn.MaxPool2d(2, stride=2)

        for p in self.parameters():
            p.requires_grad_(False)

        self._load_weights(weight_file)

    def _load_weights(self, path):
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(f"Downloading VGG19 weights to {path} ...")
            url = (
                "https://github.com/ftokarev/tf-vgg-weights/"
                "raw/master/vgg19_weights_normalized.h5"
            )
            try:
                import urllib.request

                urllib.request.urlretrieve(url, path)
            except Exception:
                import requests

                r = requests.get(url, timeout=120)
                r.raise_for_status()
                with open(path, "wb") as f:
                    f.write(r.content)
            print("Done.")

        with h5py.File(path, "r") as f:
            trained = [np.array(v, dtype="float32") for _, v in f.items()]

        for p, tp in zip(self.parameters(), trained):
            if len(tp.shape) == 4:
                tp = np.transpose(tp, (3, 2, 0, 1))
            p.data.copy_(torch.from_numpy(tp))
        print("VGG19 weights loaded.")
