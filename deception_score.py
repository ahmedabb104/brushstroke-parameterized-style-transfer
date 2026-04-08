"""
Deception Score evaluation.

Uses a VGG-16 network trained to classify WikiArt paintings into 624 artists
(from Sanakoyeu et al.). The deception rate is the fraction of stylized images
that the network assigns to the target artist.

The pretrained model is a TF checkpoint loaded directly via TensorFlow.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


# ── Artist label mapping from split.hdf5 ─────────────────────────

ARTISTS = [
    "claude-monet", "paul-cezanne", "el-greco", "paul-gauguin",
    "samuel-peploe", "vincent-van-gogh", "edvard-munch", "pablo-picasso",
    "berthe-morisot", "ernst-ludwig-kirchner", "jackson-pollock",
    "wassily-kandinsky", "nicholas-roerich",
]


def get_artist_labels(split_hdf5_path):
    """Read artist -> integer label mapping from the split.hdf5 file."""
    df = pd.read_hdf(split_hdf5_path)
    labels = {}
    for artist in ARTISTS:
        cur = df[df.index.str.startswith(artist + "_")]
        if len(cur) > 0:
            labels[artist] = int(cur["label"].iloc[0])
    return labels


# ── VGG-16 inference via TensorFlow ──────────────────────────────

# VGG ImageNet mean (RGB order)
_VGG_MEAN = [123.68, 116.779, 103.939]


class DeceptionScoreModel:
    """
    Wraps the TF-Slim VGG-16 artist classifier checkpoint.
    Builds the VGG-16 graph manually and restores weights from checkpoint.
    """

    def __init__(self, ckpt_path="deception_score_vgg/model.ckpt-790000"):
        self.reader = tf.train.load_checkpoint(ckpt_path)
        self._weights_cache = {}
        print("Deception score model loaded from TF checkpoint.")

    def _get_weight(self, name):
        if name not in self._weights_cache:
            self._weights_cache[name] = self.reader.get_tensor(name)
        return self._weights_cache[name]

    def _conv_relu(self, x, layer_name):
        """Apply conv2d + bias + ReLU using weights from checkpoint."""
        w = self._get_weight(f"{layer_name}/weights")
        b = self._get_weight(f"{layer_name}/biases")
        x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def _fc_as_conv(self, x, layer_name, relu=True):
        """Apply FC layer stored as conv weights."""
        w = self._get_weight(f"{layer_name}/weights")
        b = self._get_weight(f"{layer_name}/biases")
        x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="VALID")
        x = tf.nn.bias_add(x, b)
        if relu:
            x = tf.nn.relu(x)
        return x

    def predict(self, images_np):
        """
        Run VGG-16 classification on a batch of preprocessed images.

        Args:
            images_np: numpy array [B, 224, 224, 3], float32,
                       already mean-subtracted (RGB - VGG_MEAN)

        Returns:
            predictions: numpy array [B] of predicted class indices
        """
        x = tf.constant(images_np, dtype=tf.float32)

        # conv1
        x = self._conv_relu(x, "vgg_16/conv1/conv1_1")
        x = self._conv_relu(x, "vgg_16/conv1/conv1_2")
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding="VALID")

        # conv2
        x = self._conv_relu(x, "vgg_16/conv2/conv2_1")
        x = self._conv_relu(x, "vgg_16/conv2/conv2_2")
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding="VALID")

        # conv3
        x = self._conv_relu(x, "vgg_16/conv3/conv3_1")
        x = self._conv_relu(x, "vgg_16/conv3/conv3_2")
        x = self._conv_relu(x, "vgg_16/conv3/conv3_3")
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding="VALID")

        # conv4
        x = self._conv_relu(x, "vgg_16/conv4/conv4_1")
        x = self._conv_relu(x, "vgg_16/conv4/conv4_2")
        x = self._conv_relu(x, "vgg_16/conv4/conv4_3")
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding="VALID")

        # conv5
        x = self._conv_relu(x, "vgg_16/conv5/conv5_1")
        x = self._conv_relu(x, "vgg_16/conv5/conv5_2")
        x = self._conv_relu(x, "vgg_16/conv5/conv5_3")
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding="VALID")

        # fc6, fc7, fc8 (no dropout at inference)
        x = self._fc_as_conv(x, "vgg_16/fc6", relu=True)
        x = self._fc_as_conv(x, "vgg_16/fc7", relu=True)
        x = self._fc_as_conv(x, "vgg_16/fc8_artist_id", relu=False)

        logits = tf.squeeze(x, axis=[1, 2])  # [B, 624]
        return tf.argmax(logits, axis=1).numpy()


# ── Image preprocessing ──────────────────────────────────────────

def preprocess_image(img_path_or_tensor):
    """
    Preprocess a single image for VGG-16 classification.
    Accepts either a file path or a PyTorch [1, 3, H, W] tensor in [0, 1].
    Returns numpy array [1, 224, 224, 3] (mean-subtracted, NHWC).
    """
    if isinstance(img_path_or_tensor, (str, os.PathLike)):
        img = Image.open(img_path_or_tensor).convert("RGB")
        img = img.resize((224, 224), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)  # [224, 224, 3] in [0, 255]
    else:
        import torch
        tensor = img_path_or_tensor
        # PyTorch tensor [1, 3, H, W] in [0, 1] -> numpy [H, W, 3] in [0, 255]
        tensor = torch.nn.functional.interpolate(
            tensor, size=224, mode="bilinear", align_corners=False
        )
        arr = tensor[0].permute(1, 2, 0).cpu().numpy() * 255.0

    # Subtract VGG mean
    arr = arr - np.array(_VGG_MEAN, dtype=np.float32)
    return arr[np.newaxis]  # [1, 224, 224, 3]


# ── Deception score computation ──────────────────────────────────

def load_deception_model(ckpt_dir="deception_score_vgg"):
    """Load the VGG-16 artist classifier from TF checkpoint."""
    ckpt_path = os.path.join(ckpt_dir, "model.ckpt-790000")
    return DeceptionScoreModel(ckpt_path)


def compute_deception_rate(
    model, image_paths_or_tensors, target_artist,
    split_hdf5_path="deception_score_vgg/split.hdf5",
):
    """
    Compute the deception rate: fraction of images classified as target_artist.

    Args:
        model: DeceptionScoreModel
        image_paths_or_tensors: list of file paths or [1,3,H,W] tensors
        target_artist: artist slug (e.g. 'vincent-van-gogh')
        split_hdf5_path: path to split.hdf5 for label mapping

    Returns:
        deception_rate: float in [0, 1]
        details: dict with per-image predictions
    """
    artist_labels = get_artist_labels(split_hdf5_path)
    if target_artist not in artist_labels:
        raise ValueError(
            f"Unknown artist '{target_artist}'. Available: {list(artist_labels.keys())}"
        )
    target_label = artist_labels[target_artist]

    # Batch all images
    batch = np.concatenate(
        [preprocess_image(item) for item in image_paths_or_tensors], axis=0
    )
    predictions = model.predict(batch).tolist()

    correct = sum(1 for p in predictions if p == target_label)
    total = len(predictions)
    deception_rate = correct / total if total > 0 else 0.0

    return deception_rate, {
        "target_artist": target_artist,
        "target_label": target_label,
        "predictions": predictions,
        "correct": correct,
        "total": total,
    }
