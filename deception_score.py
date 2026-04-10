"""
Deception Score evaluation.

Uses a VGG-16 network trained to classify WikiArt paintings into 624 artists
(from Sanakoyeu et al.). The deception rate is the fraction of stylized images
that the network assigns to the target artist.

The pretrained model is a TF checkpoint loaded directly via TensorFlow.
"""

import glob
import os
import pickle
import re
import warnings
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


# artist labels from split.hdf5 (Sanakoyeu et al.)

ARTISTS = [
    "claude-monet", "paul-cezanne", "el-greco", "paul-gauguin",
    "samuel-peploe", "vincent-van-gogh", "edvard-munch", "pablo-picasso",
    "berthe-morisot", "ernst-ludwig-kirchner", "jackson-pollock",
    "wassily-kandinsky", "nicholas-roerich",
]


def _find_checkpoint_prefix(root: str) -> Optional[str]:
    """Return checkpoint path prefix."""
    best_prefix: Optional[str] = None
    best_step = -1
    for path in glob.glob(os.path.join(root, "model.ckpt-*.index")):
        m = re.search(r"model\.ckpt-(\d+)\.index$", path)
        if not m:
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step = step
            best_prefix = path[: -len(".index")]
    return best_prefix


def get_deception_paths(
    ckpt_dir: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    Locate deception score assets.
    """
    search: List[str] = []
    if ckpt_dir:
        search.append(os.path.abspath(ckpt_dir))
    for name in ("deception_score_vgg"):
        p = os.path.abspath(name)
        if p not in search:
            search.append(p)

    for root in search:
        if not os.path.isdir(root):
            continue
        split_path = os.path.join(root, "split.hdf5")
        ckpt_prefix = _find_checkpoint_prefix(root)
        if os.path.isfile(split_path) and ckpt_prefix:
            return root, split_path, ckpt_prefix


def _artist_from_index(index_value: str) -> str:
    """
    Parse artist slug from split.hdf5 index name.
    """
    idx = str(index_value)
    if "_" not in idx:
        return idx
    return idx.rsplit("_", 1)[0]


def _h5_scalar_to_str(val: object) -> str:
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, np.ndarray) and val.shape == ():
        return _h5_scalar_to_str(val.item())
    return str(val)


def get_artist_labels(
    split_hdf5_path: str,
    allowed_artists: Optional[Iterable[str]] = None,
) -> Dict[str, int]:
    """
    Read artist -> integer label mapping from split.hdf5.
    (typically 624 artists).
    """
    df = pd.read_hdf(split_hdf5_path)
    labels: Dict[str, int] = {}
    for idx, row in df.iterrows():
        artist = _artist_from_index(str(idx))
        label = int(row["label"])
        labels[artist] = label

    if allowed_artists is not None:
        allowed_set = set(allowed_artists)
        labels = {k: v for k, v in labels.items() if k in allowed_set}
    return labels


# VGG-16 inference via TensorFlow

class DeceptionScoreModel:
    """
    Wraps the VGG-16 artist classifier checkpoint.
    Builds the VGG-16 graph manually and restores weights from checkpoint.
    """

    def __init__(self, ckpt_prefix: str):
        """
        Args:
            ckpt_prefix: e.g.
                ../model.ckpt-790000
        """
        self.ckpt_prefix = ckpt_prefix
        self.reader = tf.train.load_checkpoint(ckpt_prefix)
        self._weights_cache = {}
        print(f"Deception score model loaded from TF checkpoint: {ckpt_prefix}")

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

        Input:
            images_np: numpy array [B, 224, 224, 3], float32

        Output:
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

        # fc6, fc7, fc8
        x = self._fc_as_conv(x, "vgg_16/fc6", relu=True)
        x = self._fc_as_conv(x, "vgg_16/fc7", relu=True)
        x = self._fc_as_conv(x, "vgg_16/fc8_artist_id", relu=False)
        logits = tf.squeeze(x, axis=[1, 2])
        return tf.argmax(logits, axis=1).numpy()


def preprocess_image(
    img_path_or_tensor: Union[str, os.PathLike, "np.ndarray", "object"]
) -> np.ndarray:
    """
    Preprocess a single image for Sanakoyeu's VGG-16 classification.
    Accepts: either a file path or a PyTorch [1, 3, H, W] tensor in [0, 1].
    Returns: numpy array [1, 224, 224, 3].
    """
    if isinstance(img_path_or_tensor, (str, os.PathLike)):
        img = Image.open(img_path_or_tensor).convert("RGB")
        img = img.resize((224, 224), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) 
    else:
        import torch
        tensor = img_path_or_tensor
        tensor = torch.nn.functional.interpolate(
            tensor, size=224, mode="bilinear", align_corners=False
        )
        arr = tensor[0].permute(1, 2, 0).cpu().numpy() * 255.0

    # subtract VGG imagenet mean
    arr = arr - np.array([123.68, 116.779, 103.939], dtype=np.float32)
    return arr[np.newaxis]


# deception score computation using the DeceptionScoreModel

def load_deception_model(ckpt_dir=None):
    """load Sanakoyeu's VGG-16 artist classifier from TF checkpoint."""
    _, _, ckpt_prefix = get_deception_paths(ckpt_dir)
    return DeceptionScoreModel(ckpt_prefix)


def default_split_hdf5_path(ckpt_dir=None) -> str:
    """path to split.hdf5"""
    _, split_path, _ = get_deception_paths(ckpt_dir)
    return split_path


def compute_deception_rate(
    model: DeceptionScoreModel,
    image_paths_or_tensors: List[Union[str, os.PathLike, "object"]],
    target_artist: str,
    split_hdf5_path: Optional[str] = None,
    artist_labels: Optional[Dict[str, int]] = None,
    ckpt_dir: Optional[str] = None,
) -> Tuple[float, Dict[str, object]]:
    """
    compute the deception rate: fraction of images classified correctly as target_artist.

    Inputs:
        model: DeceptionScoreModel
        image_paths_or_tensors: list of file paths or [1,3,H,W] tensors
        target_artist: artist slug (e.g. 'vincent-van-gogh')
        split_hdf5_path: path to split.hdf5
        ckpt_dir: optional if split_hdf5_path is None

    Outputs:
        deception_rate: fraction of images classified correctly as target_artist
        details: dict with per-image predictions
    """
    if split_hdf5_path is None:
        split_hdf5_path = default_split_hdf5_path(ckpt_dir)
    if artist_labels is None:
        artist_labels = get_artist_labels(split_hdf5_path)
    target_label = artist_labels[target_artist]

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
        "num_known_artists": len(artist_labels),
    }
