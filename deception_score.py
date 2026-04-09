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


# ── Artist label mapping from split.hdf5 ─────────────────────────

ARTISTS = [
    "claude-monet", "paul-cezanne", "el-greco", "paul-gauguin",
    "samuel-peploe", "vincent-van-gogh", "edvard-munch", "pablo-picasso",
    "berthe-morisot", "ernst-ludwig-kirchner", "jackson-pollock",
    "wassily-kandinsky", "nicholas-roerich",
]


def _find_checkpoint_prefix(root: str) -> Optional[str]:
    """Return checkpoint path prefix (no .index / .data*) with largest global step."""
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


def _root_has_assets(root: str) -> bool:
    split_path = os.path.join(root, "split.hdf5")
    return os.path.isfile(split_path) and _find_checkpoint_prefix(root) is not None


def get_deception_paths(
    ckpt_dir: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    Locate deception assets.

    Returns:
        root_dir: directory containing split.hdf5 and checkpoints
        split_hdf5_path: full path to split.hdf5
        checkpoint_prefix: full path prefix for tf.train.load_checkpoint (no extension)
    """
    search: List[str] = []
    if ckpt_dir:
        search.append(os.path.abspath(ckpt_dir))
    for name in ("deception_score_vgg", "evaluation_data"):
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

    raise FileNotFoundError(
        "Deception assets not found. Expected split.hdf5 and model.ckpt-* under "
        "deception_score_vgg/ or evaluation_data/. Run:\n"
        "  python3 setup_deception.py"
    )


def _artist_from_index(index_value: str) -> str:
    """
    Parse artist slug from split.hdf5 index name.

    Expected index format is typically "<artist-slug>_<image-id>".
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


def _load_split_from_pandas_fixed_hdf5(path: str) -> pd.DataFrame:
    """
    Read pandas \"fixed\" HDF5 layout (Sanakoyeu evaluation split).

    Contains /df/axis1 (pickled row index), /df/block1_values (labels), etc.
    """
    import h5py

    with h5py.File(path, "r") as f:
        if "df" not in f:
            raise ValueError("No /df group in HDF5.")
        g = f["df"]
        for key in ("axis1", "block1_items", "block1_values"):
            if key not in g:
                raise ValueError(f"Missing /df/{key} (not pandas fixed-format).")

        items = g["block1_items"][:]
        colnames = [
            x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
            for x in items
        ]
        if len(colnames) != 1 or colnames[0] != "label":
            raise ValueError(f"Expected block1_items ['label'], got {colnames}")

        labels = np.asarray(g["block1_values"])[:, 0].astype(np.int64, copy=False)

        ax = g["axis1"]
        raw = ax[0]
        blob = np.asarray(raw).tobytes()
        try:
            _vis_dep = np.exceptions.VisibleDeprecationWarning
        except AttributeError:
            _vis_dep = np.VisibleDeprecationWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=_vis_dep)
            index = pickle.loads(blob)
        index = np.asarray(index, dtype=object)
        if index.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Index length {index.shape[0]} != label length {labels.shape[0]}"
            )
        str_index = [_h5_scalar_to_str(i) for i in index]
        return pd.DataFrame({"label": labels}, index=str_index)


def _load_split_via_pytables_table(split_hdf5_path: str) -> pd.DataFrame:
    """Legacy PyTables Table layout with 'label' and optional 'index' columns."""
    import tables as tb

    h5 = tb.open_file(split_hdf5_path, mode="r")
    try:
        table = None
        if hasattr(h5.root, "df") and hasattr(h5.root.df, "table"):
            table = h5.root.df.table
        if table is None:
            for node in h5.walk_nodes(classname="Table"):
                if "label" in node.colnames:
                    table = node
                    break
        if table is None:
            raise ValueError("No HDF5 Table with a 'label' column.")

        rec = table.read()
        if "label" not in rec.dtype.names:
            raise ValueError("Table is missing 'label' field.")
        labels = rec["label"].astype("int64", copy=False)

        if "index" in rec.dtype.names:
            index = [_h5_scalar_to_str(x) for x in rec["index"]]
        else:
            index = np.arange(len(labels), dtype=np.int64).astype(str)

        return pd.DataFrame({"label": labels}, index=index)
    finally:
        h5.close()


def _load_split_dataframe(split_hdf5_path: str) -> pd.DataFrame:
    """
    Load the train/val split table from split.hdf5.

    Tries pandas first, then the fixed-format layout used by Sanakoyeu's release,
    then a generic PyTables Table layout.
    """
    errors: List[BaseException] = []
    try:
        return pd.read_hdf(split_hdf5_path)
    except Exception as err:
        errors.append(err)

    try:
        return _load_split_from_pandas_fixed_hdf5(split_hdf5_path)
    except Exception as err:
        errors.append(err)

    try:
        return _load_split_via_pytables_table(split_hdf5_path)
    except Exception as err:
        errors.append(err)

    raise RuntimeError(
        "Could not read split.hdf5 (tried pandas, fixed HDF5, PyTables Table). "
        f"Last error: {errors[-1]}"
    ) from errors[-1]


def get_artist_labels(
    split_hdf5_path: str,
    allowed_artists: Optional[Iterable[str]] = None,
) -> Dict[str, int]:
    """
    Read artist -> integer label mapping from split.hdf5.

    By default this returns all artists found in the split file (typically 624).
    Optionally filter with allowed_artists.
    """
    df = _load_split_dataframe(split_hdf5_path)
    if "label" not in df.columns:
        raise ValueError(f"'label' column not found in {split_hdf5_path}")

    labels: Dict[str, int] = {}
    for idx, row in df.iterrows():
        artist = _artist_from_index(str(idx))
        label = int(row["label"])
        if artist in labels and labels[artist] != label:
            raise ValueError(
                f"Inconsistent label mapping for artist '{artist}': "
                f"{labels[artist]} vs {label}"
            )
        labels[artist] = label

    if allowed_artists is not None:
        allowed_set = set(allowed_artists)
        labels = {k: v for k, v in labels.items() if k in allowed_set}

    return labels


# ── VGG-16 inference via TensorFlow ──────────────────────────────

# VGG ImageNet mean (RGB order)
_VGG_MEAN = [123.68, 116.779, 103.939]


class DeceptionScoreModel:
    """
    Wraps the TF-Slim VGG-16 artist classifier checkpoint.
    Builds the VGG-16 graph manually and restores weights from checkpoint.
    """

    def __init__(self, ckpt_prefix: str):
        """
        Args:
            ckpt_prefix: Path prefix passed to tf.train.load_checkpoint, e.g.
                .../model.ckpt-790000 (no .index / .data* suffix).
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

def preprocess_image(
    img_path_or_tensor: Union[str, os.PathLike, "np.ndarray", "object"]
) -> np.ndarray:
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
        if tensor.ndim != 4 or tensor.shape[1] != 3:
            raise ValueError(
                "Tensor input must have shape [B, 3, H, W]. "
                f"Got: {tuple(tensor.shape)}"
            )
        tensor = torch.nn.functional.interpolate(
            tensor, size=224, mode="bilinear", align_corners=False
        )
        arr = tensor[0].permute(1, 2, 0).cpu().numpy() * 255.0

    # Subtract VGG mean
    arr = arr - np.array(_VGG_MEAN, dtype=np.float32)
    return arr[np.newaxis]  # [1, 224, 224, 3]


# ── Deception score computation ──────────────────────────────────

def load_deception_model(ckpt_dir=None):
    """Load the VGG-16 artist classifier from TF checkpoint."""
    _, _, ckpt_prefix = get_deception_paths(ckpt_dir)
    return DeceptionScoreModel(ckpt_prefix)


def default_split_hdf5_path(ckpt_dir=None) -> str:
    """Path to split.hdf5 next to the resolved checkpoint."""
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
    Compute the deception rate: fraction of images classified as target_artist.

    Args:
        model: DeceptionScoreModel
        image_paths_or_tensors: list of file paths or [1,3,H,W] tensors
        target_artist: artist slug (e.g. 'vincent-van-gogh')
        split_hdf5_path: path to split.hdf5 (default: resolved next to checkpoint)
        ckpt_dir: optional deception root if split_hdf5_path is None

    Returns:
        deception_rate: float in [0, 1]
        details: dict with per-image predictions
    """
    if split_hdf5_path is None:
        split_hdf5_path = default_split_hdf5_path(ckpt_dir)
    if artist_labels is None:
        artist_labels = get_artist_labels(split_hdf5_path)
    if target_artist not in artist_labels:
        sample = sorted(artist_labels.keys())[:20]
        raise ValueError(
            f"Unknown artist '{target_artist}'. "
            f"First labels in split (showing up to 20): {sample} "
            f"({len(artist_labels)} artists total)."
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
        "num_known_artists": len(artist_labels),
    }
