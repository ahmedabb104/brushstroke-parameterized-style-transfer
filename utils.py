import os
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
from skimage.segmentation import slic
import torchvision.transforms as transforms
import torch


def image_loader(image_path, img_size, device):
    """Load and preprocess an image to a tensor [1, 3, H, W] in [0, 1]."""
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device, torch.float)


def clusters_to_strokes(segments, img, H, W, sec_scale=0.001, width_scale=1):
    """
    Convert SLIC superpixel segments into brushstroke parameters.

    For each superpixel cluster, finds the longest axis (via convex hull),
    computes width orthogonally, and extracts mean color.
    """
    segments += np.abs(np.min(segments))
    num_clusters = np.max(segments)

    centers, starts, ends, widths_list, colors_list = [], [], [], [], []
    num_pixels_list, stddev_list = [], []

    for idx in range(num_clusters + 1):
        mask = segments == idx
        if np.sum(mask) < 5:
            continue

        nonzero = np.nonzero(mask)
        points = np.stack((nonzero[0], nonzero[1]), axis=-1)

        try:
            hull = ConvexHull(points)
        except Exception:
            continue

        # Find the two farthest border points (longest axis of cluster)
        border_pts = points[hull.simplices.reshape(-1)]
        dists = np.sum((np.expand_dims(border_pts, 1) - border_pts) ** 2, axis=-1)
        max_a, max_b = np.nonzero(dists == np.max(dists))
        point_a = border_pts[max_a[0]]
        point_b = border_pts[max_b[0]]

        # Compute width via orthogonal intersection with hull
        v_ba = point_b - point_a
        v_orth = np.array([v_ba[1], -v_ba[0]])
        m = (point_a + point_b) / 2.0
        n = m + 0.5 * v_orth

        p = points[hull.simplices][:, 0]
        q = points[hull.simplices][:, 1]
        denom = (m[0] - n[0]) * (p[:, 1] - q[:, 1]) - (m[1] - n[1]) * (p[:, 0] - q[:, 0])
        denom[denom == 0] = 1e-8
        u = -((m[0] - n[0]) * (m[1] - p[:, 1]) - (m[1] - n[1]) * (m[0] - p[:, 0])) / denom
        valid = np.logical_and(u >= 0, u <= 1)
        intersec = p + u.reshape(-1, 1) * (q - p)
        intersec = intersec[valid]

        if len(intersec) < 2:
            continue

        w = np.sum((intersec[0] - intersec[1]) ** 2)
        if w == 0.0:
            continue

        starts.append(point_a / np.array(img.shape[:2]))
        ends.append(point_b / np.array(img.shape[:2]))
        widths_list.append(w)
        colors_list.append(np.mean(img[mask], axis=0))

        cx = np.mean(nonzero[0]) / img.shape[0]
        cy = np.mean(nonzero[1]) / img.shape[1]
        centers.append(np.array([cx, cy]))
        num_pixels_list.append(np.sum(mask))
        stddev_list.append(np.mean(np.std(img[mask], axis=0)))

    centers = np.array(centers)
    starts = np.array(starts)
    ends = np.array(ends)
    widths_arr = np.array(widths_list)
    colors_arr = np.array(colors_list, dtype=np.float32)
    num_pixels = np.array(num_pixels_list)

    N = centers.shape[0]
    rel_num_pixels = 5 * num_pixels / np.sqrt(H * W)

    # Scale locations to canvas coordinates
    location = centers.copy()
    location[:, 0] *= H
    location[:, 1] *= W
    s = starts.copy()
    s[:, 0] *= H
    s[:, 1] *= W
    e = ends.copy()
    e[:, 0] *= H
    e[:, 1] *= W

    # Make start/end relative to location
    s -= location
    e -= location

    # Control point: midpoint of s,e + small random perturbation
    c = (s + e) / 2.0 + np.stack(
        [np.random.uniform(-1, 1, N), np.random.uniform(-1, 1, N)], axis=-1
    )

    # Center the curve around its centroid
    sec_center = (s + e + c) / 3.0
    s -= sec_center
    e -= sec_center
    c -= sec_center

    # Compute width from cluster size and shape
    rel_q = np.quantile(rel_num_pixels, q=[0.3, 0.99])
    w_q = np.quantile(widths_arr, q=[0.3, 0.99])
    rel_num_pixels = np.clip(rel_num_pixels, rel_q[0], rel_q[1])
    widths_arr = np.clip(widths_arr, w_q[0], w_q[1])
    width = width_scale * rel_num_pixels.reshape(-1, 1) * widths_arr.reshape(-1, 1)

    # Scale curve control points
    s, e, c = [x * sec_scale for x in [s, e, c]]

    return (
        location.astype(np.float32),
        s.astype(np.float32),
        e.astype(np.float32),
        c.astype(np.float32),
        width.astype(np.float32),
        colors_arr,
    )


def initialize_brushstrokes(
    content_img, num_strokes, canvas_height, canvas_width,
    sec_scale, width_scale, init="sp"
):
    """
    Initialize brushstroke parameters.

    Args:
        content_img: numpy array [H, W, 3] in [0, 1] or None
        num_strokes: number of strokes
        canvas_height, canvas_width: canvas dimensions
        sec_scale: scale for curve control points
        width_scale: scale for widths
        init: 'sp' for SLIC superpixel init, 'random' for random

    Returns:
        location, s, e, c, width, color arrays
    """
    if init == "random" or content_img is None:
        color = np.random.rand(num_strokes, 3).astype(np.float32)
        width = (np.random.rand(num_strokes, 1) * width_scale).astype(np.float32)
        location = np.stack(
            [np.random.rand(num_strokes) * canvas_height,
             np.random.rand(num_strokes) * canvas_width],
            axis=-1,
        ).astype(np.float32)
        s = np.stack(
            [np.random.uniform(-1, 1, num_strokes) * canvas_height,
             np.random.uniform(-1, 1, num_strokes) * canvas_width],
            axis=-1,
        )
        e = np.stack(
            [np.random.uniform(-1, 1, num_strokes) * canvas_height,
             np.random.uniform(-1, 1, num_strokes) * canvas_width],
            axis=-1,
        )
        c = np.stack(
            [np.random.uniform(-1, 1, num_strokes) * canvas_height,
             np.random.uniform(-1, 1, num_strokes) * canvas_width],
            axis=-1,
        )
        sec_center = (s + e + c) / 3.0
        s, e, c = [x - sec_center for x in [s, e, c]]
        s, e, c = [(x * sec_scale).astype(np.float32) for x in [s, e, c]]
        return location, s, e, c, width, color

    # SLIC superpixel initialization
    segments = slic(
        content_img,
        n_segments=num_strokes,
        min_size_factor=0.02,
        max_size_factor=4.0,
        compactness=2,
        sigma=1,
        start_label=0,
    )
    return clusters_to_strokes(
        segments, content_img, canvas_height, canvas_width,
        sec_scale=sec_scale, width_scale=width_scale,
    )
