import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import numpy as np

from utils import initialize_brushstrokes


def sample_quadratic_bezier_curve(s, c, e, num_points=10):
    """
    Sample points along quadratic Bezier curves.

    Args:
        s: Start points [N, 2]
        c: Control points [N, 2]
        e: End points [N, 2]
        num_points: Number of samples per curve

    Returns:
        Points on curves [N, num_points, 2]
    """
    N = s.shape[0]
    t = torch.linspace(0.0, 1.0, num_points, device=s.device)  # [num_points]
    t = t.unsqueeze(0).expand(N, -1)  # [N, num_points]

    s_x, s_y = s[:, 0:1], s[:, 1:2]
    c_x, c_y = c[:, 0:1], c[:, 1:2]
    e_x, e_y = e[:, 0:1], e[:, 1:2]

    # Quadratic Bezier: B(t) = (1-t)^2 * s + 2(1-t)t * c + t^2 * e
    # Equivalent form: c + (1-t)^2 * (s - c) + t^2 * (e - c)
    x = c_x + (1.0 - t) ** 2 * (s_x - c_x) + t ** 2 * (e_x - c_x)
    y = c_y + (1.0 - t) ** 2 * (s_y - c_y) + t ** 2 * (e_y - c_y)
    return torch.stack([x, y], dim=-1)  # [N, num_points, 2]


def _knn_on_grid(locations, grid_points, k):
    """
    Find k nearest brushstroke indices for each grid point using cdist.

    Args:
        locations: [N, 2] brushstroke center locations
        grid_points: [M, 2] grid query points
        k: number of neighbors

    Returns:
        indices: [M, k] indices into locations
    """
    # Compute pairwise distances [M, N]
    dists = torch.cdist(grid_points, locations)
    # Get k nearest
    _, indices = torch.topk(dists, k, dim=1, largest=False)
    return indices  # [M, k]


@torch.jit.script
def _render_strokes(
    curve_points: torch.Tensor,
    locations: torch.Tensor,
    colors: torch.Tensor,
    widths: torch.Tensor,
    indices: torch.Tensor,
    H: int,
    W: int,
    K: int,
    canvas_color: float,
) -> torch.Tensor:
    """
    Core rendering: given KNN indices, render brushstrokes onto canvas.
    Implements Algorithm 1 from the paper.

    Args:
        curve_points: [N, S, 2] sampled points on each Bezier curve
        locations: [N, 2] brushstroke locations (clamped)
        colors: [N, 3] brushstroke colors (clamped 0-1)
        widths: [N, 1] brushstroke widths (exp-space)
        indices: [H, W, K] nearest brush indices per pixel
        H, W: canvas dimensions
        K: strokes per pixel
        canvas_color: background intensity

    Returns:
        canvas: [H, W, 3]
    """
    device = curve_points.device
    N, S, _ = curve_points.shape

    # Gather curve points, colors, widths for nearest brushstrokes at each pixel
    flat_idx = indices.flatten()  # [H*W*K]
    canvas_curves = curve_points[flat_idx].view(H, W, K, S, 2)
    canvas_colors = colors[flat_idx].view(H, W, K, 3)
    canvas_widths = widths[flat_idx].view(H, W, K, 1)

    # Full resolution pixel grid
    t_H = torch.linspace(0.0, float(H), H, device=device)
    t_W = torch.linspace(0.0, float(W), W, device=device)
    P_y, P_x = torch.meshgrid(t_H, t_W, indexing="ij")
    P_full = torch.stack([P_x, P_y], dim=-1)  # [H, W, 2]

    # Compute distance from each pixel to nearest point on each curve segment
    # Line segments: a -> b for consecutive sampled points
    seg_a = canvas_curves[:, :, :, :-1, :]  # [H, W, K, S-1, 2]
    seg_b = canvas_curves[:, :, :, 1:, :]   # [H, W, K, S-1, 2]
    seg_ba = seg_b - seg_a                   # [H, W, K, S-1, 2]

    # Vector from segment start to pixel
    p_a = P_full[:, :, None, None, :] - seg_a  # [H, W, K, S-1, 2]

    # Project pixel onto each line segment, clamp to [0, 1]
    t = torch.sum(seg_ba * p_a, dim=-1) / (torch.sum(seg_ba ** 2, dim=-1) + 1e-8)
    t = torch.clamp(t, 0.0, 1.0)

    # Closest point on each segment
    closest = seg_a + t.unsqueeze(-1) * seg_ba  # [H, W, K, S-1, 2]

    # Squared distance from pixel to closest point on each segment
    dist_sq = torch.sum((P_full[:, :, None, None, :] - closest) ** 2, dim=-1)  # [H, W, K, S-1]

    # Min distance across segments for each stroke
    D_per_stroke = torch.amin(dist_sq, dim=-1)  # [H, W, K]

    # Min distance across all K nearest strokes
    D = torch.amin(D_per_stroke, dim=-1)  # [H, W]

    # Soft ranking: which stroke "wins" at each pixel (softmax over inverse distance)
    ranking = F.softmax(100000.0 * (1.0 / (1e-8 + D_per_stroke)), dim=-1)  # [H, W, K]

    # Weighted color and width
    I_colors = torch.einsum("hwkc,hwk->hwc", canvas_colors, ranking)  # [H, W, 3]
    bs = torch.einsum("hwkc,hwk->hwc", canvas_widths, ranking)        # [H, W, 1]

    # Brush alpha mask (sigmoid of width - distance)
    bs_mask = torch.sigmoid(bs - D.unsqueeze(-1))  # [H, W, 1]

    # Composite onto canvas
    canvas = torch.ones_like(I_colors) * canvas_color
    I = I_colors * bs_mask + (1.0 - bs_mask) * canvas
    return I  # [H, W, 3]


def stroke_renderer(curve_points, locations, colors, widths, H, W, K, canvas_color):
    """
    Full differentiable brushstroke renderer (Algorithm 1).

    Uses a coarse grid + KNN to efficiently find relevant strokes per pixel,
    then renders with distance-based alpha compositing.

    Args:
        curve_points: [N, S, 2] points on Bezier curves
        locations: [N, 2] stroke center locations
        colors: [N, 3] stroke colors
        widths: [N, 1] stroke widths (log-space, will be exp'd)
        H, W: canvas height and width
        K: number of nearest strokes to consider per pixel
        canvas_color: background color value

    Returns:
        canvas: [H, W, 3]
    """
    colors = torch.clamp(colors, 0.0, 1.0)
    loc_x = torch.clamp(locations[:, 0:1], 0, W)
    loc_y = torch.clamp(locations[:, 1:2], 0, H)
    locations = torch.cat([loc_x, loc_y], dim=1)
    widths = torch.exp(widths)

    device = curve_points.device

    # Coarse grid for KNN (every 5 pixels)
    t_H = torch.linspace(0.0, float(H), max(int(H // 5), 1), device=device)
    t_W = torch.linspace(0.0, float(W), max(int(W // 5), 1), device=device)
    P_y, P_x = torch.meshgrid(t_H, t_W, indexing="ij")
    coarse_grid = torch.stack([P_x, P_y], dim=-1).view(-1, 2)  # [G, 2]

    # Find K nearest strokes for each coarse grid cell
    K_actual = min(K, locations.shape[0])
    indices = _knn_on_grid(locations, coarse_grid, K_actual)  # [G, K]

    # Reshape to coarse grid and upscale to full resolution
    cH, cW = len(t_H), len(t_W)
    indices = indices.view(cH, cW, K_actual).permute(2, 0, 1)  # [K, cH, cW]
    indices = TF.resize(
        indices, size=[H, W], interpolation=TF.InterpolationMode.NEAREST
    )  # [K, H, W]
    indices = indices.permute(1, 2, 0)  # [H, W, K]

    return _render_strokes(
        curve_points, locations, colors, widths, indices, H, W, K_actual, canvas_color
    )


class BrushStrokeRenderer(torch.nn.Module):
    """
    Differentiable brushstroke renderer module.

    Parameterizes N brushstrokes, each defined by:
    - location: [N, 2] center position on canvas
    - curve_s, curve_e, curve_c: [N, 2] start/end/control points (relative to location)
    - color: [N, 3] RGB color
    - width: [N, 1] stroke width (log-space)
    """

    def __init__(
        self,
        canvas_height,
        canvas_width,
        num_strokes=5000,
        samples_per_curve=10,
        strokes_per_pixel=20,
        canvas_color="gray",
        length_scale=1.1,
        width_scale=0.1,
        content_img=None,
    ):
        super().__init__()

        if canvas_color == "gray":
            self.canvas_color = 0.5
        elif canvas_color == "black":
            self.canvas_color = 0.0
        elif canvas_color == "noise":
            self.canvas_color = 0.5  # fallback; noise handled differently in forward
        else:
            self.canvas_color = 1.0

        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.num_strokes = num_strokes
        self.samples_per_curve = samples_per_curve
        self.strokes_per_pixel = strokes_per_pixel

        # Initialize brushstrokes from content image (SLIC superpixels) or randomly
        if content_img is not None:
            location, s, e, c, width, color = initialize_brushstrokes(
                content_img, num_strokes, canvas_height, canvas_width,
                length_scale, width_scale, init="sp"
            )
        else:
            location, s, e, c, width, color = initialize_brushstrokes(
                None, num_strokes, canvas_height, canvas_width,
                length_scale, width_scale, init="random"
            )

        # Flip x,y -> y,x convention
        location = location[..., ::-1].copy()
        s = s[..., ::-1].copy()
        e = e[..., ::-1].copy()
        c = c[..., ::-1].copy()

        self.curve_s = torch.nn.Parameter(torch.from_numpy(s.astype(np.float32)))
        self.curve_e = torch.nn.Parameter(torch.from_numpy(e.astype(np.float32)))
        self.curve_c = torch.nn.Parameter(torch.from_numpy(c.astype(np.float32)))
        self.color = torch.nn.Parameter(torch.from_numpy(color.astype(np.float32)))
        self.location = torch.nn.Parameter(torch.from_numpy(location.astype(np.float32)))
        self.width = torch.nn.Parameter(
            torch.from_numpy(np.log(np.maximum(width, 1e-3)).astype(np.float32))
        )

    def forward(self):
        curve_points = sample_quadratic_bezier_curve(
            s=self.curve_s + self.location,
            e=self.curve_e + self.location,
            c=self.curve_c + self.location,
            num_points=self.samples_per_curve,
        )
        canvas = stroke_renderer(
            curve_points,
            self.location,
            self.color,
            self.width,
            self.canvas_height,
            self.canvas_width,
            self.strokes_per_pixel,
            self.canvas_color,
        )
        return canvas  # [H, W, 3]
