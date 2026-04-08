import torch
import torch.nn.functional as F

from vgg import VGG19


def gram_matrix(x):
    """Compute Gram matrix for style loss."""
    B, C, H, W = x.size()
    features = x.view(B * C, H * W)
    G = torch.mm(features, features.t())
    return G.div(B * C * H * W)


class StyleTransferLosses(VGG19):
    """
    VGG19-based content and style loss computation.

    Pre-computes target content features and style Gram matrices,
    then computes losses for a given input image on forward pass.
    """

    def __init__(
        self,
        weight_file,
        content_img,
        style_img,
        content_layers,
        style_layers,
        scale_by_y=False,
        content_weights=None,
        style_weights=None,
    ):
        super().__init__(weight_file)

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.scale_by_y = scale_by_y

        content_weights = content_weights or [1.0] * len(content_layers)
        style_weights = style_weights or [1.0] * len(style_layers)
        self.content_weight_map = {}
        self.style_weight_map = {}

        self.content_features = {}
        self.style_features = {}
        self.feature_weights = {}

        # Pre-compute content and style targets
        i, j = 0, 0
        self.to(content_img.device)
        with torch.no_grad():
            cf = content_img
            sf = style_img
            for name, layer in self.named_children():
                cf = layer(cf)
                sf = layer(sf)
                if name in content_layers:
                    self.content_features[name] = cf.detach()
                    self.content_weight_map[name] = content_weights[i]
                    if scale_by_y:
                        self.feature_weights[name] = torch.minimum(
                            cf, torch.sigmoid(cf)
                        ).detach()
                    i += 1
                if name in style_layers:
                    self.style_features[name] = gram_matrix(sf).detach()
                    self.style_weight_map[name] = style_weights[j]
                    j += 1

    def forward(self, x):
        content_loss = 0.0
        style_loss = 0.0
        features = x
        for name, layer in self.named_children():
            features = layer(features)
            if name in self.content_layers:
                diff = features - self.content_features[name]
                if self.scale_by_y:
                    diff = diff * self.feature_weights[name]
                content_loss += torch.mean(diff ** 2) * self.content_weight_map[name]
            if name in self.style_layers:
                G = gram_matrix(features)
                style_loss += (
                    F.mse_loss(G, self.style_features[name], reduction="sum")
                    * self.style_weight_map[name]
                )
        return content_loss, style_loss


def total_variation_loss(locations, curve_s, curve_e, K=10):
    """
    Brushstroke total variation loss.

    Encourages neighboring brushstrokes (by location) to have similar orientations.
    Uses KNN on stroke locations to find neighbors.
    """
    se_vec = curve_e - curve_s  # [N, 2] direction vectors

    # KNN using cdist
    dists = torch.cdist(locations, locations)  # [N, N]
    _, nn_idx = torch.topk(dists, K + 1, dim=1, largest=False)
    nn_idx = nn_idx[:, 1:]  # exclude self, [N, K]

    # Gather neighbor direction vectors
    neighbor_vecs = se_vec[nn_idx.flatten()].view(se_vec.shape[0], K, 2)  # [N, K, 2]

    # Compare projections (squared terms of direction)
    def projection(z):
        x, y = z[..., 0], z[..., 1]
        return torch.stack([x ** 2, y ** 2, x * y], dim=-1)

    proj_self = projection(se_vec).unsqueeze(1)       # [N, 1, 3]
    proj_neighbors = projection(neighbor_vecs)          # [N, K, 3]

    return torch.mean(torch.sum((proj_neighbors - proj_self) ** 2, dim=-1))


def curvature_loss(curve_s, curve_e, curve_c):
    """
    Penalizes excessive curvature of Bezier curves.
    Measures how far the control point deviates from the midpoint of s-e.
    """
    v1 = curve_s - curve_c
    v2 = curve_e - curve_c
    dist_se = torch.norm(curve_e - curve_s, dim=-1) + 1e-6
    return torch.mean(torch.norm(v1 + v2, dim=-1) / dist_se)

