"""All Dynamic LeJEPA model components.

Phase 1/2 (nuScenes):  MaxEntropyEncoder | EgoMotionPredictor (Thm IV.7)
                       | PhysicsInformedDepthDecoder (Thm IV.10, Ex IV.12)
Phase 3 (MimicGen):    ViTEncoder (Thm IV.4) | ActionPredictor (Thm IV.13)
                       | ProprioDecoder (Thm IV.10 analog)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================ Phase 1/2: nuScenes ============================

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop,
                                           batch_first=True)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class MaxEntropyEncoder(nn.Module):
    """f_theta (Theorem IV.1): NO constraints besides SIGReg. z in R^K,
    decomposed [z_ego (K1) | z_static (K2)] per Theorem IV.7."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192,
                 depth=6, num_heads=6, latent_dim=256, drop_rate=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, drop=drop_rate)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, latent_dim, bias=False))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.proj_head[-1].weight, gain=1.0)

    def forward(self, x, return_features=False):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        z = self.proj_head(x[:, 0])
        return (z, x[:, 1:]) if return_features else z


class EgoMotionPredictor(nn.Module):
    """g_phi (Theorem IV.7): z_ego_{t+1} = g(z_ego_t, e_t);
    z_static_{t+1} = z_static_t (static stability, Eq. 19)."""

    def __init__(self, K_ego=64, K_static=192, ego_motion_dim=6,
                 hidden_dim=256, num_layers=3):
        super().__init__()
        self.K_ego, self.K_static = K_ego, K_static
        layers, in_dim = [], K_ego + ego_motion_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim,
                                    hidden_dim if i < num_layers - 1 else K_ego))
            if i < num_layers - 1:
                layers += [nn.GELU(), nn.LayerNorm(hidden_dim)]
        self.ego_predictor = nn.Sequential(*layers)

    def decompose(self, z):
        return z[:, :self.K_ego], z[:, self.K_ego:]

    def forward(self, z_t, ego_motion):
        z_ego, z_static = self.decompose(z_t)
        z_ego_hat = self.ego_predictor(torch.cat([z_ego, ego_motion], dim=1))
        z_hat = torch.cat([z_ego_hat, z_static], dim=1)          # static = identity
        return z_hat, z_ego, z_ego_hat, z_static


class PhysicsInformedDepthDecoder(nn.Module):
    """h_psi (Theorem IV.10, Example IV.12): LiDAR supervision + edge-aware
    smoothness. Physics lives HERE — z_static is detached before decoding."""

    def __init__(self, static_dim=192, hidden_dim=256, depth_size=56,
                 num_upsample=3, edge_thresh=0.1):
        super().__init__()
        self.depth_size, self.edge_thresh = depth_size, edge_thresh
        self.fc = nn.Sequential(nn.Linear(static_dim, hidden_dim), nn.GELU(),
                                nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        init_size = depth_size // (2 ** num_upsample)
        self.spatial_broadcast = nn.Linear(hidden_dim,
                                           init_size * init_size * hidden_dim)
        self.reshape_size = init_size
        up, ch = [], hidden_dim
        for _ in range(num_upsample):
            up += [nn.ConvTranspose2d(ch, ch // 2, 4, 2, 1),
                   nn.BatchNorm2d(ch // 2), nn.GELU()]
            ch //= 2
        up.append(nn.Conv2d(ch, 1, 3, padding=1))
        self.upsample_path = nn.Sequential(*up)

    def forward(self, z_static, img_t=None, depth_sparse=None):
        B = z_static.shape[0]
        feat = self.fc(z_static)
        feat = self.spatial_broadcast(feat).view(B, -1,
                                                 self.reshape_size, self.reshape_size)
        depth_hat = F.relu(self.upsample_path(feat)) + 0.1
        physics_loss = torch.tensor(0.0, device=z_static.device)
        if depth_sparse is not None:                              # LiDAR physics
            ds = depth_sparse.unsqueeze(1)
            mask = (ds > 0.1).float()
            n = mask.sum().clamp(min=1.0)
            physics_loss = physics_loss + ((depth_hat - ds).pow(2) * mask).sum() / n
        if img_t is not None:                                     # Eq. (31)
            physics_loss = physics_loss + 0.1 * self.edge_aware_smoothness(
                depth_hat, img_t)
        return depth_hat, physics_loss

    def edge_aware_smoothness(self, depth, image):
        D = depth.shape[-1]
        img = F.interpolate(image, size=(D, D), mode="bilinear",
                            align_corners=False).mean(dim=1, keepdim=True)
        gx = img[:, :, :, 1:] - img[:, :, :, :-1]
        gy = img[:, :, 1:, :] - img[:, :, :-1, :]
        d = depth.squeeze(1)
        lap_x = d[:, :, 2:] - 2 * d[:, :, 1:-1] + d[:, :, :-2]
        lap_y = d[:, 2:, :] - 2 * d[:, 1:-1, :] + d[:, :-2, :]
        gx = (gx[:, :, :, :-1] + gx[:, :, :, 1:]) / 2
        gy = (gy[:, :, :-1, :] + gy[:, :, 1:, :]) / 2
        wx = torch.exp(-gx.abs() / self.edge_thresh)
        wy = torch.exp(-gy.abs() / self.edge_thresh)
        return (wx * lap_x.unsqueeze(1) ** 2).mean() \
             + (wy * lap_y.unsqueeze(1) ** 2).mean()


def encoder_physics_loss(patch_tokens: torch.Tensor) -> torch.Tensor:
    """The INVALID placement (Cor IV.11): the same smoothness prior applied
    to the ENCODER's own patch features — the baseline that prior
    physics-informed JEPA work effectively used, and which the paper proves
    is destructive."""
    B, N, D = patch_tokens.shape
    H = W = int(N ** 0.5)
    feat = patch_tokens.transpose(1, 2).reshape(B, D, H, W)
    sx = (feat[:, :, :, 1:] - feat[:, :, :, :-1]).pow(2).mean()
    sy = (feat[:, :, 1:, :] - feat[:, :, :-1, :]).pow(2).mean()
    return sx + sy


# ============================ Phase 3: MimicGen =============================

class ViTEncoder(nn.Module):
    """z_t = f_theta(img_t, proprio_t) — Theorem IV.4 encoder (no constraints).
    6-layer ViT, patch 14 on 84x84 + 32-dim proprio -> z in R^256."""

    def __init__(self, img_size=84, patch=14, dim=192, depth=6, heads=3,
                 prop_dim=32, latent=256):
        super().__init__()
        n_p = (img_size // patch) ** 2
        self.patch = nn.Conv2d(3, dim, patch, patch)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, n_p + 1, dim) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, mlp_ratio=2.0)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim + prop_dim, latent), nn.GELU(),
                                  nn.LayerNorm(latent), nn.Linear(latent, latent))

    def forward(self, img, prop):
        B = img.shape[0]
        x = self.patch(img).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1) + self.pos
        for b in self.blocks:
            x = b(x)
        return self.head(torch.cat([self.norm(x[:, 0]), prop], dim=1))


class ActionPredictor(nn.Module):
    """z_hat_{t+1} = g_phi(z_t, a_t) — dynamics belong HERE (Thms IV.4b, IV.13)."""

    def __init__(self, latent=256, action_dim=14, hidden=256, layers=3):
        super().__init__()
        mods, d = [], latent + action_dim
        for _ in range(layers - 1):
            mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU()]
            d = hidden
        mods += [nn.Linear(d, latent)]
        self.net = nn.Sequential(*mods)

    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=1))


class ProprioDecoder(nn.Module):
    """q_t = h_psi(z_t) — physics belongs HERE (Theorem IV.10)."""

    def __init__(self, latent=256, prop_dim=32, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(),
                                 nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                                 nn.GELU(), nn.LayerNorm(hidden),
                                 nn.Linear(hidden, prop_dim))

    def forward(self, z):
        return self.net(z)
