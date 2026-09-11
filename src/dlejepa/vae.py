python -c "
import torch, math
from dlejepa.vae import (VAEEncoder, VAEEncoderNuscenes, VAEDeterministic,
                         vae_kl_loss, vae_plus_cov_loss, vae_latent_stats,
                         build_vae_models, build_vae_models_nuscenes)
from dlejepa.models import VAEEncoder as lazy
assert lazy is VAEEncoder, 'lazy re-export broken'

# shapes + shim determinism (Phase 3)
enc = VAEEncoder(84, 14, 192, 6, 3, 32, 256)
img, prop = torch.randn(2, 3, 84, 84), torch.randn(2, 32)
mu, lv = enc.stats(img, prop)
assert mu.shape == (2, 256) and lv.shape == (2, 256)
shim = VAEDeterministic(enc); shim.eval()
assert torch.allclose(shim(img, prop), shim(img, prop)) and torch.allclose(shim(img, prop), mu)

# Phase 2 tuple protocol (evaluate.py dispatch)
enc2 = VAEEncoderNuscenes(224, 16, 3, 192, 6, 6, 256)
x = torch.randn(2, 3, 224, 224)
assert len(enc2(x)) == 2 and len(enc2(x, return_features=True)) == 3

# KL closed form vs Monte-Carlo (fixed seed)
g = torch.Generator().manual_seed(0)
mu, lv = torch.randn(8, 256, generator=g) * 0.4, torch.randn(8, 256, generator=g) * 0.1
eps = torch.randn(400000, 8, 256, generator=g)
z = mu + torch.exp(0.5 * lv) * eps
mc = (-0.5 * (eps.pow(2) + lv - (mu + torch.exp(0.5*lv)*eps).pow(2))).sum(-1).mean()
assert abs(vae_kl_loss(mu, lv).item() - mc.item()) / mc.item() < 0.01

# noise-floor masking: rank-1 mu signal + broad posterior
#   -> d_eff(mu) low, signal fraction ~0, yet SAMPLED latents look dispersed
N = 20000
v = torch.randn(256, generator=g); a = torch.randn(N, 1, generator=g)
mu_cc = (0.05 * a * v[None, :])                    # rank-1 signal (correlated collapse)
lv_cc = torch.full((N, 256), 2 * math.log(4.0))    # sigma = 4 -> broad noise floor
stats = vae_latent_stats(lambda c: (mu_cc[c], lv_cc[c]), torch.arange(N), 'cpu')
assert stats['mu_d_eff'] < 0.1 and stats['signal_fraction_mean'] < 1e-3
zs = mu_cc + torch.exp(0.5 * lv_cc) * torch.randn_like(mu_cc)
assert zs.var(0).mean() > 1.0, 'sampled latents would look healthy'
print('vae.py OK — all contracts + masking pre-check pass')
"
