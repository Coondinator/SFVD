from typing import Tuple
import torch
import scipy
import numpy as np
from util import open_url

def compute_fvd(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

    return float(fid)


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0) # [d]
    sigma = np.cov(feats, rowvar=False) # [d, d]

    return mu, sigma

@torch.no_grad()
def compute_our_fvd(videos_fake: np.ndarray, videos_real: np.ndarray, device: str='cuda') -> float:
    detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
    detector_kwargs = dict(rescale=False, resize=False, return_features=True)
    # Return raw features before the softmax layer.

    with open_url(detector_url, verbose=False) as f:
        detector = torch.jit.load(f).eval().to(device)

    videos_fake = torch.from_numpy(videos_fake).permute(0, 4, 1, 2, 3).to(device)
    videos_real = torch.from_numpy(videos_real).permute(0, 4, 1, 2, 3).to(device)

    feats_fake = detector(videos_fake, **detector_kwargs).cpu().numpy()
    feats_real = detector(videos_real, **detector_kwargs).cpu().numpy()

    return compute_fvd(feats_fake, feats_real)