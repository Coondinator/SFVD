import io
import sys
import json
import tqdm
import torch
import decord
import scipy
import numpy as np
from typing import Tuple
import torchvision.transforms as T
from einops import rearrange, repeat
from utils import get_fvd_feats, frechet_distance, load_i3d_pretrained


def compute_fvd(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    print('feats_fake.shape', feats_fake.shape)
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)

    m = np.square(mu_gen - mu_real).sum()
    print(sigma_gen)
    print(sigma_real)
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member

    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

    return float(fid)


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0) # [d]
    print('mu', mu.size)
    sigma = np.cov(feats, rowvar=False) # [d, d]
    print('sigma', sigma.size)

    return mu, sigma

@torch.no_grad()
def compute_fvd_torch(videos_fake: np.ndarray, videos_real: np.ndarray, device: str='cuda') -> float:
    # detector_path = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
    detector_path = './i3d_torchscript.pt'
    detector_kwargs = dict(rescale=False, resize=False, return_features=True)
    # Return raw features before the softmax layer.
    # with open(detector_path, 'rb') as f:
    #     buffer = io.BytesIO(f.read())

    # Load all tensors to the original device
    detector = torch.jit.load(detector_path).eval().to(device)
    detector = torch.nn.DataParallel(detector)
    # detector = torch.jit.load(etector_path).eval().to(device)

    videos_fake = torch.from_numpy(videos_fake).permute(0, 4, 1, 2, 3).to(device)
    videos_real = torch.from_numpy(videos_real).permute(0, 4, 1, 2, 3).to(device)

    feats_fake = detector(videos_fake, **detector_kwargs).cpu().numpy()
    print('feats_fake.shape', feats_fake.shape)
    feats_real = detector(videos_real, **detector_kwargs).cpu().numpy()

    return compute_fvd(feats_fake, feats_real)


def padding_video(video: np.ndarray):
    video = video.permute(1, 2, 3, 0)
    print('before pad video.shape', video.shape)    # 目标形状

    padded_data = torch.nn.functional.pad(video, pad=(0, 4), value=0)
    padded_data = padded_data.permute(3, 0, 1, 2)
    return padded_data


def read_video(path: str, height, width) -> np.ndarray:
    video_path = path
    vr = decord.VideoReader(video_path)

    max_frames = len(vr)

    # start = random.randint(0, len(frame_range) - max_frames)
    frame_range_indices = np.arange(max_frames)
    frames = vr.get_batch(frame_range_indices).asnumpy()

    video = torch.from_numpy(rearrange(frames, "f h w c -> f c h w"))
    # print('video shape', video.shape)

    transform = T.Compose([
        # T.RandomResizedCrop(size=(height, width), scale=(0.8, 1.0), ratio=(width/height, width/height), antialias=False)
        T.Resize(min(height, width), antialias=False),
        T.CenterCrop([height, width])
    ])
    #frames = torch.from_numpy(frames)
    video = transform(video)
    print('video shape', video.shape)
    video = video.permute(0, 2, 3, 1)
    print('video shape', video.shape)
    video = padding_video(video)
    print('video shape', video.shape)
    return video


@torch.no_grad()
def calculate_fvd(videos1, videos2, device):
    # support grayscale input, if grayscale -> channel*3
    # BTCHW -> BCTHW
    # videos -> [batch_size, channel, timestamps, h, w]
    videos1 = torch.from_numpy(videos1).permute(0, 4, 1, 2, 3).to(device)
    videos2 = torch.from_numpy(videos2).permute(0, 4, 1, 2, 3).to(device)
    print("calculate_fvd...")

    # videos [batch_size, timestamps, channel, h, w]

    assert videos1.shape == videos2.shape

    # i3d = load_i3d_pretrained(device=device)
    detector_path = './i3d_torchscript.pt'
    detector_kwargs = dict(rescale=False, resize=False, return_features=True)


    # Load all tensors to the original device
    detector = torch.jit.load(detector_path).eval().to(device)
    detector = torch.nn.DataParallel(detector)
    fvd_results = []


    # for calculate FVD, each clip_timestamp must >= 10
    true_feat = get_fvd_feats(videos1, i3d=detector, device=device)
    pred_feat = get_fvd_feats(videos2, i3d=detector, device=device)
    fvd_results = frechet_distance(true_feat, pred_feat)
    print("fvd_results", fvd_results)

    '''
    for clip_timestamp in tqdm(range(10, videos1.shape[-3] + 1)):
        # get a video clip
        # videos_clip [batch_size, channel, timestamps[:clip], h, w]
        videos_clip1 = videos1[:, :, : clip_timestamp]
        videos_clip2 = videos2[:, :, : clip_timestamp]

        # get FVD features
        feats1 = get_fvd_feats(videos_clip1, i3d=detector, device=device)
        feats2 = get_fvd_feats(videos_clip2, i3d=detector, device=device)

        # calculate FVD when timestamps[:clip]
        fvd_results[clip_timestamp] = frechet_distance(feats1, feats2)

    result = {
        "value": fvd_results,
        "video_setting": videos1.shape,
        "video_setting_name": "batch_size, channel, time, heigth, width",
    }
    '''

    return fvd_results


if __name__ == '__main__':
    true_video = read_video('./out.mp4', width=224, height=224)
    true_video = np.expand_dims(true_video, axis=0).astype(np.float32)
    fake_video = read_video('./copy.mp4', width=224, height=224)
    fake_video = np.expand_dims(fake_video, axis=0).astype(np.float32)
    # compute_fvd_torch(videos_fake=fake_video, videos_real=true_video, device='cuda')
    result = calculate_fvd(videos1=true_video, videos2=fake_video, device='cuda')
    print(json.dumps(result, indent=4))
    print("Square Root of A:")
    # print(sqrt_A)