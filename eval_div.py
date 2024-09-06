import io
import os
import sys
import json
from PIL import Image
import tqdm
import torch
import decord
import numpy as np
import torchvision.transforms as T
from einops import rearrange, repeat
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MIN_THRESH = 200
VIDEO_LENGTH = 256


def read_video_from_img(path: str, height, width):
    image_files = sorted([f for f in os.listdir(path) if f.endswith('.png')])
    print('image_files:', image_files)
    images = []
    max_frames = len(image_files)
    if max_frames > VIDEO_LENGTH:
        indices = np.linspace(0, max_frames - 1, VIDEO_LENGTH, dtype=int)
        image_files = [image_files[i] for i in indices]
        for image_file in image_files:
            # 打开图像并将其转换为RGB格式
            img = Image.open(os.path.join(path, image_file)).convert('RGB')
            img_array = np.array(img)  # [None, ...] None 表示新加一个维度
            images.append(img_array)
        # 将所有图像沿着第一个维度(t)连接起来，形成 (t, h, w, c)
        # image_stack = np.concatenate(images, axis=0)
        image_stack = np.array(images).astype(np.float32)

    elif MIN_THRESH < max_frames <= VIDEO_LENGTH:
        for image_file in image_files:
            img = Image.open(os.path.join(path, image_file)).convert('RGB')
            img_array = np.array(img)
            images.append(img_array)
        image_stack = np.array(images).astype(np.float32)
        pad_length = VIDEO_LENGTH - max_frames
        last_value = image_stack[-1]
        padding = np.full((pad_length,) + image_stack.shape[1:], last_value)
        image_stack = np.concatenate((image_stack, padding), axis=0)
    else:
        print('Video is too short!')
        return None

    video = torch.from_numpy(image_stack).to(device)
    print('image_stack shape', video.shape)
    video = video.permute(0, 3, 1, 2) # [t, c, h, w]
    print('video after rearange shape', video.shape)
    transform = T.Compose([
        # T.RandomResizedCrop(size=(height, width), scale=(0.8, 1.0), ratio=(width/height, width/height), antialias=False)
        T.Resize(min(height, width), antialias=False),
        T.CenterCrop([height, width])
    ])
    video = transform(video)
    # video = video.permute(0, 2, 3, 1) # [t, h, w, c]
    video = video.permute(1, 0, 2, 3)    # [c, t, h, w]
    print('final video shape', video.shape)
    return video

def read_video_from_audio(path: str, height, width):
    mp4_file = [f for f in os.listdir(path) if f.endswith('.mp4')]
    video_path = os.path.join(path, mp4_file[0])
    vr = decord.VideoReader(video_path)
    max_frames = len(vr)
    if max_frames > VIDEO_LENGTH:
        indices = np.linspace(0, max_frames - 1, VIDEO_LENGTH, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
    elif MIN_THRESH < max_frames <= VIDEO_LENGTH:
        frame_range_indices = np.arange(max_frames)
        frames = vr.get_batch(frame_range_indices).asnumpy()
        pad_length = VIDEO_LENGTH - max_frames
        last_value = frames[-1]
        padding = np.full((pad_length,) + frames.shape[1:], last_value)
        frames = np.concatenate((frames, padding), axis=0)
    else:
        print('Video is too short!')
        return None

    frames = frames.astype(np.float32)
    video = torch.from_numpy(frames).to(device)
    print('image_stack shape', video.shape)
    video = video.permute(0, 3, 1, 2)  # [t, c, h, w]
    video = torch.from_numpy(rearrange(frames, "f h w c -> f c h w"))
    transform = T.Compose([
        # T.RandomResizedCrop(size=(height, width), scale=(0.8, 1.0), ratio=(width/height, width/height), antialias=False)
        T.Resize(min(height, width), antialias=False),
        T.CenterCrop([height, width])
    ])
    video = transform(video)
    video = video.permute(1, 0, 2, 3)

    return video

def main():
    root_path = './data/test/'
    detector_path = './i3d_torchscript.pt'
    detector_kwargs = dict(rescale=False, resize=False, return_features=True)
    # Load all tensors to the original device
    detector = torch.jit.load(detector_path).eval().to(device)
    detector = torch.nn.DataParallel(detector)
    # TODO: read all video from audio and concatenate them on the first dimension (B, C, T, H, W), here we only read one video
    video = read_video_from_img(root_path, 256, 256)
    video = video.unsqueeze(0)
    feats = detector(video, **detector_kwargs)  # [b, c, t, h, w] -> # [b, 400, 2, 2]
    print('feats shape:', feats.shape)
    mean = feats.mean()
    std = feats.std()
    print('mean:', mean)
    print('std:', std)

if __name__ == '__main__':
    main()