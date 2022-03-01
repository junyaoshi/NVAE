import h5py
import numpy as np
import cv2
from tqdm import tqdm

type = "sawyer"
file = 'berkeley_sawyer_traj9'
path = f'/home/junyao/Datasets/edward_robonet/{type}_views/{file}.hdf5'

f = h5py.File(path, 'r')

frames_key = 'frames' if type == 'sawyer' else 'observations'
frames = f[frames_key][()]
mask_key = 'mask' if type == 'sawyer' else 'masks'
mask = f[mask_key][()]

print(f'shape of frames: {frames.shape}')
print(f'shape of mask: {mask.shape}')

rgb_mask = np.expand_dims(mask, axis=-1).repeat(3, axis=-1)
print(f'shape of rgb mask: {rgb_mask.shape}')

masked_frames = frames * ~rgb_mask
print(f'shape of masked frames: {masked_frames.shape}')

size = (masked_frames.shape[2], masked_frames.shape[1])
fps = 2
frames_out = cv2.VideoWriter(
    f'../videos/{type}_views/{file}_frames.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size
)
masked_frames_out = cv2.VideoWriter(
    f'../videos/{type}_views/{file}_masked_frames.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size
)
for frame, masked_frame in tqdm(zip(frames, masked_frames), desc='processing frames'):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
    frames_out.write(frame)
    masked_frames_out.write(masked_frame)
frames_out.release()
masked_frames_out.release()
