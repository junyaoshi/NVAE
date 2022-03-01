import h5py
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import io
import imageio

cam_index = 0
type = "sawyer"
file = 'berkeley_sawyer_traj985'
metadata_path = f'/home/junyao/Datasets/edward_robonet/{type}_views/meta_data.pkl'
img_path = f'/home/junyao/Datasets/edward_robonet/{type}_views/{file}.hdf5'
mask_path = f'/home/junyao/Datasets/edward_robonet/{type}_views_small/{file}_c0.hdf5'

with open(metadata_path, 'rb') as f:
    metadata = pd.read_pickle(metadata_path, compression='gzip')
file_metadata = metadata.loc[f'{file}.hdf5']
img_f = h5py.File(img_path, 'r')
mask_f = h5py.File(mask_path, 'r')

mask_key = 'mask' if type == 'sawyer' else 'masks'
mask = mask_f[mask_key][()]
print(f'shape of mask: {mask.shape}')
rgb_mask = np.expand_dims(mask, axis=-1).repeat(3, axis=-1)
print(f'shape of rgb mask: {rgb_mask.shape}')

# adapted from https://github.com/penn-pal-lab/roboaware/blob/master/robonet/robonet/datasets/util/hdf5_loader.py
cam_group = img_f['env'][f'cam{cam_index}_video']
old_dims = file_metadata['frame_dim']
length = file_metadata['img_T']
encoding = file_metadata['img_encoding']
image_format = file_metadata['image_format']

old_height, old_width = old_dims
target_height, target_width = mask.shape[1], mask.shape[2]
resize_method = cv2.INTER_CUBIC

n_load = length
start_time = 0
frames = np.zeros((n_load, old_height, old_width, 3), dtype=np.uint8)
rgb_mask_enlarged = np.zeros((n_load, old_height, old_width, 3), dtype=bool)
if encoding == 'mp4':
    buf = io.BytesIO(cam_group['frames'][:].tostring())
    img_buffer = [img for t, img in enumerate(imageio.get_reader(
        buf, format='mp4')) if start_time <= t < n_load + start_time]
elif encoding == 'jpg':
    img_buffer = [cv2.imdecode(cam_group['frame{}'.format(t)][:], cv2.IMREAD_COLOR)[:, :, ::-1]
                  for t in range(start_time, start_time + n_load)]
else:
    raise ValueError("encoding not supported")

for t, img in enumerate(img_buffer):
    frames[t] = img

for t, img_mask in enumerate(rgb_mask):
    rgb_mask_enlarged[t] = cv2.resize(
        img_mask.astype(np.uint8), (old_width, old_height), interpolation=resize_method
    ).astype(bool)

masked_frames = frames * ~rgb_mask_enlarged
print(f'shape of masked frames: {masked_frames.shape}')

size = (masked_frames.shape[2], masked_frames.shape[1])
fps = 2
frames_out = cv2.VideoWriter(
    f'../videos/{type}_views/original/{file}_frames.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size
)
masked_frames_out = cv2.VideoWriter(
    f'../videos/{type}_views/original/{file}_masked_frames.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size
)
for frame, masked_frame in tqdm(zip(frames, masked_frames), desc='processing frames'):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
    frames_out.write(frame)
    masked_frames_out.write(masked_frame)
frames_out.release()
masked_frames_out.release()
