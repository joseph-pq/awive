from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import awive.preprocess.imageprep as ip
from awive.config import Config
from awive.loader import Loader, make_loader
from awive.preprocess.correct_image import Formatter

# %%

config_path = Path("data/config.yaml")

config = Config.from_fp(config_path)

loader: Loader = make_loader(config.dataset)
loader.has_images()
image = loader.read()
if image is None:
    raise ValueError("No image found")

# %% Preprocess first image
formatter = Formatter(config.dataset, config.preprocessing)
pos = np.array([[1016, 757], [1372, 1340], [3667, 512]])
# prev_gray = formatter.apply(image)
prev_gray = formatter.apply_distortion_correction(image)

# crop
x_min, y_min = np.min(formatter.dataset.gcp.pixels_coordinates, axis=0)
x_max, y_max = np.max(formatter.dataset.gcp.pixels_coordinates, axis=0)
new_pos = pos - np.array([x_min, y_min])
# orthorectify
m, _ = formatter._or_params
ones = np.ones((new_pos.shape[0], 1), dtype=np.float32)
hom = np.hstack([new_pos, ones])  # (N, 3)
transformed = hom @ m.T
new_pos = transformed[:, :2] / transformed[:, 2][:, None]
# precrop
y0, _ = formatter._pre_slice[0].start, formatter._pre_slice[0].stop
x0, _ = formatter._pre_slice[1].start, formatter._pre_slice[1].stop
new_pos -= np.array([x0, y0], dtype=np.float32)

prev_gray = formatter.apply_roi_extraction(prev_gray)

# rotate
ones = np.ones((new_pos.shape[0], 1), dtype=np.float32)
hom = np.hstack([new_pos, ones])  # (N, 3)
new_pos = hom @ formatter._rotation_matrix.T  # (N, 2)
# crop
y0, _ = formatter._slice[0].start, formatter._slice[0].stop
x0, _ = formatter._slice[1].start, formatter._slice[1].stop
new_pos -= np.array([x0, y0], dtype=np.float32)

prev_gray = formatter.apply_resolution(prev_gray)
if formatter.preprocessing.resolution < 1:
    new_pos *= formatter.preprocessing.resolution

# points that are out of bounds after preprocessing set to -1
for i in range(new_pos.shape[0]):
    if not (0 <= new_pos[i][0] < prev_gray.shape[1]) or not (
        0 <= new_pos[i][1] < prev_gray.shape[0]
    ):
        new_pos[i] = -1
print(new_pos)

# %%
formatter = Formatter(config.dataset, config.preprocessing)
pos = np.array([[1016, 757], [1372, 1340], [3667, 512]])
prev_gray, new_pos = formatter.apply(image, pos)

# %%
_, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(image)
axs[0].plot(pos[0][0], pos[0][1], "ro")
axs[0].plot(pos[1][0], pos[1][1], "go")
axs[1].imshow(prev_gray, cmap="gray")
axs[1].plot(new_pos[0][0], new_pos[0][1], "ro")
axs[1].plot(new_pos[1][0], new_pos[1][1], "go")
