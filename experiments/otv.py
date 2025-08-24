from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from awive.config import Config
from awive.loader import Loader, make_loader
from awive.preprocess.correct_image import Formatter

# %%
config_path = Path("data/config.yaml")

config = Config.from_fp(config_path)
# Load first image
loader: Loader = make_loader(config.dataset)
loader.has_images()
image = loader.read()
if image is None:
    raise ValueError("No image found")

# Preprocess first image
formatter = Formatter(config.dataset, config.preprocessing)
old_depths_positions = config.water_flow.profile.depths_array[:, :2]

# %%
prev_gray, depths_positions = formatter.apply(image, old_depths_positions)

# %% Check all depth positions are >0
for i, pos in enumerate(depths_positions):
    if pos[0] <= 0 or pos[1] <= 0:
        print(f"Depth position {i} is out of bounds: {pos}")
# filter out all positions with x or y <=0
depths_positions = np.array(
    [pos for pos in depths_positions if pos[0] > 0 and pos[1] > 0]
)


# %%
_, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(image)
for pos in old_depths_positions:
    axs[0].plot(pos[0], pos[1], "ro")
axs[1].imshow(prev_gray, cmap="gray")
for pos in depths_positions:
    axs[1].plot(pos[0], pos[1], "ro")
