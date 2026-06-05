from utils.visualize import save_gif
import numpy as np

frames = []

for i in range(10):
    frame = np.random.randint(
        0,
        255,
        (224,224,3),
        dtype=np.uint8
    )
    frames.append(frame)

save_gif(
    frames,
    "outputs/test.gif"
)

print("GIF Saved")