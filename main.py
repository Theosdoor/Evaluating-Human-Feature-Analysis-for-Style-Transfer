# %%
# blabla

# %%
import os
import sys
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    # install deps from pyproject.toml
    url = "https://raw.githubusercontent.com/Theosdoor/ACV_cswk/refs/heads/main/pyproject.toml?token=GHSAT0AAAAAADMXCD2D27EMDM3ERVBN5GE62MOA47A"
    !wget -O pyproject.toml {url}
    !uv pip install --system -r pyproject.toml

from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO # https://github.com/ultralytics/ultralytics


# %%
if __name__ == "__main__":
    model = YOLO('models/yolov8n.pt')


