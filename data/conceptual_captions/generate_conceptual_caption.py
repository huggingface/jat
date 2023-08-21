import io
import urllib
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import PIL.Image
from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent


USER_AGENT = get_datasets_user_agent()


def fetch_single_image(image_url, timeout=1, retries=0):
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
            break
        except Exception:
            image = None
    return image


def resize_single_image(image: PIL.Image):
    # Resize so that the bigger size is at most 352
    width, height = image.size
    print("Image size", image.size, "patches", f"{width // 16}x{height // 16} = {width // 16*height // 16}")
    if width > height:
        new_width = 352
        new_height = int(height * 352 / width)
    else:
        new_height = 352
        new_width = int(width * 352 / height)
    image = image.resize((new_width, new_height), PIL.Image.BILINEAR)
    image = np.array(image, dtype=np.uint8)
    return image


def fetch_images(batch, timeout=1, retries=0):
    fetch_single_image_with_args = partial(fetch_single_image, timeout=timeout, retries=retries)
    with ThreadPoolExecutor() as executor:
        batch["image"] = list(executor.map(fetch_single_image_with_args, batch["image_url"]))
    return batch


dataset = load_dataset("conceptual_captions", split="validation")
data_path = "data/textual/conceptual_captions/data"
with open(f"{data_path}/test/metadata.csv", "w") as f:
    f.write("file_name,additional_feature\n")

idx = 1
for sample in dataset:
    url = sample["image_url"]

    # print(url)
    image = fetch_single_image(url)
    if image is None:
        print("Failed to fetch image")
        continue
    image = resize_single_image(image)
    caption = sample["caption"]
    image = PIL.Image.fromarray(image)
    # Save image under the name 0001.png, 002.png, etc.
    image.save(f"{data_path}/test/{idx:06d}.png")

    # Add a line to the metadata.csv file
    # 0003.png,This is the caption of the third image
    # etc.
    # Also make sure to remove "," from the captions.

    with open(f"{data_path}/test/metadata.csv", "a") as f:
        # remove , from the caption
        caption = caption.replace(",", "")
        f.write(f"{idx:04d}.png,{caption}\n")

    idx += 1
