import concurrent.futures
import io
import os
import urllib

import PIL.Image
from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent


USER_AGENT = get_datasets_user_agent()
PATH = "data/test"  # or "data/train"

MAX_WORKERS = 10  # adjust to your needs
MAX_QUEUE_SIZE = 2 * MAX_WORKERS  # adjust to your needs


def fetch_single_image(image_url, timeout=1):
    print(image_url)
    try:
        request = urllib.request.Request(
            image_url,
            data=None,
            headers={"user-agent": USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=timeout) as req:
            image = PIL.Image.open(io.BytesIO(req.read()))
    except Exception:
        image = None
    return image


def resize_single_image(image: PIL.Image):
    # Resize so that the bigger size is at most 352
    width, height = image.size
    if width > height:
        new_width = 352
        new_height = int(height * 352 / width)
    else:
        new_height = 352
        new_width = int(width * 352 / height)
    image = image.resize((new_width, new_height), PIL.Image.BILINEAR)
    image = image.convert("RGB")
    return image


dataset = load_dataset("conceptual_captions", split="validation")  # or "train"
if not os.path.exists(f"{PATH}/metadata.csv"):
    with open(f"{PATH}/metadata.csv", "w") as f:
        f.write("file_name,caption,idx\n")
    dataset_idx = 0
    image_idx = 0
else:  # get the lastest index
    with open(f"{PATH}/metadata.csv", "r") as f:
        lines = f.readlines()
        image_idx = len(lines) - 1
        dataset_idx = int(lines[-1].split(",")[-1]) + 1
        print(image_idx, dataset_idx)

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_idx = {executor.submit(fetch_single_image, dataset[dataset_idx]["image_url"]): dataset_idx}
    dataset_idx += 1
    while dataset_idx < len(dataset):
        done, _ = concurrent.futures.wait(future_to_idx, return_when=concurrent.futures.FIRST_COMPLETED)
        for future in done:
            idx = future_to_idx.pop(future)
            try:
                image = future.result()
                if image is not None:
                    image = resize_single_image(image)
                    sample = dataset[idx]
                    caption = sample["caption"].replace(",", "").replace(";", "").replace("\n", "").replace("\t", "")
                    image.save(f"{PATH}/{image_idx:07d}.png", "PNG")
                    with open(f"{PATH}/metadata.csv", "a") as f:
                        f.write(f"{image_idx:07d}.png,{caption},{idx}\n")
                    image_idx += 1
            except Exception as exc:
                print(f"Generated an exception: {exc}")

        while len(future_to_idx) < MAX_QUEUE_SIZE and dataset_idx < len(dataset):
            future_to_idx[executor.submit(fetch_single_image, dataset[dataset_idx]["image_url"])] = dataset_idx
            dataset_idx += 1
