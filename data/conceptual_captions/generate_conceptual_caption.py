import io
import multiprocessing
from typing import Dict, List, Union
from urllib.request import Request, urlopen

import PIL.Image
from datasets.utils.file_utils import get_datasets_user_agent


USER_AGENT = get_datasets_user_agent()


def fetch_image(image_url: str, timeout: float = 0.5) -> PIL.Image.Image:
    """
    Fetches a single image from a given URL and returns it as a PIL Image object.

    Args:
        image_url (str): The URL of the image to fetch.
        timeout (float): The timeout value for the request (in seconds).

    Returns:
        A PIL Image object representing the fetched image, or None if the image could not be fetched.
    """
    request = Request(image_url, data=None, headers={"user-agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as req:
        image = PIL.Image.open(io.BytesIO(req.read()))
    return image


def resize_image(image: PIL.Image) -> PIL.Image:
    """
    Resize a single image to have the bigger size at most 352 pixels while maintaining aspect ratio.
    Remove metadata from the image.

    Args:
        image (PIL.Image): The image to be resized.

    Returns:
        PIL.Image: The resized image without metadata.
    """
    # Resize so that the bigger size is at most 352
    width, height = image.size
    if width > height:
        new_width = 352
        new_height = int(height * 352 / width)
    else:
        new_height = 352
        new_width = int(width * 352 / height)
    image = image.resize((new_width, new_height), PIL.Image.BILINEAR)
    image = image.convert("RGB")  # Make sure the image is RGB
    data = list(image.getdata())  # Get only the image data, and place it in a new image to remove metadata
    image_without_exif = PIL.Image.new(image.mode, image.size)
    image_without_exif.putdata(data)
    return image_without_exif


def fetch_and_resize(img_url: str) -> Union[PIL.Image.Image, None]:
    """
    Fetches an image from a given URL and resizes it.

    Args:
        img_url (str): The URL of the image to fetch.

    Returns:
        numpy.ndarray: The resized image as a NumPy array, or None if an error occurred.
    """
    try:
        image = fetch_image(img_url)
        image = resize_image(image)
    except Exception:
        image = None
    return image


def process(example: Dict[str, List[str]]) -> Dict[str, List[Union[str, PIL.Image.Image]]]:
    output = {"images": [], "text": []}

    with multiprocessing.Pool() as pool:
        images = pool.starmap(fetch_and_resize, [(url,) for url in example["image_url"]])

    for idx, image in enumerate(images):
        if image is not None:
            output["images"].append(image)
            output["text"].append(example["caption"][idx])

    return output


if __name__ == "__main__":
    from datasets import Dataset, features, load_dataset

    for split in ["train", "test"]:
        dataset = load_dataset("conceptual_captions", split="train" if split == "train" else "validation")
        num_cpu = multiprocessing.cpu_count() // 2
        dataset = dataset.map(
            process,
            batched=True,
            batch_size=200,
            remove_columns=["caption", "image_url"],
            num_proc=num_cpu,
            load_from_cache_file=True,
            features=features.Features({"images": features.Image(decode=True), "text": features.Value("string")}),
        )
        dataset.save_to_disk(f"conceptual-captions-{split}")
        dataset = Dataset.load_from_disk(f"conceptual-captions-{split}")

        retry = 500

        for i in range(retry):
            try:
                dataset.push_to_hub("jat-project/jat-dataset", "conceptual-captions", split=split)
                break
            except Exception:
                print(f"Retry {i+1}/{retry}")
