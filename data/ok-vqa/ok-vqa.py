import argparse
import io
import json
import os
import tarfile
from io import BytesIO
from tarfile import TarFile
from urllib.request import urlopen
from zipfile import ZipFile

import PIL.Image


BASE_COCO_URL = "http://images.cocodataset.org/zips/"
BASE_OK_VQA_URL = "https://okvqa.allenai.org/static/data/"

FILES_PER_TAR = 100

SPLITS = ["train", "val"]
YEAR = "2014"
DATA_DIR = "./dataset"


def download(url):
    http_response = urlopen(url)
    return http_response.read()


def download_and_unzip(url, extract_to="."):
    downloaded_file = download(url)
    zipfile = ZipFile(BytesIO(downloaded_file))
    zipfile.extractall(path=extract_to)


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


def main(download_coco=True):
    os.makedirs(DATA_DIR, exist_ok=True)
    for split in SPLITS:
        print(f"Processing split {split}")

        # Download images
        if download_coco:
            print("Downloading images file...")
            download_and_unzip(f"{BASE_COCO_URL}{split}{YEAR}.zip")
            os.makedirs(f"{DATA_DIR}/{split}", exist_ok=True)

        # Download questions
        print("Downloading questions file...")
        questions_file = f"OpenEnded_mscoco_{split}{YEAR}_questions.json"
        download_and_unzip(BASE_OK_VQA_URL + questions_file + ".zip")

        # Download annotations
        print("Downloading annotations file...")
        annotations_file = f"mscoco_{split}{YEAR}_annotations.json"
        download_and_unzip(BASE_OK_VQA_URL + annotations_file + ".zip")

        if not os.path.exists(f"{DATA_DIR}/{split}/metadata.csv"):
            with open(f"{DATA_DIR}/{split}/metadata.csv", "w") as f:
                f.write("file_name,text,idx\n")

        with open(questions_file, "r") as json_questions_file:
            with open(annotations_file, "r") as json_annotations_file:
                questions = json.load(json_questions_file)["questions"]
                annotations = json.load(json_annotations_file)["annotations"]
                tar = None
                for idx, question in enumerate(questions):
                    if idx % FILES_PER_TAR == 0:
                        if tar is not None:
                            tar.close()
                        tar = TarFile(f"{DATA_DIR}/{split}/images_{idx // FILES_PER_TAR}.tar.gz", mode="w")

                    annotation = annotations[idx]
                    assert question["question_id"] == annotation["question_id"], print(
                        f"Question id doesn't match at index {idx}"
                    )
                    assert question["image_id"] == annotation["image_id"], print(
                        f"Image id {idx} doesn't match at index {idx}"
                    )

                    text = f"Q: {question['question']} A: {annotation['answers'][0]['answer']}"
                    image_name = f"COCO_{split}{YEAR}_{question['image_id']:012d}.jpg"
                    image = PIL.Image.open(f"{split}{YEAR}/{image_name}")
                    image = resize_single_image(image)
                    output = io.BytesIO()
                    image.save(output, format="JPEG")
                    output.seek(0)
                    info = tarfile.TarInfo(image_name)
                    info.size = len(output.getvalue())
                    tar.addfile(info, fileobj=output)
                    output.close()
                    with open(f"{DATA_DIR}/{split}/metadata.csv", "a") as f:
                        f.write(f"{image_name},{text},{idx}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_coco", action="store_true")
    args = parser.parse_args()

    main(**vars(args))
