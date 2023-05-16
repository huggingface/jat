from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import os
import json
import argparse
from huggingface_hub import HfApi, repocard, upload_folder

BASE_COCO_URL = "http://images.cocodataset.org/zips/"
BASE_OK_VQA_URL = "https://okvqa.allenai.org/static/data/"

SPLITS = ["train", "val"]
YEAR = "2014"
DATA_DIR = "./dataset"


def download(url):
    http_response = urlopen(url)
    return http_response.read()


def download_and_unzip(url, extract_to='.'):
    downloaded_file = download(url)
    zipfile = ZipFile(BytesIO(downloaded_file))
    zipfile.extractall(path=extract_to)


def generate_dataset_card(
        dir_path: str,
):
    readme_path = os.path.join(dir_path, "README.md")
    readme = f"""
    The OK-VQA dataset (https://okvqa.allenai.org/) \n
    This environment was created as part of the Generally Intelligent Agents project gia:
    https://github.com/huggingface/gia \n
    \n
    """

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)

    metadata = {}
    metadata["library_name"] = "gia"
    metadata["tags"] = [
        "deep-reinforcement-learning",
        "reinforcement-learning",
        "gia",
        "multi-task",
        "multi-modal",
        "imitation-learning",
        "offline-reinforcement-learning",
    ]
    repocard.metadata_save(readme_path, metadata)


def push_to_hf(dir_path: str, repo_name: str):
    _ = HfApi().create_repo(repo_id=repo_name, private=False, exist_ok=True, repo_type="dataset")

    upload_folder(
        repo_id=repo_name, folder_path=dir_path, path_in_repo=".", ignore_patterns=[".git/*"], repo_type="dataset"
    )


def main(hf_repo_name, push_to_hub=False, download_coco=True):
    os.makedirs(DATA_DIR, exist_ok=True)
    for split in SPLITS:
        print(f"Processing split {split}")

        # Download images
        if download_coco:
            print("Downloading images file...")
            download_and_unzip(f"{BASE_COCO_URL}{split}{YEAR}.zip", extract_to=DATA_DIR)
            os.rename(f"{DATA_DIR}/{split}{YEAR}", split)

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
                for idx, question in enumerate(questions):
                    annotation = annotations[idx]
                    assert question["question_id"] == annotation["question_id"], \
                        print(f"Question id doesn't match at index {idx}")
                    assert question["image_id"] == annotation["image_id"], \
                        print(f"Image id {idx} doesn't match at index {idx}")

                    text = f"Q: {question['question']} A: {annotation['answers'][0]['answer']}"
                    image_idx = question["image_id"]
                    with open(f"{DATA_DIR}/{split}/metadata.csv", "a") as f:
                        f.write(f"{image_idx:07d}.png,{text},{idx}\n")

    if push_to_hub:
        generate_dataset_card(DATA_DIR)
        push_to_hf(DATA_DIR, hf_repo_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--download_coco", action="store_true")
    parser.add_argument("--hf_repo_name", type=str)
    args = parser.parse_args()

    main(**vars(args))
