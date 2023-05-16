from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import os

import json

from datasets import load_dataset

BASE_COCO_URL="http://images.cocodataset.org/zips/"
BASE_OK_VQA_URL="https://okvqa.allenai.org/static/data/"

SPLITS=["train", "val"]
YEAR="2014"


def download(url):
	http_response = urlopen(url)
	return http_response.read()


def download_and_unzip(url, extract_to='.'):
    downloaded_file = download(url)
    zipfile = ZipFile(BytesIO(downloaded_file))
    zipfile.extractall(path=extract_to)


for split in SPLITS:
	print(f"Processing split {split}")
	# Download images
	print("Downloading images file...")
	download_and_unzip(f"{BASE_COCO_URL}{split}{YEAR}.zip")
	os.rename(f"{split}{YEAR}", split)

	# Download questions
	print("Downloading questions file...")
	questions_file = f"OpenEnded_mscoco_{split}{YEAR}_questions.json"
	download_and_unzip(BASE_OK_VQA_URL + questions_file + ".zip")

	# Download annotations
	print("Downloading annotations file...")
	annotations_file = f"mscoco_{split}{YEAR}_annotations.json"
	download_and_unzip(BASE_OK_VQA_URL + annotations_file + ".zip")

	text, image_paths = []
	if not os.path.exists(f"{split}/metadata.csv"):
		with open(f"{split}/metadata.csv", "w") as f:
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
				with open(f"{split}/metadata.csv", "a") as f:
					f.write(f"{image_idx:07d}.png,{text},{idx}\n")


