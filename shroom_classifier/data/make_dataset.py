import json
from os import path, listdir
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import tarfile
from typing import Tuple
import pandas as pd

# Raw data can be downloaded from https://github.com/visipedia/fgvcx_fungi_comp#data
# Save it to data/raw/
# and unpack it with unpack_data()

WIDTH = 224
HEIGHT = 224
_DATA_PATH_RAW = "data/raw/"
_DATA_PATH_PROCESSED = "data/processed/"


def unpack_data(file_name: str) -> None:
    """Unpacks data from tar.gz file"""
    tar = tarfile.open(path.join(_DATA_PATH_RAW, file_name), "r:gz")
    tar.extractall(path=_DATA_PATH_RAW)


def process_data(size: Tuple = (WIDTH, HEIGHT), info: str = "val.json", out_folder: str = "sample") -> None:
    """
    Use info (json) file to process images. The images are reshaped to size (WIDTH, HEIGHT).
    Saved in process folder together with a new json file with information.

    The images are saved in the following structure:
        data/
            processed/
                out_folder/
                    class1/
                        image1.jpg
                            ...
                    class2/
                        image2.jpg
                            ...
                    ...
                out_folder.json
    """
    info = json.load(open(path.join(_DATA_PATH_RAW, info), "r"))
    new_info = {"annotations": [], "categories": [], "images": []}

    # Create folder if it does not exist
    data_folder = path.join(_DATA_PATH_PROCESSED, out_folder)
    if not path.exists(data_folder):
        os.mkdir(data_folder)

    annotations = info["annotations"]
    categories = info["categories"]
    images = info["images"]
    N = len(images)

    for i in tqdm(range(N)):
        label_dict = [cat for cat in categories if cat["id"] == annotations[i]["category_id"]].pop()
        image_dict = [image for image in images if image["id"] == annotations[i]["image_id"]].pop()

        # Reads image and resizes it
        img = Image.open(path.join(_DATA_PATH_RAW, image_dict["file_name"]))
        img = img.resize(size)

        # Saves image
        new_filename = path.join(data_folder, image_dict["file_name"][7:])
        new_filename = new_filename[:-4].replace(" ", "_").replace(".", "_") + ".JPG"
        if not path.exists(path.dirname(new_filename)):
            os.mkdir(path.dirname(new_filename))

        image_dict["file_name"] = new_filename
        image_dict["width"] = WIDTH
        image_dict["height"] = HEIGHT
        img.save(new_filename)

        # Saves new info
        new_info["annotations"].append(annotations[i])
        new_info["categories"].append(label_dict)
        new_info["images"].append(image_dict)

    # Saves new info
    file_path = path.join(_DATA_PATH_PROCESSED, f"{out_folder}.json")
    with open(file_path, "w") as json_file:
        json.dump(new_info, json_file)


def categories_dictionary(folder_path: str = _DATA_PATH_RAW) -> dict:
    """
    Creates a dictionary with all the categories and supercategories.
    """
    df_categories = []

    # Reads all json files
    for file in listdir(folder_path):
        if file.endswith(".json"):
            info = json.load(open(path.join(folder_path, file), "r"))
            df_categories.append(pd.DataFrame(info["categories"]))

    # Concatenates all dataframes
    df_categories = pd.concat(df_categories)
    categories = {}

    # Adds all categories
    for i, cat in enumerate(df_categories.name.unique()):
        categories[cat] = i
    # Adds all supercategories
    for i, cat in enumerate(df_categories.supercategory.unique()):
        categories[cat] = i
    # Saves dictionary
    np.save(path.join(_DATA_PATH_PROCESSED, "categories.npy"), categories)
    return categories


if __name__ == "__main__":
    process_data(size=(WIDTH, HEIGHT), info="val.json")
    categories_dictionary(_DATA_PATH_RAW)
