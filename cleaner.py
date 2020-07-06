import csv
import glob
import os
import pathlib
import shutil
import uuid

import cv2
import xmltodict

import numpy as np
from sklearn.model_selection import train_test_split


def get_name(f_name):
    if f_name.split(".")[-1] == "jpeg":
        return f_name[:-4]
    else:
        return f_name[:-3]


def rename_files():
    annotations = [os.path.basename(x) for x in glob.glob("data_raw/all/labels/*.xml")]

    images = []
    for ext in ["jpg", "jpeg", "png"]:
        images.extend([os.path.basename(x) for x in glob.glob(f"data_raw/all/images/*.{ext}")])

    # preparing dst folder structure
    shutil.rmtree("data/full", ignore_errors=True)
    pathlib.Path("data/full/images").mkdir(parents=True, exist_ok=False)
    pathlib.Path("data/full/labels").mkdir(parents=True, exist_ok=False)

    # this assumes that all images have their respective xml file
    # this is true for now, even though the inverse is false.
    for img_file in images:
        xml_file = get_name(img_file) + "xml"
        new_name = str(uuid.uuid4())
        shutil.copyfile(src=f"data_raw/all/images/{img_file}", dst=f"data/full/images/{new_name}.jpg")
        shutil.copyfile(src=f"data_raw/all/labels/{xml_file}", dst=f"data/full/labels/{new_name}.xml")


def xml_to_csv_rows(f_name: str):
    raw_name = f_name.split(".")[0]

    xml = xmltodict.parse(open(f"data/full/labels/{f_name}", "rb"))

    # malformed xml files
    try:
        xml = xml["annotation"]["object"]
    except KeyError:
        print(f"{raw_name}.xml is malformed, returning None.")
        return None

    # when image has only one bounding box
    if not isinstance(xml, list):
        xml = [xml]

    # output rows
    rows = []

    # loading image
    full_image = cv2.imread(f"data/full/images/{raw_name}.jpg")

    for i, obj in enumerate(xml):

        # skipping masks worn incorrectly
        if obj["name"] in ["none", "mask_weared_incorrect"]:
            continue

        x1 = int(obj["bndbox"]["xmin"])
        x2 = int(obj["bndbox"]["xmax"])
        y1 = int(obj["bndbox"]["ymin"])
        y2 = int(obj["bndbox"]["ymax"])

        # if x2 - x1 < 48 and y2 - y1 < 48:
        #     continue

        cropped_img = full_image[y1: y2, x1: x2]
        cropped_name = raw_name.split("-")[-1] + "_" + str(i)
        cv2.imwrite(f"data/images/{cropped_name}.jpg", cropped_img)

        orig_name = raw_name + ".jpg"

        # binary
        if obj["name"] in ["good", "face_mask", "with_mask"]:
            target_name = "mask"
        else:
            target_name = "no_mask"

        row = [orig_name, cropped_name + ".jpg", x1, x2, y1, y2, target_name]
        rows.append(row)

    return rows


# extracts the cropped faces and generates the csv file
def generate_cropped_dataset():
    shutil.rmtree("data/images", True)
    pathlib.Path("data/images").mkdir(parents=True, exist_ok=False)

    with open("data/data.csv", "w") as f:
        xml_files = glob.glob("data/full/labels/*.xml")

        writer = csv.writer(f)
        writer.writerow(["original", "cropped", "x1", "x2", "y1", "y2", "target"])

        for xml_file in xml_files:
            rows = xml_to_csv_rows(os.path.basename(xml_file))
            if rows is not None and rows is not []:
                writer.writerows(rows)


# checks if the generated csv ("data/data.csv") has the same number of lines as
# the number of cropped images ("data/images/**").
# dont forget that the csv file has the header line.
def consistency_check():
    with open("data/data.csv", "r") as f:
        lines = f.readlines()
        images = glob.glob("data/images/*.jpg")
        num_images = len(images)
        num_lines = len(lines) - 1

        print(f"There are {num_images} images.")
        assert num_lines == num_images, \
            f"[ASSERTION ERROR] -> csv lines ({num_lines}) and number of cropped images ({num_images}) are different."


# print total number of images and distribution of target labels
def get_stats():
    with open("data/data.csv", "r") as f:
        lines = f.readlines()

        no_masks = 0
        masks = 0

        for line in lines[1:]:
            target = line.strip().split(",")[-1]
            if target == "no_mask":
                no_masks += 1
            elif target == "mask":
                masks += 1
            else:
                print("something wrong", target)

        print("NO:", no_masks)
        print("SI:", masks)


def prepare_colab_structure():
    with open("data/data.csv", "r") as f:
        rows = list(csv.reader(f, delimiter=","))
        rows = rows[1:]

    X = []
    y = []

    # skipping header
    for row in rows:
        cropped_img_path = row[1]
        label = row[-1]

        X.append(cropped_img_path)
        y.append(label)

    y = np.array(y)
    X = np.array(X)
    tot = len(X)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.33, random_state=42, stratify=y_val,
                                                    shuffle=True)

    print("train:", len(X_train))
    print("val:", len(X_val))
    print("test:", len(X_test))

    mask = 0
    no_mask = 0
    for i in y_train:
        if i == "no_mask":
            no_mask += 1
        else:
            mask += 1

    print(f"In train ci sono SI: {mask} e NO: {no_mask}")

    mask = 0
    no_mask = 0
    for i in y_val:
        if i == "no_mask":
            no_mask += 1
        else:
            mask += 1

    print(f"In val ci sono SI: {mask} e NO: {no_mask}")

    mask = 0
    no_mask = 0
    for i in y_test:
        if i == "no_mask":
            no_mask += 1
        else:
            mask += 1

    print(f"In test ci sono SI: {mask} e NO: {no_mask}")

    for i, img in enumerate(X_train):
        shutil.copyfile(src=f"data/images/{img}",
                        dst=f"data/train/{y_train[i]}/{img}")

    print("finished train")

    for i, img in enumerate(X_val):
        shutil.copyfile(src=f"data/images/{img}",
                        dst=f"data/validation/{y_val[i]}/{img}")

    print("finished val")

    for i, img in enumerate(X_test):
        shutil.copyfile(src=f"data/images/{img}",
                        dst=f"data/test/{y_test[i]}/{img}")

    print("finished test")


def main():
    pass
    # rename_files()
    generate_cropped_dataset()
    consistency_check()
    prepare_colab_structure()


if __name__ == "__main__":
    main()
