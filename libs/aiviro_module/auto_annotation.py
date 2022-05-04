import os
import time
import pathlib
import multiprocessing
import numpy as np
from datetime import datetime
from typing import List, Tuple

from aiviro.utils import bound_box, file_utils, image_utils
from libs.aiviro_module.utils import MySubimageSearchService
from libs.aiviro_module.voc_objects import VocAnnotation, VocBuilder


def image_process(
    in_sis: MySubimageSearchService,
    org_img: np.ndarray,
    sub_img: np.ndarray,
    indx: int,
    label: str
):
    res = in_sis.find_subimage(org_img, sub_img, match_threshold=0.93, label=label)

    if indx % 100 == 0:
        print(f"\tP: {indx} done")

    if res:
        return res, indx
    else:
        return ()


def is_box_inside(container: bound_box.BoundBox, inside_box: bound_box.BoundBox):
    return (container.x_min < inside_box.center_point.x < container.x_max) and\
        (container.y_min < inside_box.center_point.y < container.y_max)


sis = MySubimageSearchService()


def annotate_image(image_path: pathlib.Path, sub_image_db: List[np.ndarray], labels: List[str]):
    cpus = 4
    if multiprocessing.cpu_count() <= 4:
        cpus = max(multiprocessing.cpu_count() - 1, 1)

    s_time = time.time()
    file_name = image_path.stem
    img_to_annotate = file_utils.load_image(str(image_path))

    new_voc_annot = VocBuilder(
        image_path,
        img_to_annotate.shape[1],
        img_to_annotate.shape[0]
    )

    # Process image
    pool_data = [(sis, img_to_annotate, sub_img, i, labels[i]) for i, sub_img in enumerate(sub_image_db, 0)]
    with multiprocessing.Pool(cpus) as p:
        results = p.starmap(image_process, pool_data)

    filtered_results = []
    box_to_remove = []

    for b_tuple in filter(lambda b: b, results):
        boxes = b_tuple[0]
        for box in boxes:
            # Check for duplicity
            add_it = True
            for f_box in filtered_results:
                # Ignore icons in other elements (as this is correct behaviour)
                if (f_box.true_label == "icon" and box.true_label != "icon") or\
                        (f_box.true_label != "icon" and box.true_label == "icon"):
                    continue

                if is_box_inside(box, f_box):  # At the same place
                    if box.area > f_box.area:
                        box_to_remove.append(f_box)
                    else:
                        add_it = False

            if add_it:
                filtered_results.append(box)

    filtered_results = list(set(filtered_results) - set(box_to_remove))
    for box in filtered_results:
        new_voc_annot.add(
            box.true_label,
            box.x_min, box.y_min,
            box.x_max, box.y_max
        )

    new_voc_annot.save(image_path.parent / f"{file_name}.xml")
    print(f"\tEvaluation time: {time.time() - s_time} s.")


def create_database(xml_folder: pathlib.Path, database_folder: pathlib.Path) -> Tuple[List[np.ndarray], List[str]]:
    # Delete old database
    for f in database_folder.glob("*.png"):
        os.remove(f)

    # Get all annotation files
    xml_files = sorted(xml_folder.glob("**/*.xml"))
    print(f"\tXml files loaded: {len(xml_files)}")

    # Find unique boxes
    u_hashes = set()
    u_boxes = []

    for annot_f in xml_files:
        voc_annot = VocAnnotation(annot_f)
        for box in voc_annot.boxes:
            img_hash_size = 5 if box.area < 25 * 25 else 15
            h = image_utils.image_perceptual_hash(box.img, hash_size=img_hash_size) # 15 - default
            if h in u_hashes:
                continue

            u_hashes.add(h)
            u_boxes.append(box)

    print(f"\tNumber of unique boxes: {len(u_boxes)}")

    sub_images: List[np.ndarray] = []
    sub_images_labels: List[str] = []
    # Create sub-image database
    for i, box in enumerate(u_boxes):
        img_name = box.true_label + f"_subimg_{i}_" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".png"
        file_utils.save_image(
            str(database_folder / img_name),
            box.img
        )
        sub_images.append(box.img)
        sub_images_labels.append(box.true_label)

    return sub_images, sub_images_labels


def load_database(database_folder: pathlib.Path) -> Tuple[List[np.ndarray], List[str]]:
    sub_images: List[np.ndarray] = []
    sub_images_labels: List[str] = []

    for img_path in database_folder.glob("*.png"):
        sub_images_labels.append(img_path.stem.split("_")[0])
        sub_images.append(file_utils.load_image(str(img_path)))

    return sub_images, sub_images_labels
