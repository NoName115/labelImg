import os
import time
import glob
import multiprocessing
import numpy as np
from datetime import datetime
from typing import List, Tuple
from libs import aiviro_utils


def image_process(
    in_sis: aiviro_utils.MySubimageSearchService,
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


def is_box_inside(container: aiviro_utils.BoundBox, inside_box: aiviro_utils.BoundBox):
    return (container.x_min < inside_box.center_point[0] < container.x_max) and\
        (container.y_min < inside_box.center_point[1] < container.y_max)


sis = aiviro_utils.MySubimageSearchService()


def annotate_image(image_path: str, sub_image_db: List[np.ndarray], labels: List[str]):
    cpus = 4
    if multiprocessing.cpu_count() <= 4:
        cpus = max(multiprocessing.cpu_count() - 1, 1)

    s_time = time.time()
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    img_to_annotate = aiviro_utils.load_image(image_path)

    new_voc_annot = aiviro_utils.VocBuilder(
        os.path.dirname(image_path),
        os.path.basename(image_path),
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
    new_voc_annot.save(os.path.join(os.path.dirname(image_path), file_name + ".xml"))
    print(f"\tEvaluation time: {time.time() - s_time} s.")


def create_database(xml_folder: str, database_folder: str) -> Tuple[List[np.ndarray], List[str]]:
    # Delete old database
    for f in glob.glob(f'{os.path.normpath(database_folder)}/*.png'):
        os.remove(f)

    xml_files = sorted(filter(lambda x: x.endswith(".xml"), os.listdir(xml_folder)))
    print(f"\tXml files loaded: {len(xml_files)}")

    # Find unique boxes
    u_hashes = set()
    u_boxes = []

    for xml_file in xml_files:
        voc_annot = aiviro_utils.VocAnnotation(os.path.join(xml_folder, xml_file))
        for box in voc_annot.boxes:
            h = aiviro_utils.image_perceptual_hash(box.img, hash_size=15)
            if h in u_hashes:
                continue

            u_hashes.add(h)
            u_boxes.append(box)

    print(f"\tNumber of unique boxes: {len(u_boxes)}")

    sub_images: List[np.ndarray] = []
    sub_images_labels: List[str] = []
    # Create sub-image database
    for box in u_boxes:
        aiviro_utils.save_image(
            os.path.join(
                database_folder,
                box.true_label + "_subimg_" + datetime.now().isoformat() + ".png"
            ),
            box.img
        )
        sub_images.append(box.img)
        sub_images_labels.append(box.true_label)

    return sub_images, sub_images_labels


def load_database(database_folder: str) -> Tuple[List[np.ndarray], List[str]]:
    sub_images: List[np.ndarray] = []
    sub_images_labels: List[str] = []

    for sub_img in filter(lambda x: x.endswith(".png"), os.listdir(database_folder)):
        sub_images_labels.append(sub_img.split("_")[0])
        sub_images.append(aiviro_utils.load_image(os.path.join(database_folder, sub_img)))

    return sub_images, sub_images_labels
