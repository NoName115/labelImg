import os
import cv2
import json
import time
import glob
import multiprocessing
import numpy as np
from datetime import datetime
from typing import List, Tuple
from xml.dom import minidom
from xml.etree import ElementTree as Et
from aiviro.utils import bound_box, file_utils, image_utils
from aiviro.constants.ui_constants import AiviroCoreElementLabels


class MyBox(bound_box.BoundBox):

    def __init__(self, xm, ym, xmx, ymx, label: str, img=None):
        self.true_label = label
        if label in ["texticonbutton", "textbutton"]:
            label = "button"

        super().__init__(xm, ym, xmx, ymx, AiviroCoreElementLabels(label))
        self.img = img

    def __hash__(self) -> int:
        return hash((
            self.x_min,
            self.y_min,
            self.x_max,
            self.y_max,
            self.true_label,
            json.dumps(self.extra, sort_keys=True, cls=bound_box.BoundBoxJsonEncoder)
        ))


class VocBuilder:
    top: Et.Element

    def __init__(self, folder, filename, fullpath, width, height):

        etop = Et.Element("annotation")

        self.sub_elem(etop, "folder", folder)

        self.sub_elem(etop, "filename", filename)
        self.sub_elem(etop, "path", fullpath)

        esource = self.sub_elem(etop, "source")
        self.sub_elem(esource, "database", "Unknown")

        esize = self.sub_elem(etop, "size")
        self.sub_elem(esize, "width", str(width))
        self.sub_elem(esize, "height", str(height))
        self.sub_elem(esize, "depth", "3")

        self.sub_elem(etop, "segmented", "0")

        self.top = etop

    def add(self, name, left, top, right, bottom, pose="Unspecified"):
        etop = self.top
        eobject = self.sub_elem(etop, "object")

        self.sub_elem(eobject, "name", name)
        self.sub_elem(eobject, "pose", pose)
        self.sub_elem(eobject, "truncated", "0")
        self.sub_elem(eobject, "difficult", "0")
        self.sub_elem(eobject, "occluded", "0")

        ebndbox = self.sub_elem(eobject, "bndbox")
        self.sub_elem(ebndbox, "xmin", str(left))
        self.sub_elem(ebndbox, "xmax", str(right))
        self.sub_elem(ebndbox, "ymin", str(top))
        self.sub_elem(ebndbox, "ymax", str(bottom))

        return self

    def save(self, outfile, indent="    "):
        # reparse using minidom for indentation
        rough = Et.tostring(self.top, "utf-8")
        pretty = minidom.parseString(rough).documentElement.toprettyxml(indent=indent)
        with open(outfile, "w") as out:
            out.write(pretty)

    @staticmethod
    def sub_elem(parent, tag, text=None):
        sub = Et.SubElement(parent, tag)
        if tag != None:
            sub.text = text
        return sub


class VocAnnotation:

    def __init__(self, filename: str):
        tree = Et.parse(filename)
        root = tree.getroot()

        self.filename_annot = os.path.basename(filename)

        self.folder = os.path.dirname(filename)  # root.find("folder").text
        self.filename_img = root.find("filename").text
        self.path = root.find("path").text

        size = root.find("size")
        self.width = size.find("width").text
        self.height = size.find("height").text

        self.boxes = self._process_objects(root.findall("object"))

    def _process_objects(self, xml_objects):
        boxes = []
        img = file_utils.load_image(os.path.join(self.folder, self.filename_img))
        # print(os.path.join(self.folder, self.filename_annot))

        for obj in xml_objects:
            label = obj.find("name").text
            # if label in ["texticonbutton", "textbutton"]:
            #    label = "button"

            bndbox = obj.find("bndbox")
            x_min = int(bndbox.find("xmin").text)
            y_min = int(bndbox.find("ymin").text)
            x_max = int(bndbox.find("xmax").text)
            y_max = int(bndbox.find("ymax").text)

            box = MyBox(
                x_min,
                y_min,
                x_max,
                y_max,
                label,
                img[y_min: y_max, x_min: x_max]
            )
            boxes.append(box)
        return boxes


class MySubimageSearchService:

    def find_subimage(self, image: np.ndarray, to_find: np.ndarray, match_threshold: float, label: str)\
            -> List[bound_box.BoundBox]:
        """Finds all occurrences of a given image in another image.

        This function tolerates up to 20% differences in scale, but is most reliable at identical scales.

        Parameters:
            image - Image in which to search for subimages
            to_find - Subimage which to look for
            match_threshold - Threshold of image similarity to consider it a match

        Return:
            List of tuples (bbox, score) where bbox is a bounding box of a found element and
            score is the score of that match.
            """
        # Preprocess images for template matching
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template = cv2.cvtColor(to_find, cv2.COLOR_BGR2GRAY)
        template_width, template_height = template.shape[::-1]

        # Prepare data
        found_matches: List[bound_box.BoundBox] = []
        max_object_count = 50  # chosen by agreement

        # Template matching
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= match_threshold)

        if len(loc[0]) > max_object_count:
            return []

        # Construct bounding boxes out of found matches
        for pt in zip(*loc[::-1]):
            # Calculate the bounding box size on the original image
            x_min, y_min = int(pt[0]), int(pt[1])
            x_max, y_max = int((pt[0] + template_width)), int((pt[1] + template_height))
            found_matches.append(
                MyBox(
                    xm=x_min, ym=y_min,
                    xmx=x_max, ymx=y_max,
                    label=label,
                )
            )

        # Filter matches of the same object found on different scales
        # This can be found by checking bounding box iou
        return self._filter_duplicate_matches(found_matches, overlap_threshold=0.5)  # Unique matches for each object

    @staticmethod
    def _filter_duplicate_matches(found_objects: List[bound_box.BoundBox], overlap_threshold: float)\
            -> List[bound_box.BoundBox]:
        """Filters duplicate template matching matches for the same object.

        As template matching is done on multiple scales, some objects can be matched multiple times. This function
        filters all duplicate matches so that each object in the image is only matched once.

        Parameters:
            found_objects - object matches to filter
            overlap_threshold - IoU of match bounding boxes for which to consider the matches duplicate

        Return:
            Subset of found_object where each match corresponds to a different object than all other matches"""
        # Unique matches for each object
        filtered_found: List[bound_box.BoundBox] = []

        for found in found_objects:
            for compare in found_objects:
                if bound_box.bbox_iou(found, compare) > overlap_threshold:
                    if found.score < compare.score or \
                            (found.score == compare.score and compare in filtered_found):
                        break
            else:
                filtered_found.append(found)

        return filtered_found


def image_process(in_sis: MySubimageSearchService, org_img: np.ndarray, sub_img: np.ndarray, indx: int, label: str):
    res = in_sis.find_subimage(org_img, sub_img, match_threshold=0.93, label=label)

    if indx % 100 == 0:
        print(f"\tP: {indx} done")

    if res:
        return res, indx
    else:
        return ()


def is_box_inside(container, inside_box):
    return (container.x_min < inside_box.center_point.x < container.x_max) and\
        (container.y_min < inside_box.center_point.y < container.y_max)


sis = MySubimageSearchService()


def annotate_image(image_path: str, sub_image_db: List[np.ndarray], labels: List[str]):
    cpus = 4
    if multiprocessing.cpu_count() <= 4:
        cpus = max(multiprocessing.cpu_count() - 1, 1)

    s_time = time.time()
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    img_to_annotate = file_utils.load_image(image_path)

    new_voc_annot = VocBuilder(
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
                if (f_box.true_label == "icon" and box.true_label != "icon") or (f_box.true_label != "icon" and box.true_label == "icon"):
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
        voc_annot = VocAnnotation(os.path.join(xml_folder, xml_file))
        for box in voc_annot.boxes:
            h = image_utils.image_perceptual_hash(box.img, hash_size=15)
            if h in u_hashes:
                continue

            u_hashes.add(h)
            u_boxes.append(box)

    print(f"\tNumber of unique boxes: {len(u_boxes)}")

    sub_images: List[np.ndarray] = []
    sub_images_labels: List[str] = []
    # Create sub-image database
    for box in u_boxes:
        file_utils.save_image(
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
        sub_images.append(file_utils.load_image(os.path.join(database_folder, sub_img)))

    return sub_images, sub_images_labels
