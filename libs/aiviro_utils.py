import os
import cv2
import numpy as np
from typing import Tuple, List
from enum import Enum, unique
from xml.dom import minidom
from xml.etree import ElementTree as Et


@unique
class AiviroCoreElementLabels(Enum):
    DEFAULT = "n/a"

    # objects which can be detected below
    # never put '-' into the label name !!
    TEXT_ELEMENT = "text"
    BUTTON_ELEMENT = "button"
    ICON_ELEMENT = "icon"
    TEXT_AREA_ELEMENT = "textarea"
    INPUT_ELEMENT = "input"
    CHECK_BOX_ELEMENT = "checkbox"
    RADIO_BUTTON_ELEMENT = "radiobutton"
    TOGGLE_ELEMENT = "toggle"


class BoundBox:
    def __init__(
        self,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
        label: AiviroCoreElementLabels = AiviroCoreElementLabels.DEFAULT,
        score: float = 0.0
    ):
        assert x_min < x_max and y_min < y_max, (
            f"Invalid x/y positions, x_min: '{x_min}', x_max: '{x_max}',"
            f" y_min: '{y_min}', y_max: '{y_max}'"
        )
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

        self.label = label
        self.score = score

    @property
    def center_point(self) -> Tuple[int, int]:
        return (self.x_min + self.x_max) // 2, (self.y_min + self.y_max) // 2

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def area(self) -> int:
        return self.width * self.height

    def __str__(self) -> str:
        return (
            f"<BoundBox at ;"  # {hex(id(self))}
            f' min_pt="{[self.x_min, self.y_min]}"'
            f' max_pt="{[self.x_max, self.y_max]}"'
            f' label="{self.label.value}/>'
        )

    __repr__ = __str__

    def __hash__(self) -> int:
        return hash((
            self.x_min,
            self.y_min,
            self.x_max,
            self.y_max,
            self.label.value,
        ))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoundBox):
            return NotImplemented
        return (
            self.x_min == other.x_min
            and self.y_min == other.y_min
            and self.x_max == other.x_max
            and self.y_max == other.y_max
            and self.label == other.label
        )


class MyBox(BoundBox):

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
            self.true_label
        ))


def _interval_overlap(interval_a: Tuple[int, int], interval_b: Tuple[int, int]) -> int:
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def bbox_iou(area1: BoundBox, area2: BoundBox) -> float:
    intersect_w = _interval_overlap(
        (area1.x_min, area1.x_max), (area2.x_min, area2.x_max)
    )
    intersect_h = _interval_overlap(
        (area1.y_min, area1.y_max), (area2.y_min, area2.y_max)
    )

    intersection = intersect_w * intersect_h

    w1, h1 = area1.x_max - area1.x_min, area1.y_max - area1.y_min
    w2, h2 = area2.x_max - area2.x_min, area2.y_max - area2.y_min

    union = w1 * h1 + w2 * h2 - intersection

    return float(intersection) / union


def load_image(filepath: str) -> np.ndarray:
    """Loads image defined in filepath

    Arguments:
        filepath {str} -- path to image to load
    Returns:
        OpenCV matrix representation of the image
    """
    return cv2.imread(filepath, cv2.IMREAD_COLOR)


def save_image(file_path: str, image: np.ndarray):
    """[summary]
        Saves image in selected path
    Arguments:
        file_path {str} -- path where to save image
        image {OpenCV matrix} -- image to be saved
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    cv2.imwrite(file_path, image)


def image_perceptual_hash(image: np.ndarray, hash_size: int = 12) -> int:
    """
    Method to calculate perceptual image hash. This function result also includes the source image size.
    This hash can be used to find similar images.
    :param image: Source image to be hashed.
    :param hash_size: Size of the hash. The higher the value the more finer, but larger the hash.
    :return: Hash of the source image. Result size hash_size**2/8 + 4 bytes
    """
    fingerprint_size = image.shape[0] << 16 | image.shape[1]

    scaled_image = cv2.resize(image, (hash_size + 1, hash_size))
    fingerprint_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)

    diffs = fingerprint_image[:, 1:] > fingerprint_image[:, :-1]
    image_hash = sum([0x1 << i for i, v in enumerate(diffs.flatten()) if v])
    return image_hash << 32 | fingerprint_size


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
        img = load_image(os.path.join(self.folder, self.filename_img))
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
            -> List[BoundBox]:
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
        found_matches: List[BoundBox] = []
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
    def _filter_duplicate_matches(found_objects: List[BoundBox], overlap_threshold: float)\
            -> List[BoundBox]:
        """Filters duplicate template matching matches for the same object.

        As template matching is done on multiple scales, some objects can be matched multiple times. This function
        filters all duplicate matches so that each object in the image is only matched once.

        Parameters:
            found_objects - object matches to filter
            overlap_threshold - IoU of match bounding boxes for which to consider the matches duplicate

        Return:
            Subset of found_object where each match corresponds to a different object than all other matches"""
        # Unique matches for each object
        filtered_found: List[BoundBox] = []

        for found in found_objects:
            for compare in found_objects:
                if bbox_iou(found, compare) > overlap_threshold:
                    if found.score < compare.score or \
                            (found.score == compare.score and compare in filtered_found):
                        break
            else:
                filtered_found.append(found)

        return filtered_found
