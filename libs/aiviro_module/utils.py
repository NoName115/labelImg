from typing import List

import cv2
import numpy as np
from aiviro.utils import bound_box
from aiviro.utils.bound_box import BoundBox
from aiviro.constants.ui_constants import AiviroCoreElementLabels


class MyBox(BoundBox):
    def __init__(self, xm, ym, xmx, ymx, label: str, img=None):
        self.true_label = label
        if label in ["texticonbutton", "textbutton"]:
            label = "button"

        super().__init__(xm, ym, xmx, ymx, AiviroCoreElementLabels(label))
        self.img = img

    def __hash__(self) -> int:
        return hash((self.x_min, self.y_min, self.x_max, self.y_max, self.true_label))


class MySubimageSearchService:

    def find_subimage(
        self,
        image: np.ndarray,
        to_find: np.ndarray,
        match_threshold: float,
        label: str
    ) -> List[bound_box.BoundBox]:
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
    def _filter_duplicate_matches(
        found_objects: List[bound_box.BoundBox],
        overlap_threshold: float
    ) -> List[bound_box.BoundBox]:
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
