from typing import List
from aiviro.utils import bound_box
from aiviro.services.layout_service import LayoutService
from libs.aiviro_module.utils import KNOWN_LABELS, MyBox


class AnnotationChecker:

    def __init__(self, input_data: List[MyBox]):
        self._data = input_data
        self.messages: List[str] = []

    @property
    def is_valid(self) -> bool:
        return not self.messages

    def check_all(self):
        self.check_invalid_labels()
        self.check_intersecting_icons()
        self.check_texticonbutton_contain_icon()
        self.check_textbutton_does_not_contain_icons()

    def check_intersecting_icons(self):
        ls = LayoutService(robot_id='label-img')

        icons = list(filter(lambda box: box.true_label in ['icon', 'checkbox', 'radiobutton'], self._data))
        if icons:
            # Check if intersecting
            if ls.are_intersecting(icons):
                for bb1 in icons:
                    for bb2 in icons:
                        if (
                            bb1 != bb2
                            and bound_box.area_inside_to_full_ratio(bb1, bb2) > 0.75
                        ):
                            self.messages.append(f"--> {bb1} & {bb2} are intersecting")

    def check_texticonbutton_contain_icon(self):
        text_icon_buttons = list(filter(lambda box: box.true_label == 'texticonbutton', self._data))
        if text_icon_buttons:
            icons = list(filter(lambda box: box.true_label in ['icon', 'checkbox', 'radiobutton'], self._data))
            for bb in text_icon_buttons:
                contain = False
                for ic in icons:
                    if bound_box.is_inside(ic, bb):
                        contain = True
                        break
                if not contain:
                    self.messages.append(f"--> {bb} does not contain icon")

    def check_textbutton_does_not_contain_icons(self):
        text_buttons = list(filter(lambda box: box.true_label == 'textbutton', self._data))
        if text_buttons:
            icons = list(filter(lambda box: box.true_label in ['icon', 'checkbox', 'radiobutton'], self._data))
            for bb in text_buttons:
                contain = False
                for ic in icons:
                    if bound_box.is_inside(ic, bb):
                        contain = True
                if contain:
                    self.messages.append(f"--> {bb} contain icon")

    def check_invalid_labels(self):
        for box in self._data:
            if box.true_label not in KNOWN_LABELS:
                self.messages.append(f"--> {box} has invalid label")
