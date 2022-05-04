import pathlib
from typing import List, Optional
from xml.dom import minidom
from xml.etree import ElementTree as Et

from aiviro.utils import file_utils
from libs.aiviro_module.utils import MyBox


class VocBuilder:
    def __init__(self, file_path: pathlib.Path, width: int, height: int):
        etop = Et.Element("annotation")

        self.sub_elem(etop, "folder", str(file_path.parent))
        self.sub_elem(etop, "filename", str(file_path.name))
        self.sub_elem(etop, "path", str(file_path))

        esource = self.sub_elem(etop, "source")
        self.sub_elem(esource, "database", "Unknown")

        esize = self.sub_elem(etop, "size")
        self.sub_elem(esize, "width", str(width))
        self.sub_elem(esize, "height", str(height))
        self.sub_elem(esize, "depth", "3")

        self.sub_elem(etop, "segmented", "0")

        self.top = etop

    def add(self, name: str, left: int, top: int, right: int, bottom: int, pose: str = "Unspecified"):
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

    def save(self, output_file: pathlib.Path, indent="    "):
        # reparse using minidom for indentation
        rough = Et.tostring(self.top, "utf-8")
        pretty = minidom.parseString(rough).documentElement.toprettyxml(indent=indent)
        with open(output_file, "w") as out:
            out.write(pretty)

    @staticmethod
    def sub_elem(parent: Et.Element, tag: Optional[str], text: Optional[str] = None):
        sub = Et.SubElement(parent, tag)
        if tag is not None:
            sub.text = text
        return sub


class VocAnnotation:
    def __init__(self, xml_file_path: pathlib.Path):
        tree = Et.parse(xml_file_path)
        root = tree.getroot()

        self.filename_annot = xml_file_path.name

        self.folder = xml_file_path.parent  # root.find("folder").text
        self.filename_img = root.find("filename").text
        self.path = root.find("path").text

        size = root.find("size")
        self.width = size.find("width").text
        self.height = size.find("height").text

        self.boxes = self._process_objects(root.findall("object"))

    def _process_objects(self, xml_objects: List[Et.Element]):
        boxes = []

        img_path = self.folder / self.filename_img
        img = file_utils.load_image(str(img_path))
        if img is None:
            raise RuntimeError(f"Image not found at: {img_path}")

        for obj in xml_objects:
            label = obj.find("name").text

            b_box = obj.find("bndbox")
            x_min = int(b_box.find("xmin").text)
            y_min = int(b_box.find("ymin").text)
            x_max = int(b_box.find("xmax").text)
            y_max = int(b_box.find("ymax").text)

            box = MyBox(
                x_min, y_min, x_max, y_max, label, img[y_min:y_max, x_min:x_max]
            )
            boxes.append(box)
        return boxes
