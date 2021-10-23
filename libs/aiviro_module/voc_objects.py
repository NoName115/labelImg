import os
from typing import List
from xml.dom import minidom
from xml.etree import ElementTree as Et

from aiviro.utils import file_utils
from libs.aiviro_module.utils import MyBox


class VocBuilder:
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

    def _process_objects(self, xml_objects: List[Et.Element]):
        boxes = []
        img_path = os.path.join(self.folder, self.filename_img)
        img = file_utils.load_image(img_path)
        if img is None:
            raise RuntimeError(f"Image not found at: {img_path}")

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
                x_min, y_min, x_max, y_max, label, img[y_min:y_max, x_min:x_max]
            )
            boxes.append(box)
        return boxes
