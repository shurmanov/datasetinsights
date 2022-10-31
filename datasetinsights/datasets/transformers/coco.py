import hashlib
import json
import logging
import multiprocessing
import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pycocotools.mask import encode as mask_to_rle
from pycocotools.mask import decode as decode_to_mask
import cv2
from skimage import measure
from tqdm import tqdm

import datasetinsights.constants as const
from datasetinsights.datasets.transformers.base import DatasetTransformer
from datasetinsights.datasets.unity_perception import (
    AnnotationDefinitions,
    Captures,
)
from datasetinsights.datasets.unity_perception.validation import NoRecordError

logger = logging.getLogger(__name__)

process_pool = multiprocessing.Pool()

def uuid_to_int(input_uuid):
    try:
        u = int(str(uuid.UUID(input_uuid).int)[:9])
    except (AttributeError, ValueError):
        u = int(
            hashlib.md5(str(input_uuid).encode("utf8")).hexdigest(), base=16
        )

    return u


class COCOInstancesTransformer(DatasetTransformer, format="COCO-Instances"):
    """Convert Synthetic dataset to COCO format.

    This transformer convert Synthetic dataset into annotations in instance
    format (e.g. instances_train2017.json, instances_val2017.json)

    Note: We assume "valid images" in the COCO dataset must contain at least one
    bounding box annotation. Therefore, all images that contain no bounding
    boxes will be dropped. Instance segmentation are considered optional
    in the converted dataset as some synthetic dataset might be generated
    without it.

    Args:
        data_root (str): root directory of the dataset
    """

    # The annotation_definition.name is not a reliable way to know the type
    # of annotation definition. This will be improved once the perception
    # package introduced the annotation definition type in the future.
    BBOX_NAME = r"^(?:2[dD]\s)?bounding\sbox$"
    INSTANCE_SEGMENTATION_NAME = r"^instance\ssegmentation$"

    def __init__(self, data_root):
        self._data_root = Path(data_root)

        ann_def = AnnotationDefinitions(
            data_root, version=const.DEFAULT_PERCEPTION_VERSION
        )
        self._bbox_def = ann_def.find_by_name(self.BBOX_NAME)
        try:
            self._instance_segmentation_def = ann_def.find_by_name(
                self.INSTANCE_SEGMENTATION_NAME
            )
            self._has_instance_seg = True
        except NoRecordError as e:
            logger.warning(
                "Can't find instance segmentation annotations in the dataset. "
                "The converted file will not contain instance segmentation."
            )
            logger.warning(e)
            self._instance_segmentation_def = None
            self._has_instance_seg = False

        captures = Captures(
            data_root=data_root, version=const.DEFAULT_PERCEPTION_VERSION
        )
        self._bbox_captures = captures.filter(self._bbox_def["id"])
        if self._instance_segmentation_def:
            self._instance_segmentation_captures = captures.filter(
                self._instance_segmentation_def["id"]
            )

    def execute(self, output, **kwargs):
        """Execute COCO Transformer

        Args:
            output (str): the output directory where converted dataset will
              be stored.
        """
        self._copy_images(output)
        self._process_instances(output)

    def _copy_images(self, output):
        image_to_folder = Path(output) / "images"
        image_to_folder.mkdir(parents=True, exist_ok=True)
        for _, row in self._bbox_captures.iterrows():
            image_from = self._data_root / row["filename"]
            if not image_from.exists():
                continue
            capture_id = uuid_to_int(row["id"])
            image_to = image_to_folder / f"camera_{capture_id}.png"
            shutil.copy(str(image_from), str(image_to))

    def _process_instances(self, output):
        output = Path(output) / "annotations"
        output.mkdir(parents=True, exist_ok=True)
        instances = {
            "info": {"description": "COCO compatible Synthetic Dataset"},
            "licences": [{"url": "", "id": 1, "name": "default"}],
            "images": self._images(),
            "annotations": self._annotations(),
            "categories": self._categories(),
        }
        output_file = output / "instances.json"
        with open(output_file, "w") as out:
            json.dump(instances, out)

    def _images(self):
        images = []
        for _, row in self._bbox_captures.iterrows():
            image_file = self._data_root / row["filename"]
            if not image_file.exists():
                continue
            with Image.open(image_file) as im:
                width, height = im.size
            capture_id = uuid_to_int(row["id"])
            record = {
                "file_name": f"camera_{capture_id}.png",
                "height": height,
                "width": width,
                "id": capture_id,
            }
            images.append(record)

        return images

    @staticmethod
    def _compute_segmentation(instance_id, seg_instances, seg_img):
        segmentation = []
        for ins in seg_instances:
            if instance_id == ins["instance_id"]:
                ins_color = ins["color"]
                if np.shape(seg_img)[-1] == 4:
                    ins_color = (
                        ins_color["r"],
                        ins_color["g"],
                        ins_color["b"],
                        ins_color["a"],
                    )
                else:
                    ins_color = (ins_color["r"], ins_color["g"], ins_color["b"])

                ins_mask = (seg_img == ins_color).prod(axis=-1).astype(np.uint8)
                segmentation = mask_to_rle(np.asfortranarray(ins_mask))
                segmentation["counts"] = segmentation["counts"].decode()

        return segmentation

    def _get_instance_seg_img(self, seg_row):
        file_path = (
            self._data_root / seg_row["annotation.filename"].to_list()[0]
        )

        with Image.open(file_path) as img:
            w, h = img.size
            if np.shape(img)[-1] == 4:
                img = np.array(img.getdata(), dtype=np.uint8).reshape(h, w, 4)
            else:
                img = np.array(img.getdata(), dtype=np.uint8).reshape(h, w, 3)

        return img

    def _annotations(self):
        annotations = []
        for _, row in self._bbox_captures.iterrows():
            image_id = uuid_to_int(row["id"])
            if self._has_instance_seg:
                seg_row = self._instance_segmentation_captures.loc[
                    self._instance_segmentation_captures["id"] == str(row["id"])
                ]
                seg_instances = seg_row["annotation.values"].to_list()[0]
                seg_img = self._get_instance_seg_img(seg_row)

            for ann in row["annotation.values"]:
                instance_id = ann["instance_id"]
                x = ann["x"]
                y = ann["y"]
                w = ann["width"]
                h = ann["height"]
                area = float(w) * float(h)

                if self._has_instance_seg:
                    segmentation = self._compute_segmentation(
                        instance_id, seg_instances, seg_img
                    )
                else:
                    segmentation = []

                record = {
                    "segmentation": segmentation,
                    "area": area,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [x, y, w, h],
                    "category_id": ann["label_id"],
                    "id": uuid_to_int(row["annotation.id"])
                    | uuid_to_int(ann["instance_id"]),
                }
                annotations.append(record)

        return annotations

    def _categories(self):
        categories = []
        for r in self._bbox_def["spec"]:
            record = {
                "id": r["label_id"],
                "name": r["label_name"],
                "supercategory": "default",
            }
            categories.append(record)

        return categories


class COCOKeypointsTransformer(DatasetTransformer, format="COCO-Keypoints"):
    """Convert Synthetic dataset to COCO format.
    This transformer convert Synthetic dataset into annotations
    in person keypoint format
    (e.g. person_keypoints_train2017.json, person_keypoints_val2017.json).
    Note: We assume "valid images" in the COCO dataset must contain at least one
    bounding box annotation. Therefore, all images that contain no bounding
    boxes will be dropped. Instance segmentation are considered optional
    in the converted dataset as some synthetic dataset might be generated
    without it.
    Args:
        data_root (str): root directory of the dataset
    """

    # The annotation_definition.name is not a reliable way to know the type
    # of annotation definition. This will be improved once the perception
    # package introduced the annotation definition type in the future.
    BBOX_NAME = r"^(?:2[dD]\s)?bounding\sbox$"
    KPT_NAME = r"^(?:2[dD]\s)?keypoints$"
    INSTANCE_SEGMENTATION_NAME = r"^instance\ssegmentation$"

    def __init__(self, data_root):
        self._data_root = Path(data_root)
        self._has_instance_seg = False

        ann_def = AnnotationDefinitions(
            data_root, version=const.DEFAULT_PERCEPTION_VERSION
        )
        self._bbox_def = ann_def.find_by_name(self.BBOX_NAME)
        self._kpt_def = ann_def.find_by_name(self.KPT_NAME)
        try:
            self._instance_segmentation_def = ann_def.find_by_name(
                self.INSTANCE_SEGMENTATION_NAME
            )
            self._has_instance_seg = True
        except NoRecordError as e:
            logger.warning(
                "Can't find instance segmentation annotations in the dataset. "
                "The converted file will not contain instance segmentation."
            )
            logger.warning(e)
            self._instance_segmentation_def = None

        captures = Captures(
            data_root=data_root, version=const.DEFAULT_PERCEPTION_VERSION
        )
        self._bbox_captures = captures.filter(self._bbox_def["id"])
        self._kpt_captures = captures.filter(self._kpt_def["id"])
        if self._instance_segmentation_def:
            self._instance_segmentation_captures = captures.filter(
                self._instance_segmentation_def["id"]
            )

    def execute(self, output, **kwargs):
        """Execute COCO Transformer
        Args:
            output (str): the output directory where converted dataset will
              be stored.
        """
        self._copy_images(output)
        self._process_instances(output)

    def _copy_images(self, output):
        image_to_folder = Path(output) / "images"
        image_to_folder.mkdir(parents=True, exist_ok=True)
        for _, row in self._bbox_captures.iterrows():
            image_from = self._data_root / row["filename"]
            if not image_from.exists():
                continue
            capture_id = uuid_to_int(row["id"])
            image_to = image_to_folder / f"camera_{capture_id}.png"
            shutil.copy(str(image_from), str(image_to))

    def _process_instances(self, output):
        output = Path(output) / "annotations"
        output.mkdir(parents=True, exist_ok=True)
        instances = {
            "info": {"description": "COCO compatible Synthetic Dataset"},
            "licences": [{"url": "", "id": 1, "name": "default"}],
            "images": self._images(),
            "annotations": self._annotations_fast(),
            "categories": self._categories(),
        }
        output_file = output / "keypoints.json"
        with open(output_file, "w") as out:
            json.dump(instances, out)

    def _images(self):
        images = []
        for _, row in self._bbox_captures.iterrows():
            image_file = self._data_root / row["filename"]
            if not image_file.exists():
                continue
            with Image.open(image_file) as im:
                width, height = im.size
            capture_id = uuid_to_int(row["id"])
            record = {
                "file_name": f"camera_{capture_id}.png",
                "height": height,
                "width": width,
                "id": capture_id,
            }
            images.append(record)

        return images

    def _get_instance_seg_img(self, seg_row):
        file_path = (
            self._data_root / seg_row["annotation.filename"]
        )

        with Image.open(file_path) as img:
            w, h = img.size
            if np.shape(img)[-1] == 4:
                img = np.array(img.getdata(), dtype=np.uint8).reshape(h, w, 4)
            else:
                img = np.array(img.getdata(), dtype=np.uint8).reshape(h, w, 3)

        return img

    @staticmethod
    def _binary_mask_to_polygon(binary_mask, tolerance=0):
        """Converts a binary mask to COCO polygon representation
        Args:
            binary_mask: a 2D binary numpy array where '1's represent the object
            tolerance: Maximum distance from original points of polygon to approximated
                polygonal chain. If tolerance is 0, the original coordinate array is returned.
        """
        polygons = []
        # pad mask to close contours of shapes which start and end at an edge
        padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
        contours = measure.find_contours(padded_binary_mask, 0.5)
        contours = np.subtract(contours, 1)
        for contour in contours:
            # contour = close_contour(contour)
            if not np.array_equal(contour[0], contour[-1]):
                contour = np.vstack((contour, contour[0]))

            contour = measure.approximate_polygon(contour, tolerance)
            if len(contour) < 3:
                continue
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().astype(np.uint8).tolist()
            # after padding and subtracting 1 we may get -0.5 points in our segmentation
            segmentation = [0 if i < 0 else i for i in segmentation]
            polygons.append(segmentation)

        return polygons

    @staticmethod
    def _compute_segmentation(seg_instance, seg_img):
        seg_color = seg_instance["color"]
        if np.shape(seg_img)[-1] == 4:
            seg_color = (
                seg_color["r"],
                seg_color["g"],
                seg_color["b"],
                seg_color["a"],
            )
        else:
            seg_color = (seg_color["r"], seg_color["g"], seg_color["b"])

        ins_mask = (seg_img == seg_color).prod(axis=-1).astype(np.uint8)
        segs = COCOKeypointsTransformer._binary_mask_to_polygon(ins_mask, tolerance=10)

        return segs

    def _annotations(self):
        annotations = []
        for [_, row_bb], [_, row_kpt], [_, row_seg] in tqdm(zip(
                self._bbox_captures.iterrows(),
                self._kpt_captures.iterrows(),
                self._instance_segmentation_captures.iterrows())
        ):
            image_id = uuid_to_int(row_bb["id"])
            # seg_row = self._instance_segmentation_captures.loc[
            #     self._instance_segmentation_captures["id"] == str(row_bb["id"])
            # ]
            # seg_instances = seg_row["annotation.values"].to_list()[0]
            seg_img = self._get_instance_seg_img(row_seg)

            for ann_bb, ann_kpt, ann_seg in zip(
                row_bb["annotation.values"], row_kpt["annotation.values"], row_seg["annotation.values"]
            ):
                # --- bbox ---
                instance_id = ann_bb["instance_id"]
                x = ann_bb["x"]
                y = ann_bb["y"]
                w = ann_bb["width"]
                h = ann_bb["height"]
                area = float(w) * float(h)
                segmentation = self._compute_segmentation(
                    ann_seg, seg_img
                )
                if len(segmentation) == 0:
                    continue

                # -- kpt ---
                keypoints_vals = []
                num_keypoints = 0
                for kpt in ann_kpt["keypoints"]:
                    keypoints_vals.append(
                        [
                            int(np.floor(kpt["x"])),
                            int(np.floor(kpt["y"])),
                            kpt["state"],
                        ]
                    )
                    if int(kpt["state"]) != 0:
                        num_keypoints += 1

                keypoints_vals = [
                    item for sublist in keypoints_vals for item in sublist
                ]

                record = {
                    "segmentation": segmentation,
                    "area": area,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [x, y, w, h],
                    "keypoints": keypoints_vals,
                    "num_keypoints": num_keypoints,
                    "category_id": ann_bb["label_id"],
                    "id": uuid_to_int(row_bb["annotation.id"])
                    | uuid_to_int(ann_bb["instance_id"]),
                }
                annotations.append(record)

        return annotations

    def _annotations_fast(self):
        annotations = {}
        segmentation_annotations = []
        for _, row_kpt in tqdm(self._kpt_captures.iterrows()):
            data_root = Path(self._data_root)
            row_seg = row_kpt["annotations"][0]
            row_bb = row_kpt["annotations"][1]
            assert row_kpt.annotations[0]["id"].startswith("instance"), "Assert failed, should start with 'instance'"
            assert row_kpt.annotations[1]["id"].startswith("bounding"), "Assert failed should start with 'bounding'"

            image_id = uuid_to_int(row_bb["id"])
            seg_img_path = self._data_root / row_seg["filename"]

            row_bb = pd.Series(row_bb)
            row_kpt = pd.Series(row_kpt['annotation.values'])
            row_seg = pd.Series(row_seg)

            rows_merged = pd.merge(row_bb, row_kpt, on='instance_id', how='inner')
            rows_merged = pd.merge(rows_merged, row_seg, on='instance_id', how='inner')

            for i, row in rows_merged.iterrows():
                x = row['x']
                y = row['y']
                w = row['width']
                h = row['height']
                area = float(w) * float(h)
                # segmentation = COCOKeypointsTransformer._compute_segmentation(
                #     row, seg_img
                # )
                seg_color = row['color']
                # -- kpt ---
                keypoints_vals = []
                num_keypoints = 0
                for kpt in row["keypoints"]:
                    keypoints_vals.append(
                        [
                            int(np.floor(kpt["x"])),
                            int(np.floor(kpt["y"])),
                            kpt["state"],
                        ]
                    )
                    if int(kpt["state"]) != 0:
                        num_keypoints += 1

                keypoints_vals = [
                    item for sublist in keypoints_vals for item in sublist
                ]
                rec_id = i
                record = {
                    # "segmentation": segmentation,
                    "area": area,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [x, y, w, h],
                    "keypoints": keypoints_vals,
                    "num_keypoints": num_keypoints,
                    "category_id": row["label_id"],
                    "id": rec_id,
                }
                seg_ann = {
                    'rec_id': rec_id,
                    'color': seg_color,
                    'img_path': seg_img_path.as_posix(),
                }
                segmentation_annotations.append(seg_ann)
                annotations[rec_id] = record

        seg_annotations = process_pool.imap_unordered(compute_segmentation, segmentation_annotations, chunksize=32)

        for seg_ann in seg_annotations:
            if len(seg_ann["segs"]) == 0:
                del annotations[seg_ann["rec_id"]]
                continue
            annotations["segmentation"] = seg_ann["segs"]

        return annotations.values()

    def _categories(self):
        categories = []
        key_points = []
        skeleton = []

        for kp in self._kpt_def["spec"][0]["key_points"]:
            key_points.append(kp["label"])

        for sk in self._kpt_def["spec"][0]["skeleton"]:
            skeleton.append([sk["joint1"] + 1, sk["joint2"] + 1])

        for r in self._bbox_def["spec"]:
            record = {
                "id": r["label_id"],
                "name": r["label_name"],
                "supercategory": "default",
                "keypoints": key_points,
                "skeleton": skeleton,
            }
            categories.append(record)

        return categories


def compute_segmentation(ann: dict):
    seg_color = ann["color"]
    img_path = ann["img_path"]

    with Image.open(img_path) as seg_img:
        w, h = seg_img.size
        if np.shape(seg_img)[-1] == 4:
            seg_img = np.array(seg_img.getdata(), dtype=np.uint8).reshape(h, w, 4)
        else:
            seg_img = np.array(seg_img.getdata(), dtype=np.uint8).reshape(h, w, 3)

        if np.shape(seg_img)[-1] == 4:
            seg_color = (
                seg_color["r"],
                seg_color["g"],
                seg_color["b"],
                seg_color["a"],
            )
        else:
            seg_color = (seg_color["r"], seg_color["g"], seg_color["b"])

        ins_mask = (seg_img == seg_color).prod(axis=-1).astype(np.uint8)

    segs = COCOKeypointsTransformer._binary_mask_to_polygon(ins_mask, tolerance=10)
    return {"rec_id": ann["rec_id"], "segs": segs}
