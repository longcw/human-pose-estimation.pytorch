from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import pickle
from collections import defaultdict

# import json_tricks as json
import ujson as json
import numpy as np

from dataset.JointsDataset import JointsDataset
from .coco import COCODataset
from .mpii import MPIIDataset
from .posetrack import PoseTrackDataset


logger = logging.getLogger(__name__)


class MOTDataset(JointsDataset):

    def __init__(self, cfg, root, image_set, is_train, transform=None, pose_format='coco'):
        super(MOTDataset, self).__init__(cfg, root, image_set, is_train, transform)
        assert image_set in {'train', 'test'}

        format_dataset = {
            'coco': COCODataset,
            'mpii': MPIIDataset,
            'posetrack': PoseTrackDataset
        }[pose_format]
        self.num_joints = getattr(format_dataset, 'num_joints')
        self.flip_pairs = getattr(format_dataset, 'flip_pairs')

        self.parent_ids = None
        self.pixel_std = 200

        self.nms_thre = cfg.TEST.NMS_THRE
        self.image_thre = cfg.TEST.IMAGE_THRE
        self.oks_thre = cfg.TEST.OKS_THRE
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE
        self.bbox_file = cfg.TEST.COCO_BBOX_FILE
        self.use_gt_bbox = cfg.TEST.USE_GT_BBOX
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height

        # load annotations
        self.seq_images, self.seq_detections = self._load_dataset()
        logger.info('=> num_images: {}'.format(sum(len(images) for images in self.seq_images.values())))

        self.db = self._get_db()
        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _load_dataset(self):
        """Load image path and detections/gts"""
        seq_root = os.path.join(self.root, self.image_set)
        seq_names = os.listdir(seq_root)

        # read detections
        seq_detections = {}
        for seq in seq_names:
            det_file = os.path.join(seq_root, seq, 'det', 'det.txt')
            dets = self.read_mot_results(det_file, is_gt=False, is_ignore=False)
            seq_detections[seq] = dets

        # read frame names
        seq_images = {}
        for seq in seq_names:
            image_root = os.path.join(seq_root, seq, 'img1')
            image_names = [os.path.join(image_root, name) for name in sorted(os.listdir(image_root))]
            seq_images[seq] = image_names

        return seq_images, seq_detections

    def _get_db(self):
        if self.use_gt_bbox:
            raise NotImplementedError
        else:
            return self._load_coco_person_detection_results()

    def _load_coco_person_detection_results(self):
        logger.info('=> Total boxes: {}'.format(sum(
            sum(len(frame_dets) for frame_dets in dets.values()) for dets in self.seq_detections.values())))

        kpt_db = []
        for seq, images in self.seq_images.items():
            detections = self.seq_detections[seq]
            for image_file in images:
                frame_id = int(image_file[-10:-4])
                frame_dets = detections.get(frame_id, [])  # [tlwh, tid, score]

                for tlwh, track_id, score in frame_dets:
                    center, scale = self._box2cs(tlwh)
                    joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
                    joints_3d_vis = np.ones(
                        (self.num_joints, 3), dtype=np.float)
                    kpt_db.append({
                        'image': image_file,
                        'image_id': (seq, frame_id),
                        'bbox_tlwh': np.asarray(tlwh),
                        'center': center,
                        'scale': scale,
                        'score': score,
                        'joints_3d': joints_3d,
                        'joints_3d_vis': joints_3d_vis,
                    })

        return kpt_db

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    @staticmethod
    def read_mot_results(filename, is_gt, is_ignore):
        valid_labels = {1}
        ignore_labels = {2, 7, 8, 12}
        results_dict = dict()
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                for line in f.readlines():
                    linelist = line.split(',')
                    if len(linelist) < 7:
                        continue
                    fid = int(linelist[0])
                    if fid < 1:
                        continue
                    results_dict.setdefault(fid, list())

                    if is_gt:
                        if 'MOT16-' in filename or 'MOT17-' in filename:
                            label = int(float(linelist[7]))
                            mark = int(float(linelist[6]))
                            if mark == 0 or label not in valid_labels:
                                continue
                        score = 1
                    elif is_ignore:
                        if 'MOT16-' in filename or 'MOT17-' in filename:
                            label = int(float(linelist[7]))
                            vis_ratio = float(linelist[8])
                            if label not in ignore_labels and vis_ratio >= 0:
                                continue
                        else:
                            continue
                        score = 1
                    else:
                        score = float(linelist[6])

                    tlwh = tuple(map(float, linelist[2:6]))
                    target_id = int(linelist[1])

                    results_dict[fid].append((tlwh, target_id, score))

        return results_dict
