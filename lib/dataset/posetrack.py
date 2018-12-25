# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

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
from nms.nms import oks_nms
from third_part.poseval.evaluate import evaluate as posetrack_evaluate


logger = logging.getLogger(__name__)


class PoseTrackDataset(JointsDataset):
    '''
    "keypoints": {
        0: "nose",
        1: "head_bottom",
        2: "head_top",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    },
	"skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    '''
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super(PoseTrackDataset, self).__init__(cfg, root, image_set, is_train, transform)
        assert image_set in {'train', 'val', 'test'}

        self.num_joints = 17
        self.flip_pairs = [[3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]
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
        self.all_images_dict, self.all_annotations_dict, self.seq_categories_dict = self._load_annotations()
        logger.info('=> num_images: {}'.format(len(self.all_images_dict)))

        self.db = self._get_db()
        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _load_annotations(self):
        anno_root = os.path.join(self.root, 'annotations', self.image_set)
        all_images_dict = {}
        all_annotations_dict = {}
        seq_categories_dict = {}
        for seq_anno_file in os.listdir(anno_root):
            seq_name = os.path.splitext(seq_anno_file)[0]
            with open(os.path.join(anno_root, seq_anno_file), 'r') as f:
                raw_annotation = json.load(f)
            images = raw_annotation['images']
            annotations = raw_annotation['annotations']

            # select images
            for image in images:
                image['seq_name'] = seq_name
                all_images_dict[image['id']] = image

            # index annotations by image_id
            for anno in annotations:
                image_id = anno['image_id']
                all_annotations_dict.setdefault(image_id, [])
                all_annotations_dict[image_id].append(anno)

            # save categories
            seq_categories_dict[seq_name] = raw_annotation['categories']

        return all_images_dict, all_annotations_dict, seq_categories_dict

    def _get_db(self):
        if self.use_gt_bbox:
            return self._get_gt_db()
        else:
            return self._load_coco_person_detection_results()

    def _get_gt_db(self):
        assert self.use_gt_bbox

        gt_db = []
        for image in self.all_images_dict.values():
            image_id = image['id']
            image_path = os.path.join(self.root, image['file_name'])

            annotations = self.all_annotations_dict.get(image_id, [])
            for anno in annotations:
                keypoints = np.asarray(anno['keypoints'], dtype=np.float).reshape(self.num_joints, 3)
                if not np.any(keypoints[:, 2] > 0):
                    continue

                keypoints_vis = np.zeros_like(keypoints)
                keypoints_vis[keypoints[:, 2] > 0, 0:2] = 1

                tlwh = anno['bbox']
                center, scale = self._box2cs(tlwh)

                gt_db.append({
                    'image': image_path,
                    'image_id': image_id,
                    'center': center,
                    'scale': scale,
                    'joints_3d': keypoints,
                    'joints_3d_vis': keypoints_vis,
                })
        return gt_db

    def _load_coco_person_detection_results(self):
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            logger.error('=> Load %s fail!' % self.bbox_file)
            return None

        logger.info('=> Total boxes: {}'.format(len(all_boxes)))

        kpt_db = []
        num_boxes = 0
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue

            image_id = det_res['image_id']
            image = self.all_images_dict[image_id]
            image_path = os.path.join(self.root, image['file_name'])
            box = det_res['bbox']
            score = det_res['score']

            if score < self.image_thre:
                continue

            num_boxes = num_boxes + 1

            center, scale = self._box2cs(box)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.ones(
                (self.num_joints, 3), dtype=np.float)
            kpt_db.append({
                'image': image_path,
                'image_id': image_id,
                'center': center,
                'scale': scale,
                'score': score,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
            })

        logger.info('=> Total boxes after fliter low score@{}: {}'.format(
            self.image_thre, num_boxes))
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

    # need double check this API and classes field
    def evaluate(self, cfg, preds, output_dir, all_boxes, img_path, image_ids, *args, **kwargs):
        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)

        # person x (keypoints)
        _kpts = []
        for idx, kpt in enumerate(preds):
            _kpts.append({
                'keypoints': kpt,
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': img_path[idx],
                'image_id': image_ids[idx]
            })
        # image x person x (keypoints)
        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image_id']].append(kpt)

        # rescoring and oks nms
        num_joints = self.num_joints
        in_vis_thre = self.in_vis_thre
        oks_thre = self.oks_thre
        oks_nmsed_kpts = {}
        for image_id, img_kpts in kpts.items():
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > in_vis_thre:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score
            keep = oks_nms([img_kpts[i] for i in range(len(img_kpts))], oks_thre)
            if len(keep) == 0:
                oks_nmsed_kpts[image_id] = img_kpts
            else:
                oks_nmsed_kpts[image_id] = [img_kpts[_keep] for _keep in keep]

        self._write_posetrack_keypoint_results(oks_nmsed_kpts, res_folder)
        if 'test' not in self.image_set:
            return self._do_python_keypoint_eval(res_folder)
        else:
            return {'Null': 0}, 0

    def _write_posetrack_keypoint_results(self, keypoints_dict, res_folder):
        all_results_dict = {}
        # write images
        for image_id, image in self.all_images_dict.items():
            seq_name = image['seq_name']
            all_results_dict.setdefault(seq_name, {
                'images': [],
                'annotations': [],
            })
            all_results_dict[seq_name]['images'].append(image)

        # write keypoints
        for image_id, image_kpts in keypoints_dict.items():
            image = self.all_images_dict[image_id]
            seq_name = image['seq_name']

            for kpt_dict in image_kpts:
                keypoints = np.asarray(kpt_dict['keypoints'], dtype=float)
                scores = np.copy(keypoints[:, 2])
                anno = {
                    'image_id': int(image_id),
                    'track_id': -1,
                    'keypoints': keypoints.reshape(-1).tolist(),
                    'scores': scores.tolist()
                }
                all_results_dict[seq_name]['annotations'].append(anno)

        # write categories
        for seq_name in all_results_dict.keys():
            all_results_dict[seq_name]['categories'] = self.seq_categories_dict[seq_name]

        # write results
        for seq_name, results in all_results_dict.items():
            res_file = os.path.join(res_folder, '{}.json'.format(seq_name))
            logger.info('=> Writing results json to %s' % res_file)
            with open(res_file, 'w') as f:
                json.dump(results, f)

    def _do_python_keypoint_eval(self, res_folder):
        anno_root = os.path.join(self.root, 'annotations', self.image_set)
        output_folder = os.path.join(os.path.dirname(res_folder), 'eval_output')
        name_values, perf_indicator = posetrack_evaluate(anno_root, res_folder, output_folder,
                                                         eval_pose=True, eval_tracking=False, save_per_seq=False)

        logger.info('=> coco eval results saved to %s' % output_folder)

        return name_values, perf_indicator
