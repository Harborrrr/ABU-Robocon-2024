import glob
import argparse
import os.path as osp
from typing import Callable

import cv2
import numpy as np


# mscoco class names
CLASS_LABELS = ['blue','red','purple','basket']


def parse_arguments(desc):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i', '--input_path', dest='input_path', required=True, type=str,
                        help='Path to Input: Video File or Image file')
    parser.add_argument('-m', '--media_type', dest='media_type', default='image', type=str,
                        choices=('image', 'video'),
                        help='Type of Input: image, video. (default: %(default)s)')
    parser.add_argument('-t', '--threshold', dest='threshold', default=0.6, type=float,
                        help='Detection Threshold. (default: %(default)s)')
    parser.add_argument('--ox', '--onnx_path', dest='onnx_path', default="models/yolov5s.onnx", type=str,
                        help='Path to ONNX model. (default: %(default)s)')
    parser.add_argument('-o', '--output_dir', dest='output_dir', default='output', type=str,
                        help='Output directory. (default: %(default)s)')
    parser.add_argument('-c', '--num_classes', dest='num_classes', default=80, type=int,
                        help='Num of classes. (default: %(default)s)')

    return parser.parse_args()


class DataStreamer(object):

    """Iterable DataStreamer class for generating numpy arr images
    Generates orig image and pre-processed image

    For loading data into detectors
    """

    def __init__(self, src_path: str, media_type: str = "image", preprocess_func: Callable = None):
        """Init DataStreamer Obj

        src_path : str
            path to a single image/video or path to directory containing images
        media_type : str
            inference media_type "image" or "video"
        preprocess_func : Callable function
            preprocessesing function applied to PIL images
        """
        if media_type not in {'video', 'image'}:
            raise NotImplementedError(
                f"{media_type} not supported in streamer. Use video or image")
        self.img_path_list = []
        self.vid_path_list = []
        self.idx = 0
        self.media_type = media_type
        self.preprocess_func = preprocess_func

        if media_type == "video":
            if osp.isfile(src_path):
                self.vid_path_list.append(src_path)
                self.vcap = cv2.VideoCapture(src_path)
            elif osp.isdir(src_path):
                raise NotImplementedError(
                    f"dir iteration supported for video media_type. {src_path} must be a video file")
        elif media_type == "image":
            if osp.isfile(src_path):
                self.img_path_list.append(src_path)
            elif osp.isdir(src_path):
                img_exts = ['*.png', '*.PNG', '*.jpg', '*.jpeg']
                for ext in img_exts:
                    self.img_path_list.extend(
                        glob.glob(osp.join(src_path, ext)))

    def __iter__(self):
        return self

    def __next__(self):
        """Get next image or frame as numpy array

        """
        orig_img = None
        if self.media_type == 'image':
            if self.idx < len(self.img_path_list):
                orig_img = cv2.imread(self.img_path_list[self.idx])
                orig_img = orig_img[..., ::-1]
                self.idx += 1
        elif self.media_type == 'video':
            if self.idx < len(self.vid_path_list):
                ret, frame = self.vcap.read()
                if ret:
                    # orig_img = frame[..., ::-1]
                    orig_img = frame
                else:
                    self.idx += 1
        if orig_img is not None:
            proc_img = None
            if self.preprocess_func is not None:
                proc_img = self.preprocess_func(orig_img)
                proc_img = np.expand_dims(proc_img, axis=0)
            return np.array(orig_img), proc_img
        raise StopIteration
