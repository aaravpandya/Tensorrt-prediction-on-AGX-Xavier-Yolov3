
from __future__ import print_function
import common

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw
import cv2 as cv
from yolov3_to_onnx import download_file
from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES
import time
import sys
import os
import argparse
sys.path.insert(1, os.path.join(sys.path[0], ".."))

TRT_LOGGER = trt.Logger()

parser = argparse.ArgumentParser(description='Predict on video using Yolov3.')
parser.add_argument('-v', '--video', type=str,
                    help='Path to video.')


def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color='blue'):
    """Draw the bounding boxes on the original input image and return it.

    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """
    draw = ImageDraw.Draw(image_raw)

    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(
            x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(
            y_coord + height + 0.5).astype(int))

        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        draw.text((left, top - 12),
                  '{0} {1:.2f}'.format(all_categories[category], score), fill=bbox_color)

    return image_raw


def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 28  # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(
                    onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(
                onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def main():
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""
    args = parser.parse_args()
    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    onnx_file_path = 'yolov3.onnx'
    engine_file_path = "yolov3.trt"

    cam = cv.VideoCapture(args.video)
    # img = cv.imread("dog.jpg")

    input_resolution_yolov3_HW = (608, 608)

    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
    postprocessor_args = {"yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],
                          "yolo_anchors": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                                           (59, 119), (116, 90), (156, 198), (373, 326)],
                          "obj_threshold": 0.6,
                          "nms_threshold": 0.5,
                          "yolo_input_resolution": input_resolution_yolov3_HW}
    postprocessor = PostprocessYOLO(**postprocessor_args)
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        while True:
            _ret, img = cam.read()
            if(_ret is False):
                break
            image_raw, image = preprocessor.process_image(img)
            shape_orig_WH = image_raw.size
            output_shapes = [(1, 255, 19, 19),
                             (1, 255, 38, 38), (1, 255, 76, 76)]
            trt_outputs = []
            inputs[0].host = image
            trt_outputs = common.do_inference(
                context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            trt_outputs = [output.reshape(shape)
                           for output, shape in zip(trt_outputs, output_shapes)]
            boxes, classes, scores = postprocessor.process(
                trt_outputs, (shape_orig_WH))
            if(boxes is None):
                continue
            obj_detected_img = draw_bboxes(
                image_raw, boxes, scores, classes, ALL_CATEGORIES)
            det_img = np.array(obj_detected_img)
            cv.imshow("frame", det_img)
            cv.waitKey(5)


if __name__ == '__main__':
    main()
