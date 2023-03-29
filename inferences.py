import argparse
import logging
import os
from os.path import join, basename, splitext

import cv2
from tqdm import tqdm
from glob import glob
import paddle.inference as paddle_infer
import numpy as np
from paddle.inference import PrecisionType

from utils import preprocess, draw_bbox


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s -%(module)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %p",
    level=logging.INFO,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file", type=str, default="yolov7_tiny_300e_coco/model.pdmodel"
    )
    parser.add_argument(
        "--params_file",
        type=str,
        default="yolov7_tiny_300e_coco/model.pdiparams",
    )
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--img_file", type=str, default="demo.png")
    parser.add_argument(
        "--run_mode",
        type=str,
        default="trt_fp16",
        help="Run_mode which can be: trt_fp32, trt_fp16, trt_int8 and gpu_fp16.",
    )
    return parser.parse_args()


def read_img(args, img_file):
    img = cv2.imread(img_file)
    h, w = img.shape[:2]  # get height and width
    if h == w:  # if image is square, no need to add gray area
        resized = cv2.resize(img, (640, 640), cv2.INTER_AREA)  # resize to 640*640
    else:  # if image is not square, add gray area
        dif = max(h, w)  # get maximum dimension
        interpolation = (
            cv2.INTER_AREA if dif > 640 else cv2.INTER_CUBIC
        )  # choose interpolation method based on dif
        x_pos = int((dif - w) / 2.0)  # get x position of image center
        y_pos = int((dif - h) / 2.0)  # get y position of image center
        if len(img.shape) == 3:  # if image has 3 channels (BGR)
            mask = np.zeros(
                (dif, dif, 3), dtype=img.dtype
            )  # create a black mask with 3 channels
            mask[y_pos : y_pos + h, x_pos : x_pos + w, :] = img[
                :, :, :
            ]  # put the image on the mask
        else:  # if image has 1 channel (grayscale)
            mask = np.zeros(
                (dif, dif), dtype=img.dtype
            )  # create a black mask with 1 channel
            mask[y_pos : y_pos + h, x_pos : x_pos + w] = img[
                :, :
            ]  # put the image on the mask
        resized = cv2.resize(
            mask, (640, 640), interpolation
        )  # resize the mask to 640*640
    resized = np.transpose(resized, (2, 0, 1))
    resized = np.expand_dims(resized, axis=0)
    return resized.astype("float32")


def single_pred(
    args, input_names, predictor, data, input_handle_img, input_handle_scale_factor
):
    # img = read_img(args,args.img_file)
    # img = cv2.imread(args.img_file)
    # img,scale_factor = preprocess(img,640)

    # data = [img,scale_factor]

    # for i,(name) in enumerate(input_names):
    #     input_handle = predictor.get_input_handle(name)
    #     input_handle.reshape([1, 3, 640, 640]) if i == 0 else input_handle.reshape([1, 1, 2])
    #     input_handle.copy_from_cpu(data[i].copy())
    input_handle_img.copy_from_cpu(data[0])
    input_handle_scale_factor.copy_from_cpu(data[1])
    # input_handle_shape.copy_from_cpu(np.array([640,640]).reshape((1,2)).astype(np.float32))
    

    predictor.run()

    results = []
    # get out data from output tensor
    output_names = predictor.get_output_names()

    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)
    tqdm.write(f"results:{results}")


def main():
    args = get_args()
    logging.info(f"args:{args}")

    logging.info(f"creating configs and predictor ...")

    config = paddle_infer.Config(args.model_file, args.params_file)

    config.enable_use_gpu(1000, 0)
    # # config.EnableProfile()
    # config.enable_memory_optim()
    # # config.enable_use_gpu(500, 0)
    # # config.enable_use_gpu(500, 0)
    # print("是否启用了GPU:")
    # print(config.use_gpu())

    if args.run_mode == "trt_fp32":
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=5,
            precision_mode=PrecisionType.Float32,
            use_static=False,
            use_calib_mode=False,
        )
    elif args.run_mode == "trt_fp16":
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=5,
            precision_mode=PrecisionType.Half,
            use_static=False,
            use_calib_mode=False,
        )
    elif args.run_mode == "trt_int8":
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=5,
            precision_mode=PrecisionType.Int8,
            use_static=False,
            use_calib_mode=True,
        )

    predictor = paddle_infer.create_predictor(config)

    # logging.info(f"input_names: \n")
    input_names = predictor.get_input_names()
    logging.info(f"input_names:{input_names}")

    input_handle_img = predictor.get_input_handle("image")
    input_handle_img.reshape([1, 3, 640, 640])
    input_handle_scale_factor = predictor.get_input_handle("scale_factor")
    input_handle_scale_factor.reshape([1, 1, 2])
    # input_handel_shape = predictor.get_input_handle("im_shape")
    # # reshape input for im_shape
    # input_handel_shape.reshape(np.array([640,640]).reshape((1,2)).astype(np.float32).shape)
    

    imgs_paths = glob(join("test_imgs", "*.jpg"))[:20]
    datas = [preprocess(cv2.imread(img_path), 640) for img_path in imgs_paths]
    i = 0
    for img_path in tqdm(imgs_paths):
        single_pred(
            args,
            input_names,
            predictor,
            datas[i],
            input_handle_img,
            input_handle_scale_factor,
            # input_handel_shape,
        )
        i += 1


if __name__ == "__main__":
    main()
