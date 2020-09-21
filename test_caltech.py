import argparse

from models_2 import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import matplotlib.pyplot as plt
import numpy as np


def detect(save_img=False):
    imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    save_txt = True

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    '''
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    '''
    # Initialize model
    model = Darknet(opt.cfg, imgsz).to(device)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Eval mode
    model.eval()

    # Fuse Conv2d + BatchNorm2d layers 提升模型速度。
    # model.fuse()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    save_img = True
    for j in range(6,11):
        image_fold = 'G:/yolov3-master/data/caltech/code/data-USA/images/set%02d' % j
        for root, dirs, files in os.walk(image_fold):
            for dir_name in dirs:
                source = 'G:/yolov3-master/data/caltech/code/data-USA/images/set%02d/%s' % (j, dir_name)

                dataset = LoadImages(source, img_size=imgsz)    # input file/folder

                # Get names and colors
                names = load_classes(opt.names)
                colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

                # Run inference
                t0 = time.time()
                img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img

                _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
                for path, img, im0s, vid_cap in dataset:
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # Inference
                    t1 = torch_utils.time_synchronized()
                    pred = model(img, augment=opt.augment)[0]
                    t2 = torch_utils.time_synchronized()

                    # Apply NMS
                    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                               multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

                    # Process detections
                    for i, det in enumerate(pred):  # detections for image i
                        if webcam:  # batch_size >= 1
                            p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                        else:
                            p, s, im0 = path, '', im0s

                        out_path = 'G:/yolov3-master/data/caltech/code/res/my_method/set%02d/%s' % (j, dir_name)
                        save_path = str(Path(out_path) / Path(p).name)

                        s += '%gx%g ' % img.shape[2:]  # print string
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh

                        if det is not None and len(det):
                            # Rescale boxes from imgsz to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                            # Print results
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += '%g %ss, ' % (n, names[int(c)])  # add to string

                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                # 每个都有个空的
                                with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                                    file.write('')  # label format
                                if cls != 0:
                                    continue
                                else:
                                # Write to file
                                    # print(cls)
                                    xywh = xyxy2xywh2(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()  # normalized xywh
                                    # save .txt
                                    with open(save_path[:save_path.rfind('.')] + '.txt', 'a+') as file:
                                        file.write(('%g ' * 5 + '\n') % (*xywh, conf))  # label format
                        else:
                            with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                                file.write('')  # label format


                        # Print time (inference + NMS)
                        print('%sDone. (%.3fs)' % (s, t2 - t1))

                if save_txt:
                    print('Results saved to %s' % os.getcwd() + os.sep + out)

                print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-ghost-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/caltech.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/ghost-eca-spp.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.005, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file
    print(opt)

    with torch.no_grad():
        detect()

