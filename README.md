# yolov7-pose-whole-body
Implementation of "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors" combined with
"Whole-Body Human Pose Estimation in the Wild".

This repo seeks to combine the aforementioned papers/repos to add extra keypoints to yolo-pose models.

Pose estimation implimentation is based on [YOLO-Pose](https://arxiv.org/abs/2204.06806). 

## Pretrained models
[yolov7-tiny-pose](https://drive.google.com/drive/folders/14-k2wyG0P00PHlXbjGGG7IZjiq8vBQIy?usp=sharing)

``` shell
python train.py --data data/coco_kpts.yaml --cfg cfg/yolov7-tiny-pose.yaml --batch-size 64 --img 640 --kpt-label --sync-bn --device 0  --hyp data/hyp.pose.yaml --nkpt 133 --weights PATH_TO_PRETRAINED_WEIGHTS epochs 500
```
## Dataset preparation

[[Keypoints Labels of MS COCO 2017]](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-keypoints.zip)

COCO Whole-Body: https://github.com/jin-s13/COCO-WholeBody

Handy COCO to YOLO conversion script in `utils/coco2yolo.py`.

## Training

[yolov7-w6-person.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-person.pt)

``` shell
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train.py --data data/coco_kpts.yaml --cfg cfg/yolov7-w6-pose.yaml --weights weights/yolov7-w6-person.pt --batch-size 128 --img 960 --kpt-label --sync-bn --device 0,1,2,3,4,5,6,7 --name yolov7-w6-pose --hyp data/hyp.pose.yaml
```

## Deploy
TensorRT:[https://github.com/nanmi/yolov7-pose](https://github.com/nanmi/yolov7-pose)

## Testing

[yolov7-w6-pose.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt)

``` shell
python test.py --data data/coco_kpts.yaml --img 960 --conf 0.001 --iou 0.65 --weights yolov7-w6-pose.pt --kpt-label
```

## Citation

```
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/JUGGHM/OREPA_CVPR2022](https://github.com/JUGGHM/OREPA_CVPR2022)
* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)
* [https://github.com/jin-s13/COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody)

</details>
