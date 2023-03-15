python -m torch.distributed.launch --nproc_per_node 8 train.py --data data/coco_kpts.yaml --cfg cfg/yolov7-tiny-pose.yaml --weights weights/yolov7-tiny.pt --batch-size 128 --img 256 --kpt-label --sync-bn --device 0 --name yolov7-tiny-pose --hyp data/hyp.pose.yaml