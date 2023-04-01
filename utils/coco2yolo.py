#COCO 格式的数据集转化为 YOLO 格式的数据集
#--json_path 输入的json文件路径
#--save_path 保存的文件夹名字，默认为当前目录下的labels。
import os
import json
from tqdm import tqdm
import argparse
import numpy as np
parser = argparse.ArgumentParser()
#这里根据自己的json文件位置，换成自己的就行
parser.add_argument('--json_path', default='coco_kpts/annotations/instances_val2017.json',type=str, help="input: coco format(json)")
#这里设置.txt文件保存位置
parser.add_argument('--save_path', default='coco_kpts/labels/val2017', type=str, help="specify where to save the output dir of labels")
parser.add_argument('--split', default='train', type=str,
help="specify train/val split")
arg = parser.parse_args()
def convert(size, box, keypoints):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
#round函数确定(xmin, ymin, xmax, ymax)的小数位数
    x = round(x * dw, 6)
    w = round(w * dw, 6)
    y = round(y * dh, 6)
    h = round(h * dh, 6)
    kpt_sets = []
    # normalize [keypoints, foot_kpts, face_kpts, lefthand_kpts, righthand_kpts]
    for kpt_set in keypoints:
        kpt_set.copy()
        kpt_set[::3] = np.array(kpt_set[::3]) * dw
        kpt_set[1::3] = np.array(kpt_set[1::3]) * dh
        kpt_sets.append(kpt_set)
    return (x, y, w, h), list(np.concatenate(kpt_sets).flat)
if __name__ == '__main__':
    json_file =   arg.json_path # COCO Object Instance 类型的标注
    ana_txt_save_path = arg.save_path  # 保存的路径
    split = arg.split
    data = json.load(open(json_file, 'r'))
    if not os.path.exists(ana_txt_save_path):
        os.makedirs(ana_txt_save_path)
    id_map = {} # coco数据集的id不连续！重新映射一下再输出！
    with open(os.path.join(ana_txt_save_path, 'classes.txt'), 'w') as f:
        # 写入classes.txt
        for i, category in enumerate(data['categories']):
            f.write(f"{category['name']}\n")
            id_map[category['id']] = i
    # print(id_map)
    #这里需要根据自己的需要，更改写入图像相对路径的文件位置。
    list_file = open(os.path.join(ana_txt_save_path, split+'.txt'), 'w')
    for img in tqdm(data['images']):
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]
        head, tail = os.path.splitext(filename)
        ana_txt_name = head + ".txt"  # 对应的txt名字，与jpg一致
        f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')
        for ann in data['annotations']:
            if ann['image_id'] == img_id:
                box, keypoints = convert((img_width, img_height), ann["bbox"], [ann["keypoints"], ann["foot_kpts"], ann["face_kpts"], ann["lefthand_kpts"], ann["righthand_kpts"]])
                line = (id_map[ann["category_id"]], *box, *keypoints)
                f_txt.write(('%g ' * len(line)).rstrip() % line + '\n')
        f_txt.close()
        #将图片的相对路径写入train2017或val2017的路径
        list_file.write('./images/%s/%s.jpg\n' %(split,head))
    list_file.close()
