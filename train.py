"""
Code to train and evaluate object detection models on a data set using the
detectron2 framework.

Author: Hassan-Roland Nasser
Email: roland.nasser@agroscope.admin.ch
Date: May 4th, 2023
"""

###############################################################################
# IMPORTS
###############################################################################
import torch
import torchvision
import numpy as np
import pandas as pd
import os, json, cv2, shutil, string, pytz
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import random
import json
import re
import mlflow
from PIL import Image
from io import BytesIO
from pylab import *
from datetime import datetime 
import argparse

from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.utils.logger import setup_logger, log_every_n_seconds
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultTrainer, HookBase, DefaultPredictor
from detectron2.config import get_cfg

from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader, MetadataCatalog
from detectron2.data import transforms as T
from detectron2.data import DatasetCatalog, DatasetMapper
from detectron2.data import get_detection_dataset_dicts
from detectron2.data import build_detection_train_loader

# Set up logging
setup_logger()
###############################################################################
# CUDA CAPABILITIES and TORCH VERSIONS
###############################################################################
print('Torch Version: {}'.format(torch.__version__))
print('TorchVision Version: {}'.format(torchvision.__version__))

device = torch.device('cuda')
print('Device Properties: {}'.format(torch.cuda.get_device_properties(device)))
print('Memory Allocated: {}'.format(torch.cuda.memory_allocated(device)))
print('Memory Allocated: {}'.format(torch.cuda.memory_reserved(device)))

if torch.cuda.is_available():
    print('Module torch.cuda available from {}'.format(torch.cuda))


###############################################################################
# PARSING ARGUMENTS
###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--arch', required=True, help='Model architecture')
parser.add_argument('--ipb', type=int, required=True, help='Images per batch')
parser.add_argument('--iter', type=int, required=True, help='# of iterations')
args = parser.parse_args()
print('Model architecture:', args.arch)
print('Images per batch:', args.ipb)
print('Number of iterations:', args.iter)

###############################################################################
# BENCHMARK CONFIGURATIONS
###############################################################################
archs = args.arch
module_arch = 'COCO-Detection/' + args.arch
IMS_PER_BATCH = args.ipb
MAX_ITER = args.iter

###############################################################################
# MODEL CONFIGURATIONS
###############################################################################
Case_Study = 'chunks'
BATCH_SIZE_PER_IMAGE = 32

# Add other configurations for training (hyperparameters and augmentations)
RumexAugmentations = [
    T.RandomBrightness(0.9, 1.1),
    T.RandomContrast(0.5, 2),
    T.RandomSaturation(0.5, 2),
    T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
    T.RandomFlip(prob=0.5, horizontal=False, vertical=True)
    ]
BASE_LR = 0.00025  # pick a good LR
NUM_IMAGES_INF_LOG = 20

###############################################################################
# BENCHMARK CONFIGURATIONS
###############################################################################

exp_name = 'Detectron2_Chunks' + module_arch.split('/')[1].split('.')[0]
dataset_name = '20220823_HaldenSued_S_10_F_50_O_stra_ID1'
output_folder = 'results_A100D-2-20C/' + module_arch.split('/')[-1].split('.')[0] + '_' + \
 str(Case_Study) + '_IPB_' + str(IMS_PER_BATCH) + '_MI_' + str(MAX_ITER)
mount_point = './images'
dataset_dir = os.path.join(mount_point, dataset_name)
if Case_Study == 'chunks':
    image_dir = os.path.join(dataset_dir, '1_images/Chunks')
else:
    image_dir = os.path.join(dataset_dir, '1_images')

annotations_dir = os.path.join(dataset_dir, '3_annotations')

print('Available Datasets:')
DataSets = [d for d in os.listdir(mount_point) if os.path.isdir(os.path.join(mount_point, d))]


output_train = os.path.join(annotations_dir, 'train_chuncks.json')
output_val = os.path.join(annotations_dir, 'val_chuncks.json')
output_test = os.path.join(annotations_dir, 'test_chuncks.json')

###############################################################################
# DEFINING THE DATASETS
###############################################################################

classes = ["rumex_plant"]
partitions = ['train_chuncks', 'test_chuncks', 'val_chuncks']

for partition in partitions:
    if partition not in MetadataCatalog.list():
        ann_file = os.path.join(annotations_dir, partition + '.json')
        register_coco_instances(partition, {}, ann_file, image_dir)
    MetadataCatalog.get(partition).set(thing_classes=classes)

metadata = MetadataCatalog.get('rumex_train').set(thing_classes=classes)
for partition in partitions:
    print(MetadataCatalog.get(partition))


###############################################################################
# PREPARING MODEL CONFIGURATIONS
###############################################################################
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(module_arch))
cfg.DATASETS.TRAIN = ('train_chuncks',)
cfg.DATASETS.VAL = ('val_chuncks',)
cfg.DATASETS.TEST = ('test_chuncks',)
cfg.DATALOADER.NUM_WORKERS = 8
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(module_arch)
cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH  # Batch size of (default = 8)
cfg.SOLVER.BASE_LR = BASE_LR  # pick a good LR
cfg.SOLVER.MAX_ITER = MAX_ITER    # Number of iterations
cfg.SOLVER.STEPS = []         # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = BATCH_SIZE_PER_IMAGE
#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
if module_arch == "faster_rcnn":
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
elif module_arch == "retinanet":
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
cfg.OUTPUT_DIR = output_folder
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
print('The model will be saved in {}'.format(cfg.OUTPUT_DIR))
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

###############################################################################
# PREPARING TRAINGER
###############################################################################


class RumexTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=RumexAugmentations)
        return build_detection_train_loader(cfg, mapper=mapper)


###############################################################################
# TRAINING
###############################################################################
start_time_training = datetime.now()

trainer = RumexTrainer(cfg)
trainer.resume_or_load(resume=False)
results = trainer.train()

end_time_training = datetime.now()

elapsed_time = end_time_training - start_time_training
elapsed_minutes_training = int(elapsed_time.total_seconds() / 60)

###############################################################################
# PREPARING PREDICTOR
###############################################################################

experiment_directory = output_folder
cfg.MODEL.WEIGHTS = os.path.join(output_folder, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

###############################################################################
# PREDICTIONS
###############################################################################
start_time_inference = datetime.now()

evaluator = COCOEvaluator('test_chuncks', cfg, False, output_dir=output_folder)
val_loader = build_detection_test_loader(cfg, 'test_chuncks')
inf_results = inference_on_dataset(predictor.model, val_loader, evaluator)

end_time_inference = datetime.now()
elapsed_time = end_time_inference - start_time_inference
elapsed_minutes_inference = int(elapsed_time.total_seconds() / 60)


###############################################################################
# SAVING INFERENCE AND TIME RESULTS
###############################################################################

results_dict = {
    "AP": inf_results['bbox']['AP'],
    "AP50": inf_results['bbox']['AP50'],
    "AP75": inf_results['bbox']['AP75'],
    "APs": inf_results['bbox']['APs'],
    "APm": inf_results['bbox']['APm'],
    "APl": inf_results['bbox']['APl']}
metrics_file = os.path.join(output_folder, 'ap_metrics.json')
with open(metrics_file, 'w') as fp:
    json.dump(results_dict, fp)

# Create dictionary with results for time spent
time_spent = {
    "start_time_training": start_time_training.strftime("%Y-%m-%d %H:%M:%S"),
    "end_time_training": end_time_training.strftime("%Y-%m-%d %H:%M:%S"),
    "elapsed_minutes_training": elapsed_minutes_training,
    "start_time_inference": start_time_inference.strftime("%Y-%m-%d %H:%M:%S"),
    "end_time_inference": end_time_inference.strftime("%Y-%m-%d %H:%M:%S"),
    "elapsed_minutes_inference": elapsed_minutes_inference

}
time_file = os.path.join(output_folder, 'time_spent.json')
with open(time_file, 'w') as fp:
    json.dump(time_spent, fp)

###############################################################################
# SOME SAMPLE INFERENCE IMAGES
###############################################################################

inf_path = os.path.join(output_folder, 'sample_inference_images')

if not os.path.exists(inf_path):
    os.makedirs(inf_path)
    print(f"{inf_path} created!")
else:
    print(f"{inf_path} already exists!")

test_dict = get_detection_dataset_dicts('test_chuncks')

for d in test_dict[0:NUM_IMAGES_INF_LOG]:
    im = cv2.imread(d["file_name"])
    print(im[:, :, ::-1].shape)
    im_string = d["file_name"].split('/')[-1]
    print('Processing the image {}'.format(im_string))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (7, 5))
    ground_truth_visualizer = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.5)
    ground_truth_image = ground_truth_visualizer.draw_dataset_dict(d)
    ax1.imshow(cv2.cvtColor(ground_truth_image.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image \n'+ im_string , loc='center', fontsize=8)
    ax1.set_xticks([])
    ax1.set_yticks([])
    outputs = predictor(im)
    pred_visualizer = Visualizer(im[:, :, ::-1],
                    metadata=metadata,
                    scale=0.5)
    pred_image = pred_visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    ax2.imshow(cv2.cvtColor(pred_image.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    ax2.set_title('Model output', loc='center', fontsize=8)
    ax2.set_xticks([])
    ax2.set_yticks([])

    fig_name = im_string.split('.')[0] + '_TH_' +\
        str(int(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST*100)) + '.png'
    fig_name = os.path.join(inf_path, fig_name)
    savefig(fig_name, dpi = 300)

###############################################################################
# PLOTTING PERFORMANCE PARAMETERS
###############################################################################

json_file_name = output_folder + '/metrics.json'
parsed=[]
df = pd.read_json(json_file_name, lines=True )
fig = figure(figsize=(8,6), dpi=300)
y1 = fig.add_subplot(111)
y1.set_xlabel('Iterations')
y2 = y1.twinx()
# Plotting the total_loss
y1.plot(df['iteration'], df['total_loss'], color="darkslategray", linewidth=0.3,linestyle="-",label='total_loss')
# Plotting the box regression loss
y1.plot(df['iteration'], df['loss_box_reg'], color="green", linewidth=0.3,linestyle="-",label='loss_box_reg')
# Plotting the class loss
y1.plot(df['iteration'], df['loss_cls'], color="blue", linewidth=0.3,linestyle="-",label='loss_class')
# Plotting the loss_rpn_cls
y1.plot(df['iteration'], df['loss_rpn_cls'], color="orange", linewidth=0.3,linestyle="-",label='loss_rpn_cls')
# Plotting the false_negative
y1.plot(df['iteration'], df['loss_rpn_cls'], color="black", linewidth=0.3,linestyle="-",label='loss_rpn_cls')
# Plotting the loss_rpn_loc
y1.plot(df['iteration'], df['loss_rpn_loc'], color="goldenrod", linewidth=0.3,linestyle="-",label='loss_rpn_loc')
# Plotting the learning rate
y2.set_ylim(0,max(df['lr'])/0.8)
y2.plot(df['iteration'], df['lr'], color="purple", linewidth=1.0, linestyle="-",label='lr')
y2.set_ylabel('lr')
y2.legend(loc = 2)
y1.legend(loc = 1)
fig_name = output_folder + '/losses.png'
savefig(fig_name)


fig = figure(figsize=(8,6), dpi=300)
y1 = fig.add_subplot(111)
y1.set_xlabel('Iterations')
# Plotting the roi_head/num_bg_samples
y1.plot(df['iteration'], df['roi_head/num_bg_samples'], color="red", linewidth=0.3,linestyle="-",label='roi_head/num_bg_samples')
# Plotting the roi_head/num_fg_samples
y1.plot(df['iteration'], df['roi_head/num_fg_samples'], color="blue", linewidth=0.3,linestyle="-",label='roi_head/num_fg_samples')
# Plotting the false_negative
y1.plot(df['iteration'], df['rpn/num_neg_anchors'], color="black", linewidth=0.3,linestyle="-",label='rpn/num_neg_anchors')
# Plotting the false_negative
y1.plot(df['iteration'], df['rpn/num_pos_anchors'], color="orange", linewidth=0.3,linestyle="-",label='rpn/num_pos_anchors')
y1.legend(loc = 1)
fig_name = output_folder + '/numbers.png'
savefig(fig_name)

fig = figure(figsize=(8,6), dpi=300)
y1 = fig.add_subplot(111)
y1.set_xlabel('Iterations')
# Plotting the fast_rcnn/fg_cls_accuracy
y1.plot(df['iteration'], df['fast_rcnn/fg_cls_accuracy'], color="cornflowerblue", linewidth=0.3,linestyle="-",label='fast_rcnn/fg_cls_accuracy')
# Plotting the class accuracy
y1.plot(df['iteration'], df['fast_rcnn/cls_accuracy'], color="red", linewidth=0.3,linestyle="-",label='cls_accuracy')
# Plotting the false_negative
y1.plot(df['iteration'], df['fast_rcnn/false_negative'], color="purple", linewidth=0.3,linestyle="-",label='false_negative')
y1.legend(loc = 1)
fig_name = output_folder + '/accuracies.png'
savefig(fig_name)







