import sys
import os

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'MSMFormer'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets'))

import numpy as np
import cv2
import open3d as o3d
from tqdm import tqdm
from detectron2.data import MetadataCatalog, DatasetCatalog
from meanshiftformer.config import add_meanshiftformer_config
from datasets import OCIDDataset, OSDObject
from datasets.tabletop_dataset import TableTopDataset, getTabletopDataset
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.config import get_cfg
from tabletop_config import add_tabletop_config
from datasets.pushing_dataset import PushingDataset
from PIL import Image

from utils.evaluation import multilabel_metrics
# ignore some warnings
import warnings
import torch
from config import cfg
warnings.simplefilter("ignore", UserWarning)
from test_utils import test_dataset, test_sample, test_sample_crop, test_dataset_crop, Network_RGBD, test_sample_crop_nolabel, get_result_from_network
from utils.mask import visualize_segmentation
import imageio.v2 as imageio
from lib.datasets.load_OSD_UOAIS import inpaint_depth, normalize_depth, unnormalize_depth
import copy
dirname = os.path.dirname(__file__)
import json


def get_general_predictor(cfg_file, weight_path, input_image="RGBD_ADD"):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_meanshiftformer_config(cfg)
    cfg_file = cfg_file
    cfg.merge_from_file(cfg_file)
    add_tabletop_config(cfg)
    cfg.SOLVER.IMS_PER_BATCH = 1  #

    cfg.INPUT.INPUT_IMAGE = input_image
    if input_image == "RGBD_ADD":
        cfg.MODEL.USE_DEPTH = True
    else:
        cfg.MODEL.USE_DEPTH = False
    # arguments frequently tuned
    cfg.TEST.DETECTIONS_PER_IMAGE = 20
    weight_path = weight_path
    cfg.MODEL.WEIGHTS = weight_path
    predictor = Network_RGBD(cfg)
    return predictor, cfg


# def get_predictor_crop(cfg_file=cfg_file_MSMFormer_crop, weight_path=weight_path_MSMFormer_crop, input_image="RGBD_ADD"):
#     return get_general_predictor(cfg_file, weight_path, input_image=input_image)

# set datasets
# use_my_dataset = True
# for d in ["train", "test"]:
#     if use_my_dataset:
#         DatasetCatalog.register("tabletop_object_" + d, lambda d=d: TableTopDataset(d))
#     else:
#         DatasetCatalog.register("tabletop_object_" + d, lambda d=d: getTabletopDataset(d))

def process_label(foreground_labels):
    """ Process foreground_labels
            - Map the foreground_labels to {0, 1, ..., K-1}

        @param foreground_labels: a [H x W] numpy array of labels

        @return: foreground_labels
    """
    # Find the unique (nonnegative) foreground_labels, map them to {0, ..., K-1}
    unique_nonnegative_indices = np.unique(foreground_labels)
    mapped_labels = foreground_labels.copy()
    for k in range(unique_nonnegative_indices.shape[0]):
        mapped_labels[foreground_labels == unique_nonnegative_indices[k]] = k
    foreground_labels = mapped_labels
    return foreground_labels

def read_file(file_path):
    f = open(file_path,"r")
    lines = f.readlines()
    data_list = []
    for line in lines:
        data_list.append(line.strip('\n'))
    return data_list

class CameraInfo():
    """ Camera intrisics for point cloud creation. """

    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale


def create_point_cloud_from_depth_image(depth, camera, organized=True):
    """ Generate point cloud using depth image only.

        Input:
            depth: [numpy.ndarray, (H,W), numpy.float32]
                depth image
            camera: [CameraInfo]
                camera intrinsics
            organized: bool
                whether to keep the cloud in image shape (H,W,3)

        Output:
            cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    assert (depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud


def adjust_intrinsics(K, original_size, new_size):
    """
    Adjust the intrinsic matrix K after resizing an image.

    Parameters:
    - K: Original 3x3 intrinsic matrix.
    - original_size: Tuple (original_height, original_width) of the original image size.
    - new_size: Tuple (new_height, new_width) of the resized image size.

    Returns:
    - K_new: Adjusted 3x3 intrinsic matrix.
    """
    original_height, original_width = original_size
    new_height, new_width = new_size

    # Calculate scaling factors
    scale_x = new_width / original_width
    scale_y = new_height / original_height

    # Adjust the intrinsic matrix
    K_new = np.array([
        [scale_x * K[0, 0], 0, scale_x * K[0, 2]],
        [0, scale_y * K[1, 1], scale_y * K[1, 2]],
        [0, 0, 1]
    ])

    return K_new


eps = 1e-8
dataset = 'OCID' # 'OCID' 'PhoCAL' 'OSD' 'Housecat6D'
# # RGB
cfg_file_MSMFormer = os.path.join(dirname, '../../MSMFormer/configs/UOAIS_ResNet50.yaml')
weight_path_MSMFormer = os.path.join(dirname, "../../data/checkpoints/uoais/OCID_MSMFormer_RGB_UOAIS_SIM.pth")

# # RGBD
# cfg_file_MSMFormer = os.path.join(dirname, '../../MSMFormer/configs/UOAIS_UCN.yaml')
# weight_path_MSMFormer = os.path.join(dirname, "../../data/checkpoints/uoais/OSD_RGBD_MSMFormer_UOAIS_SIM.pth")
# weight_path_MSMFormer = os.path.join(dirname, "../../data/checkpoints/uoais/OCID_RGBD_MSMFormer_UOAIS_SIM.pth")

# cfg_file_MSMFormer = os.path.join(dirname, '../../MSMFormer/configs/mixture_UCN.yaml')
# weight_path_MSMFormer = os.path.join(dirname, "../../data/checkpoints/tabletop_rgbd/norm_RGBD_pretrained.pth")

# refine config
cfg_file_MSMFormer_crop = os.path.join(dirname, "../../MSMFormer/configs/crop_mixture_ResNet50.yaml")
weight_path_MSMFormer_crop = os.path.join(dirname, "../../data/checkpoints/tabletop_rgb/crop_RGB_pretrained.pth")

# cfg_file_MSMFormer_crop = os.path.join(dirname, "../../MSMFormer/configs/crop_mixture_UCN.yaml")
# weight_path_MSMFormer_crop = os.path.join(dirname, "../../data/checkpoints/tabletop_rgbd/crop_RGBD_pretrained.pth")

# ckpt_ver = 'ocid' # 'osd' 'ocid'
use_rgbd = False
infer_size = (640, 480)
dataset_root = os.path.join("/media/user/data1/dataset", dataset)
# method_id = 'msmformer_{}_rgbd_mask'.format(ckpt_ver) if use_rgbd else 'msmformer_rgb_mask'
method_id = 'msmformer_rgbd_mask' if use_rgbd else 'msmformer_rgb_mask'
mask_save_root = os.path.join('/media/user/data1/rcao/result/uois', dataset, method_id)
refine_mask_save_root = os.path.join('/media/user/data1/rcao/result/uois', dataset, method_id.replace('_mask', '_refine_mask'))
os.makedirs(mask_save_root, exist_ok=True)
os.makedirs(refine_mask_save_root, exist_ok=True)
vis_save = True
vis_save_root = os.path.join(mask_save_root, 'vis')
os.makedirs(vis_save_root, exist_ok=True)
refine_vis_save_root = os.path.join(refine_mask_save_root, 'vis')
os.makedirs(refine_vis_save_root, exist_ok=True)
image_list = read_file(os.path.join(dataset_root, 'data_list.txt'))
input_model = 'RGBD_ADD' if use_rgbd else 'COLOR'

stage1_time_list = []
steage2_time_list = []
infer_time_list = []

def get_predictor(cfg_file=cfg_file_MSMFormer, weight_path=weight_path_MSMFormer, input_image="RGBD_ADD"):
    return get_general_predictor(cfg_file, weight_path, input_image=input_image)

def get_predictor_crop(cfg_file=cfg_file_MSMFormer_crop, weight_path=weight_path_MSMFormer_crop, input_image="RGBD_ADD"):
    return get_general_predictor(cfg_file, weight_path, input_image=input_image)


predictor, cfg = get_predictor(cfg_file=cfg_file_MSMFormer,
                                weight_path=weight_path_MSMFormer,
                                input_image=input_model
                                )

predictor_crop, cfg_crop = get_predictor_crop(cfg_file=cfg_file_MSMFormer_crop,
                                                weight_path=weight_path_MSMFormer_crop,
                                                input_image=input_model)
# load image
results =[]
for image_idx, image_path in enumerate(tqdm(image_list)):
    image_name = os.path.basename(image_path).split('.')[0]

    if dataset == 'OCID':
        image_dir = os.path.join(*os.path.dirname(image_path).split('/')[:-1])
        vis_save_dir = ''
        gt_mask_path = os.path.join(dataset_root, image_path.replace('rgb', 'label'))
        gt_mask = np.array(Image.open(gt_mask_path))
        gt_mask[gt_mask == 1] = 0
        if 'table' in gt_mask_path:
            gt_mask[gt_mask == 2] = 0
        # pcd_path = os.path.join(dataset_root, image_path.replace('rgb', 'pcd'))
        # pcd_file = pcd_path.replace('png', 'pcd')
        # pcloud = o3d.io.read_point_cloud(pcd_file)
        # pcloud = np.asarray(pcloud.points).astype(np.float32)
        # pcloud[np.isnan(pcloud)] = 0
        width, height = gt_mask.shape[1], gt_mask.shape[0]
        depth_path = os.path.join(dataset_root, image_path.replace('rgb', 'depth'))
        depth_img = imageio.imread(depth_path)
        
    elif dataset == 'OSD':
        image_dir = ''
        vis_save_dir = ''
        gt_mask_path = os.path.join(dataset_root, image_path.replace('image_color', 'annotation'))
        gt_mask = np.array(Image.open(gt_mask_path))
        depth_path= os.path.join(dataset_root, image_path.replace('image_color', 'disparity'))
        depth_img = imageio.imread(depth_path)
        
    elif dataset == 'PhoCAL':
        image_dir = os.path.join(*os.path.dirname(image_path).split('/')[:-1])
        vis_save_dir = image_dir
        gt_mask_path = os.path.join(dataset_root, image_path.replace('rgb', 'mask'))
        gt_mask = np.array(Image.open(gt_mask_path))
        depth_path = os.path.join(dataset_root, image_path.replace('rgb', 'depth'))
        depth_img = imageio.imread(depth_path)
        width, height = gt_mask.shape[1], gt_mask.shape[0]
        with open(os.path.join(dataset_root, image_dir, 'scene_camera.json')) as f:
            intrinsics_json = json.load(f)
        intrinsics = intrinsics_json['rgb']
        width, height = intrinsics['width'], intrinsics['height']
        camera_info = CameraInfo(width, height, intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy'], intrinsics['depth_scale'])
        
    elif dataset == 'Housecat6D':
        image_dir = os.path.join(*os.path.dirname(image_path).split('/')[:-1])
        vis_save_dir = image_dir
        gt_mask_path = os.path.join(dataset_root, image_path.replace('rgb', 'instance'))
        gt_mask = np.array(Image.open(gt_mask_path))
        depth_path = os.path.join(dataset_root, image_path.replace('rgb', 'depth'))
        depth_img = imageio.imread(depth_path)
        intrinsics = np.loadtxt(os.path.join(dataset_root, image_dir, 'intrinsics.txt'))
        width, height = gt_mask.shape[1], gt_mask.shape[0]
        factor_depth = 1000.0
        camera_info = CameraInfo(width, height, intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2], factor_depth)
        
    im = cv2.imread(os.path.join(dataset_root, image_path))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # im_vis = copy.deepcopy(im)
    im = cv2.resize(im, infer_size, interpolation=cv2.INTER_CUBIC)
    im_vis = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    image_blob = (torch.from_numpy(im).permute(2, 0, 1) - torch.Tensor([123.675, 116.280, 103.530]).view(-1, 1, 1).float()) / torch.Tensor([58.395, 57.120, 57.375]).view(-1, 1, 1).float()
    sample = {'image_color': image_blob}
    
    # gt_mask = process_label(gt_mask)
    if use_rgbd:
        # if dataset == 'OCID':
        #     depth = pcloud.reshape(height, width, 3)
        # elif dataset == 'PhoCAL':
        #     depth = np.array(Image.open(depth_path), dtype=np.float32)
        #     depth = create_point_cloud_from_depth_image(depth, camera_info, organized=True)

        origin_depth = copy.deepcopy(depth_img)
        depth_img = normalize_depth(depth_img)
        # print(depth_img.shape, depth_img.min(), depth_img.max())
        depth_img = cv2.resize(depth_img, infer_size, interpolation=cv2.INTER_NEAREST)
        depth = inpaint_depth(depth_img, factor=1, kernel_size=5, dilate=True) / 255

        # origin_inpaint_depth = unnormalize_depth(depth_img*255)
        # print(depth_img.shape, depth_img.min(), depth_img.max())
        # depth_img = depth_img / 255.0
        # depth = torch.from_numpy(depth_img).permute(2, 0, 1).float()

        # depth_img = cv2.resize(depth_img, infer_size, interpolation=cv2.INTER_NEAREST)
        # intrinsics = adjust_intrinsics(intrinsics, (height, width), (infer_size[1], infer_size[0]))
        # camera_info = CameraInfo(infer_size[0], infer_size[1], intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2], factor_depth)
        # depth = create_point_cloud_from_depth_image(depth_img, camera_info, organized=True)
        
        # origin_inpaint_depth = create_point_cloud_from_depth_image(origin_inpaint_depth[:, :, 0], camera_info, organized=True)
        # scene = o3d.geometry.PointCloud()
        # scene.points = o3d.utility.Vector3dVector(depth.reshape(-1, 3))
        # scene.colors = o3d.utility.Vector3dVector(im_vis.reshape(-1, 3)/255)
        # o3d.visualization.draw_geometries([scene])
        sample['depth'] = torch.from_numpy(depth).permute(2, 0, 1).float()
    
    pred_mask, pred_mask_refined, stage1_time, stage2_time = test_sample_crop_nolabel(cfg, sample, predictor, predictor_crop, visualization=False, topk=False, confident_score=0.7, low_threshold=0.4, print_result=False)

    stage1_time_list.append(stage1_time)
    steage2_time_list.append(stage2_time)
    infer_time_list.append(stage1_time+stage2_time)

print("Average stage1 time: ", np.mean(stage1_time_list))
print("Average stage2 time: ", np.mean(steage2_time_list))
print("Average inference time: ", np.mean(infer_time_list))
    # pred_mask = get_result_from_network(cfg, image, depth, None, predictor, False, 0.7, 0.4, False)  # 0.7 0.4
    # eval_metrics = multilabel_metrics(pred_mask.astype(np.uint8), gt_mask)
    # print("file name: ", image_name)
    # print("first:", eval_metrics)
    
    # result = np.zeros(7)
    # result[0] = eval_metrics['Objects F-measure']
    # result[1] = eval_metrics['Objects Precision']
    # result[2] = eval_metrics['Objects Recall']
    # result[3] = eval_metrics['Boundary F-measure']
    # result[4] = eval_metrics['Boundary Precision']
    # result[5] = eval_metrics['Boundary Recall']
    # result[6] = eval_metrics['obj_detected_075_percentage']
    # results.append(result)
    
    # print("Data type of pred_mask:", pred_mask.dtype)  # 打印数据类型
    # print("Shape of pred_mask:", pred_mask.shape)      # 打印形状
    # print("Minimum value in pred_mask:", np.min(pred_mask))  # 打印最小值
    # print("Maximum value in pred_mask:", np.max(pred_mask))  # 打印最大值
    # print("Unique values in pred_mask:", np.unique(pred_mask))  # 打印所有独特的值

    # pred_mask = (pred_mask / np.max(pred_mask)) * 255
    # pred_mask = np.nan_to_num(pred_mask, nan=0)
    # result = Image.fromarray(pred_mask.astype(np.uint8))
    # mask_save_path = os.path.join(mask_save_root, image_dir)
    # os.makedirs(mask_save_path, exist_ok=True)
    # result.save(os.path.join(mask_save_path, '{}.png'.format(image_name)))

    # pred_mask_refined = (pred_mask_refined / np.max(pred_mask_refined)) * 255
    # pred_mask_refined = np.nan_to_num(pred_mask_refined, nan=0)
    # result_refined = Image.fromarray(pred_mask_refined.astype(np.uint8))
    # refine_mask_save_path = os.path.join(refine_mask_save_root, image_dir)
    # os.makedirs(refine_mask_save_path, exist_ok=True)
    # result_refined.save(os.path.join(refine_mask_save_path, '{}.png'.format(image_name)))
    
    # if vis_save:
    #     visualize_segmentation(im=im_vis, masks=pred_mask, save_dir=os.path.join(vis_save_root, "{}_{}_seg.png".format(vis_save_dir, image_name)))
    #     visualize_segmentation(im=im_vis, masks=pred_mask_refined, save_dir=os.path.join(refine_vis_save_root, "{}_{}_seg.png".format(vis_save_dir, image_name)))
        
        
# results = np.stack(results, axis=0)
# print('Overlap Prec:{}, Rec:{}, F_score:{}, Boundary Prec:{}, Rec:{}, F_score:{}, %75:{}'. \
# format(np.mean(results[:, 1]), np.mean(results[:, 2]), np.mean(results[:, 0]),
#         np.mean(results[:, 4]), np.mean(results[:, 5]), np.mean(results[:, 3]), np.mean(results[:, 6])))
# np.save('OCID_msmformer_rgb_results_new.npy', results)
