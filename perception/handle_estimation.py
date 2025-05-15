#!/usr/bin/env python3.10
# filepath: ~/alphaz_ws/src/handle_detection/scripts/handle_detection_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as rosImage

from cv_bridge import CvBridge
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

import detectron2
from detectron2.utils.logger import setup_logger


center_net_path = os.path.join(os.path.dirname(__file__), 'Detic/third_party/CenterNet2')
sys.path.insert(0, os.path.abspath(center_net_path))


# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from Detic.detic.modeling.text.text_encoder import build_text_encoder
from Detic.detic.modeling.utils import reset_cls_test


from centernet.config import add_centernet_config
from Detic.detic.config import add_detic_config
from segment_anything import sam_model_registry, SamPredictor
from geometry_msgs.msg import Point
import os

# absolute path to this file
THIS_FILE = os.path.abspath(__file__)
# directory containing this file
THIS_DIR  = os.path.dirname(THIS_FILE)

############################################
def DETIC_predictor():
    # Build the detector and download our pretrained weights
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    config_path = os.path.expanduser(THIS_DIR+"/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1 # set threshold for this model
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True # For better visualization purpose. Set to False for all classes.
    cfg.MODEL.DEVICE='cuda' # 'cuda' or 'cpu'
    detic_predictor = DefaultPredictor(cfg)
    return detic_predictor

def SAM_predictor(device):
    sam_checkpoint = THIS_DIR + "/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = device
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    return sam_predictor

def Detic(im, metadata, detic_predictor, visualize=False):
    if im is None:
        print("Error: Unable to read the image file")

    # Run model and show results
    output =detic_predictor(im[:, :, ::-1])  # Detic expects BGR images.
    v = Visualizer(im, metadata)
    out = v.draw_instance_predictions(output["instances"].to('cpu'))
    instances = output["instances"].to('cpu')
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.numpy()
    if visualize:
        visualize_detic(out)
    return boxes, classes

def SAM(im, boxes, class_idx, metadata, sam_predictor):
    sam_predictor.set_image(im)
    input_boxes = torch.tensor(boxes, device=sam_predictor.device)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(input_boxes, im.shape[:2])
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    return masks

def visualize_output(im, masks, input_boxes, classes, image_save_path, mask_only=False):
    plt.figure(figsize=(10, 10))
    plt.imshow(im)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    if not mask_only:
        for box, class_name in zip(input_boxes, classes):
            show_box(box, plt.gca())
            x, y = box[:2]
            plt.gca().text(x, y - 5, class_name, color='white', fontsize=12, fontweight='bold', bbox=dict(facecolor='green', edgecolor='green', alpha=0.5))
    plt.axis('off')
    plt.savefig(image_save_path)
    plt.show()

def visualize_detic(output):
    output_im = output.get_image()[:, :, ::-1]
    cv2.imshow("Detic Predictions", output_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def custom_vocab(detic_predictor, classes, threshold=0.3):
    vocabulary = 'custom'
    metadata = MetadataCatalog.get("__unused2")
    metadata.thing_classes = classes # Change here to try your own vocabularies!
    classifier = get_clip_embeddings(metadata.thing_classes)
    num_classes = len(metadata.thing_classes)
    reset_cls_test(detic_predictor.model, classifier, num_classes)

    # Reset visualization threshold
    output_score_threshold = threshold
    for cascade_stages in range(len(detic_predictor.model.roi_heads.box_predictor)):
        detic_predictor.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = output_score_threshold
    return metadata

def get_clip_embeddings(vocabulary, prompt='a '):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb

############################################


class HandleDetectionNode(Node):
    def __init__(self):
        super().__init__('handle_detection_node')

        # Parameters
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.classes = ["handle"]
        self.threshold = 0.3

        # Initialize predictors
        self.detic_predictor = DETIC_predictor()
        self.sam_predictor = SAM_predictor(self.device)
        self.metadata = custom_vocab(self.detic_predictor, self.classes, self.threshold)

        # CV Bridge
        self.bridge = CvBridge()

        # Subscribers
        self.create_subscription(rosImage, "/isaac_sim/color", self.image_callback, 10)
        self.create_subscription(rosImage, "/isaac_sim/depth", self.depth_callback, 10)
        # self.create_subscription(rosImage, "/camera_0/color/image_raw", self.image_callback, 10)
        # self.create_subscription(rosImage, "/camera_0/depth/image_raw", self.depth_callback, 10)
        # Publisher
        self.handle_detection_pub = self.create_publisher(rosImage, "/handle_detection_head", 10)
        self.grip_publisher = self.create_publisher(Point, "/grip_point", 10)
        # self.handle_detection_pub = self.create_publisher(rosImage, "/handle_detection_head", 10)
        # self.grip_publisher = self.create_publisher(Point, "/grip_point", 10)

        self.depth_image = None
        # OpenCV Window
        # cv2.namedWindow("Handle Detection", cv2.WINDOW_NORMAL)
        self.ppx = 320.0
        self.ppy = 240.0
        self.fx = 617.0
        self.fy = 617.0

    def depth_callback(self, msg):
        # Convert ROS Image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.depth_image = np.array(cv_image, dtype=np.float32)
        pass

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # Perform detection
        image = np.array(cv_image, dtype=np.uint8)
        boxes, class_idx = Detic(image, self.metadata, self.detic_predictor, visualize=False)

        if len(boxes) > 0:
            masks = SAM(cv_image, boxes, class_idx, self.metadata, self.sam_predictor)
            classes = [self.metadata.thing_classes[idx] for idx in class_idx]
            mask = masks[0].cpu().numpy()
            mask = (mask * 255).astype(np.uint8).squeeze()

            # Compute the center of the mask
            mask_indices = np.argwhere(mask > 0)
            y_center, x_center = np.mean(mask_indices, axis=0).astype(int)

            # Publish the mask image
            mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
            self.handle_detection_pub.publish(mask_msg) 

            # Convert depth to 3D point
            print(x_center, y_center)
            try:
                x, y, z = self.depth2point(x_center, y_center, self.depth_image[y_center, x_center])
                print(self.depth_image.shape)
                print("3D Point:", x, y, z)
                point_msg = Point()
                point_msg.x = float(x)
                point_msg.y = float(y)
                point_msg.z = float(z)
                self.grip_publisher.publish(point_msg)
            except Exception as e:
                pass
        else:
            self.get_logger().info("No handles detected.")
        pass
    
    def depth2point(self, x, y, depth_value):
        x = (x - self.ppx) / self.fx * depth_value
        y = (y - self.ppy) / self.fy * depth_value
        z = depth_value
        # x = - (x - self.ppx) / self.fx * depth_value
        # z = - (y - self.ppy) / self.fy * depth_value
        # y = -depth_value
        return x, y, z

def main(args = None):
    rclpy.init(args=args)
    setup_logger() 
    node = HandleDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()