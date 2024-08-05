"""Monocular depth module to train Gaussian Splatting model."""
import os.path as osp
from enum import Enum

import sys
import torch

import cv2
from matplotlib import pyplot as plt
from gaussian_splatting_py.load_yaml import load_config
from gaussian_splatting_py.vision_utils.vision_utils import preprocess_image, resize_if_too_large

from gaussian_splatting_py.metric_depth.depth_anything_v2.dpt import DepthAnythingV2 as MetricDepthAnythingV2



MODEL_LIST = {
    'DEPTH_ANYTHING': 'DepthAnythingV2',
    'METRIC_3D': 'Metric3DV2'
}

FINETUNED_DATASET = 'hypersim'
MAX_DA_DEPTH = 20.0


DEPTH_ANYTHING_CONFIGS = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }


def build_model(cfg):
    model_cfg = cfg["model"]
    model_name = model_cfg["name"]
    model_type = model_cfg["type"]
    if model_name == MODEL_LIST['DEPTH_ANYTHING']:
        model = MetricDepthAnythingV2(**{**DEPTH_ANYTHING_CONFIGS[model_type], 'max_depth': MAX_DA_DEPTH})
        checkpoint_dir = osp.join(osp.dirname(osp.abspath(__file__)), "checkpoints") 
        
        model.load_state_dict(torch.load(f'{checkpoint_dir}/depth_anything_v2_metric_{FINETUNED_DATASET}_{model_type}.pth', map_location='cpu'))
        model.cuda().eval()
        return model
        
    elif model_name == MODEL_LIST['METRIC_3D']:
        model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_large', pretrain=True)
        model.cuda().eval()
        return model
    else:
        raise NameError(f"Model {model_name} not supported. Try one of {MODEL_LIST.keys()}")

class MonocularDepth(object):
    """Monocular depth module to train Gaussian Splatting model."""
    def __init__(self, cfg):
        
        self.model_name = cfg["model"]["name"]
        self.model = build_model(cfg)
        
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total trainable parameters in model: {pytorch_total_params}")
        
    def __call__(self, img):
        """run inference on the model"""
        # clear cuda cache
        torch.cuda.empty_cache()
        output_dict = {}
        img = resize_if_too_large(img)
        print(f"Image shape: {img.shape}")
        with torch.no_grad():
            if self.model_name == MODEL_LIST['DEPTH_ANYTHING']:
                pred_depth = self.model.infer_image(img)
                output_dict['depth'] = pred_depth
                return output_dict
            
            elif self.model_name == MODEL_LIST['METRIC_3D']:
                img = preprocess_image(img).unsqueeze(0).cuda()
                pred_depth, confidence, output_dict = self.model.inference({'input': img})
                pred_depth[confidence < 0.5] = 0
                pred_depth[pred_depth > 20] = 0
                pred_depth = pred_depth.squeeze().cpu().detach().numpy()
                pred_depth = cv2.resize(pred_depth, (pred_depth.shape[1] - 8, pred_depth.shape[0] - 8)) 
                
                confidence = confidence.squeeze().cpu().detach().numpy()
                
                output_dict['depth'] = pred_depth
                output_dict['confidence'] = confidence
                output_dict['output_dict'] = output_dict
                return output_dict
            
            else:
                raise NameError(f"Model {self.model_name} not supported. Try one of {MODEL_LIST.keys()}")
        
        
if __name__ == '__main__':
    # read from config file
    cfg = load_config('sample_mde_config.yml')
    monocular_depth = MonocularDepth(cfg)
    
    # load image
    img = cv2.imread('sample_imgs/clutter.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # run inference
    output_dict = monocular_depth(img)
    depth = output_dict['depth']
    
    # show depth
    plt.imshow(depth)
    plt.show()