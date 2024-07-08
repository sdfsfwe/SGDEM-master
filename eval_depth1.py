import os
import cv2
# Public libraries
import numpy as np
import torch
from torchvision.utils import save_image

# Local imports
import colors
from arguments import DepthEvaluationArguments
from harness import Harness


class DepthEvaluator(Harness):
    def _init_validation(self, opt):
        self.fixed_depth_scaling = opt.depth_validation_fixed_scaling
        self.ratio_on_validation = opt.depth_ratio_on_validation
        self.val_num_log_images = opt.eval_num_images

    def evaluate(self):
        print('Evaluate depth predictions:', flush=True)
        print(self.val_num_log_images)
        scores, ratios, images = self._run_depth_validation(self.val_num_log_images)

        for domain in images:
            domain_dir = os.path.join(self.log_path, 'eval_images', domain)
            os.makedirs(domain_dir, exist_ok=True)

            for i, (color_gt, depth_gt, depth_pred) in enumerate(images[domain]):
                image_path1= os.path.join(domain_dir, f'depth_{i}.png')
                image_path = os.path.join(domain_dir, f'img_{i}.png')

                save_image(
                    color_gt,
                    image_path
                )
                
                save_image(
                    depth_pred,
                    image_path1
                )

        self._log_gpu_memory()
        
    def depth_to_color(self,depth_map):
        depth_map = depth_map.numpy()
        # 将深度映射到伪彩色
        min_depth = np.min(depth_map)
        max_depth = np.max(depth_map)
        
        # 使用OpenCV的applyColorMap函数来进行伪彩色映射
        depth_colormap = cv2.applyColorMap(
            np.uint8(255 * (depth_map - min_depth) / (max_depth - min_depth)),
            cv2.COLORMAP_JET
        )
        
        depth_colormap = torch.from_numpy(depth_colormap)
        return depth_colormap


if __name__ == "__main__":
    opt = DepthEvaluationArguments().parse()

    if opt.model_load is None:
        raise Exception('You must use --model-load to select a model state directory to run evaluation on')

    if opt.sys_best_effort_determinism:
        import random

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        random.seed(1)

    evaluator = DepthEvaluator(opt)
    evaluator.evaluate()