import sys
from pathlib import Path
import logging
from ..utils.base_model import BaseModel
logger = logging.getLogger(__name__)
lightglue_path = Path(__file__).parent / '../../third_party/LightGlue'
sys.path.append(str(lightglue_path))
from lightglue import LightGlue as LG

class LightGlue(BaseModel):
    default_conf = {
        'match_threshold': 0.2,
        'filter_threshold': 0.2,
        'width_confidence': 0.99,  # for point pruning
        'depth_confidence': 0.95,  # for early stopping,
        'pretrained': 'superpoint',
        'model_name': 'superpoint_lightglue.pth',
        # 'input_dim': 256,
        # 'descriptor_dim': 256,
    }
    required_inputs = [
        'image0', 'keypoints0', 'scores0', 'descriptors0',
        'image1', 'keypoints1', 'scores1', 'descriptors1',
    ]

    def _init(self, conf):
        weight_path = lightglue_path / 'weights' / conf['model_name']
        conf['weights'] = str(weight_path)
        conf['filter_threshold'] = conf['match_threshold']
        self.net = LG( **conf)
        logger.info(f'Load lightglue model done.')

    def _forward(self, data):
        data['keypoints0'] = data['keypoints0'][None]
        data['keypoints1'] = data['keypoints1'][None]
        data['descriptors0'] = data['descriptors0'].permute(0, 2, 1)
        data['descriptors1'] = data['descriptors1'].permute(0, 2, 1)
        return self.net(data)
