import torch
from ..utils.base_model import BaseModel
import sys
from pathlib import Path
import subprocess
import logging
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).parent / '../../third_party'))
from ASpanFormer.src.ASpanFormer.aspanformer import ASpanFormer as _ASpanFormer
from ASpanFormer.src.config.default import get_cfg_defaults
from ASpanFormer.src.utils.misc import lower_config
from ASpanFormer.demo import demo_utils 
aspanformer_path = Path(__file__).parent / '../../third_party/ASpanFormer'

class ASpanFormer(BaseModel):
    default_conf = {
        'weights': 'outdoor',
        'match_threshold': 0.2,
        'config_path': aspanformer_path / 'configs/aspan/outdoor/aspan_test.py',
        'model_name': 'weights_aspanformer.tar',
    }
    required_inputs = [
        'image0',
        'image1'
    ]
    proxy = 'http://localhost:1080'
    aspanformer_models = {
        'weights_aspanformer.tar': 'https://drive.google.com/uc?id=1eavM9dTkw9nbc-JqlVVfGPU5UvTTfc6k'
    }

    def _init(self, conf):
        model_path = aspanformer_path / 'weights' / Path(conf['weights'] + '.ckpt')
        # Download the model.
        if not model_path.exists():
            # model_path.parent.mkdir(exist_ok=True)
            tar_path = aspanformer_path / conf['model_name']
            link = self.aspanformer_models[conf['model_name']]
            cmd = ['gdown', link, '-O', tar_path, '--proxy', self.proxy]
            logger.info(f'Downloading the Gluestick model with `{cmd}`.')
            subprocess.run(cmd, check=True)
            cmd = ['tar xvf', str(tar_path)]
            subprocess.run(cmd, check=True)
            logger.info(f'Unzip model file `{cmd}`.')

        logger.info(f'Loading GlueStick model...')
    
        config = get_cfg_defaults()
        config.merge_from_file(conf['config_path'])
        _config = lower_config(config)
        self.net = _ASpanFormer(config=_config['aspan'])
        weight_path =  model_path
        state_dict = torch.load(str(weight_path), map_location='cpu')['state_dict']
        self.net.load_state_dict(state_dict, strict=False)

    def _forward(self, data):
        data_ = {'image0': data['image0'],
                 'image1': data['image1'],}
        self.net(data_,online_resize=True)
        corr0 = data_['mkpts0_f']
        corr1 = data_['mkpts1_f']
        pred = {}
        pred['keypoints0'], pred['keypoints1'] = corr0, corr1
        return pred