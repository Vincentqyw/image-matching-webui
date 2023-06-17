import os
import contextlib
import joblib
from typing import Union
from loguru import _Logger, logger
from itertools import chain

import torch
from yacs.config import CfgNode as CN
from pytorch_lightning.utilities import rank_zero_only
import cv2
import numpy as np

def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


def upper_config(dict_cfg):
    if not isinstance(dict_cfg, dict):
        return dict_cfg
    return {k.upper(): upper_config(v) for k, v in dict_cfg.items()}


def log_on(condition, message, level):
    if condition:
        assert level in ['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL']
        logger.log(level, message)


def get_rank_zero_only_logger(logger: _Logger):
    if rank_zero_only.rank == 0:
        return logger
    else:
        for _level in logger._core.levels.keys():
            level = _level.lower()
            setattr(logger, level,
                    lambda x: None)
        logger._log = lambda x: None
    return logger


def setup_gpus(gpus: Union[str, int]) -> int:
    """ A temporary fix for pytorch-lighting 1.3.x """
    gpus = str(gpus)
    gpu_ids = []
    
    if ',' not in gpus:
        n_gpus = int(gpus)
        return n_gpus if n_gpus != -1 else torch.cuda.device_count()
    else:
        gpu_ids = [i.strip() for i in gpus.split(',') if i != '']
    
    # setup environment variables
    visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
    if visible_devices is None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in gpu_ids)
        visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
        logger.warning(f'[Temporary Fix] manually set CUDA_VISIBLE_DEVICES when specifying gpus to use: {visible_devices}')
    else:
        logger.warning('[Temporary Fix] CUDA_VISIBLE_DEVICES already set by user or the main process.')
    return len(gpu_ids)


def flattenList(x):
    return list(chain(*x))


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument
    
    Usage:
        with tqdm_joblib(tqdm(desc="My calculation", total=10)) as progress_bar:
            Parallel(n_jobs=16)(delayed(sqrt)(i**2) for i in range(10))
            
    When iterating over a generator, directly use of tqdm is also a solutin (but monitor the task queuing, instead of finishing)
        ret_vals = Parallel(n_jobs=args.world_size)(
                    delayed(lambda x: _compute_cov_score(pid, *x))(param)
                        for param in tqdm(combinations(image_ids, 2),
                                          desc=f'Computing cov_score of [{pid}]',
                                          total=len(image_ids)*(len(image_ids)-1)/2))
    Src: https://stackoverflow.com/a/58936697
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def draw_points(img,points,color=(0,255,0),radius=3):
    dp = [(int(points[i, 0]), int(points[i, 1])) for i in range(points.shape[0])]
    for i in range(points.shape[0]):
        cv2.circle(img, dp[i],radius=radius,color=color)
    return img
    

def draw_match(img1, img2, corr1, corr2,inlier=[True],color=None,radius1=1,radius2=1,resize=None):
    if resize is not None:
        scale1,scale2=[img1.shape[1]/resize[0],img1.shape[0]/resize[1]],[img2.shape[1]/resize[0],img2.shape[0]/resize[1]]
        img1,img2=cv2.resize(img1, resize, interpolation=cv2.INTER_AREA),cv2.resize(img2, resize, interpolation=cv2.INTER_AREA) 
        corr1,corr2=corr1/np.asarray(scale1)[np.newaxis],corr2/np.asarray(scale2)[np.newaxis]
    corr1_key = [cv2.KeyPoint(corr1[i, 0], corr1[i, 1], radius1) for i in range(corr1.shape[0])]
    corr2_key = [cv2.KeyPoint(corr2[i, 0], corr2[i, 1], radius2) for i in range(corr2.shape[0])]

    assert len(corr1) == len(corr2)

    draw_matches = [cv2.DMatch(i, i, 0) for i in range(len(corr1))]
    if color is None:
        color = [(0, 255, 0) if cur_inlier else (0,0,255) for cur_inlier in inlier]
    if len(color)==1:
        display = cv2.drawMatches(img1, corr1_key, img2, corr2_key, draw_matches, None,
                              matchColor=color[0],
                              singlePointColor=color[0],
                              flags=4
                              )
    else:
        height,width=max(img1.shape[0],img2.shape[0]),img1.shape[1]+img2.shape[1]
        display=np.zeros([height,width,3],np.uint8)
        display[:img1.shape[0],:img1.shape[1]]=img1
        display[:img2.shape[0],img1.shape[1]:]=img2
        for i in range(len(corr1)):
            left_x,left_y,right_x,right_y=int(corr1[i][0]),int(corr1[i][1]),int(corr2[i][0]+img1.shape[1]),int(corr2[i][1])
            cur_color=(int(color[i][0]),int(color[i][1]),int(color[i][2]))
            cv2.line(display, (left_x,left_y), (right_x,right_y),cur_color,1,lineType=cv2.LINE_AA)
    return display
