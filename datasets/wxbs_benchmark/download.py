from wxbs_benchmark.dataset import *
dset = EVDDataset('.EVD',  download=True)
dset = WxBSDataset('.WxBS', subset='test', download=True)