from wxbs_benchmark.dataset import *  # noqa: F403

dset = EVDDataset(".EVD", download=True)  # noqa: F405
dset = WxBSDataset(".WxBS", subset="test", download=True)  # noqa: F405
