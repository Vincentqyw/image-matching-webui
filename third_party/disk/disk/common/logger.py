from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, path):
        self.sw           = SummaryWriter(path)
        self.tag_counters = {}

    def add_scalar(self, tag, value):
        if tag not in self.tag_counters:
            self.tag_counters[tag] = 0

        counter = self.tag_counters[tag] 

        self.sw.add_scalar(tag, value, global_step=counter)
        self.tag_counters[tag] += 1

    def add_scalars(self, tag_to_value, prefix=''):
        if prefix != '':
            prefix = f'{prefix}/'

        for tag, value in tag_to_value.items():
            tag = f'{prefix}{tag}'
            self.add_scalar(tag, value)
