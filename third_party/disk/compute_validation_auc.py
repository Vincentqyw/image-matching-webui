import argparse, os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tensorflow.core.util import event_pb2

def read_deltas(path):
    reader = tf.data.TFRecordDataset(path)

    Δ_θ = []
    Δ_T = []
    for serialized_event in tqdm(reader):
        event = event_pb2.Event.FromString(serialized_event.numpy())
        for value in event.summary.value:
            if value.tag == 'test/pose/Δ_θ':
                Δ_θ.append(value.simple_value)
            if value.tag == 'test/pose/Δ_T':
                Δ_T.append(value.simple_value)

    Δ_θ = np.array(Δ_θ)
    Δ_T = np.array(Δ_T)

    return Δ_θ, Δ_T

def calculate_auc(Δ_θ, Δ_T, length):
    error = np.maximum(Δ_θ, Δ_T)

    # if we run this script while training is still going on, we may find
    # ourselves halfway through a validation run, in which case the number
    # of events will not be divisible by `length`. We clip to only include
    # full epochs. WARNING: this means that miss-specifying
    # `--validation--length` will cause silent errors
    error = error[:length * (error.shape[0] // length)]
    error = error.reshape(-1, length)

    # bins for 0..10 degrees of error
    bins = np.arange(0, 11)
    aucs = []
    for e, errs in enumerate(error):
        hist, _edges = np.histogram(errs, bins=bins)
        hist = hist / length
        auc = hist.cumsum().mean()
        aucs.append(auc)

    return np.array(aucs)

parser = argparse.ArgumentParser(
    description=('This script is used for evaluating the validation time '
                 'stereo pose estimation AUC in order to pick the best '
                 'checkpoint for IMW2020 challenge submission. It reads the '
                 'tensorboard logfiles, picks the pose estimation errors, '
                 'groups them by epoch and calculates AUC for each epoch.'),

    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--validation-length', type=int, default=750,
    help=('The number of image pairs evaluated at each epoch. Unfortunately, '
          'tensorboard doesn\'t store that with events, so we have to just '
          'assume that first 0:vl events correspond to 1st epoch, vl:2*vl to '
          '2nd epoch, and so on (vl == --validation-length).')
)
parser.add_argument(
    'paths', type=str, nargs='+',
    help=('Point to (multiple) tensorboard event files (by default, their '
          'names are something like '
          'events.out.tfevents.1601909417.tyszkiew-disk-1.30.0')
)
args = parser.parse_args()

for path in args.paths:
    abs_path = os.path.abspath(path)
    deltas = read_deltas(abs_path)
    auc = calculate_auc(*deltas, validation_length=args.validation_length)
    plt.plot(auc, label=path)

plt.legend()
plt.show()
