import torch, random, argparse
from tqdm import tqdm

from disk.common import Logger
from disk.data import get_datasets
from disk.model import DISK, ConsistentMatcher, CycleMatcher
from disk.loss import Reinforce, DepthReward, EpipolarReward, \
                      PoseQuality, DiscreteMetric

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    'data_path', type=str,
    help=('Path to the datasets. This should point to the location with '
          '`megadepth` and `imw2020-val` directories.'),
)
parser.add_argument(
    '--reward', choices=['epipolar', 'depth'], default='depth',
    help='Reward criterion to use'
)
parser.add_argument(
    '--save-dir', type=str, default='artifacts',
    help=('Path for saving artifacts (checkpoints and tensorboard logs). Will '
          'be created if doesn\'t exist')
)
parser.add_argument(
    '--batch-size', type=int, default=2,
    help='The size of the batch',
)
parser.add_argument(
    '--chunk-size', type=int, default=5000,
    help=('The number of batches in the (pseudo) epoch. We run validation and '
          'save a checkpoint once per epoch, as well as use them for scheduling'
          ' the reward annealing'),
)
parser.add_argument(
    '--substep', type=int, default=1,
    help=('Number of batches to accumulate gradients over. Can be increased to'
          ' compensate for smaller batches on GPUs with less VRAM'),
)
parser.add_argument(
    '--warmup', type=int, default=250,
    help=('The first (pseudo) epoch can be much shorter, this avoids wasting '
          'time.'),
)
parser.add_argument(
    '--height', type=int, default=768,
    help='We train on images resized to (height, width)',
)
parser.add_argument(
    '--width', type=int, default=768,
    help='We train on images resized to (height, width)',
)
parser.add_argument(
    '--train-scene-limit', type=int, default=1000,
    help=('Different scenes in the dataset have a different amount of '
          'covisible image triplets. We (randomly) subselect '
          '--train-scene-limit of them for training, to avoid introducing '
          'a data bias towards those scenes.')
)
parser.add_argument(
    '--test-scene-limit', type=int, default=250,
    help=('Different scenes in the dataset have a different amount of '
          'covisible image triplets. We (randomly) subselect '
          '--test-scene-limit of them for validation to avoid '
          'to avoid introducing a bias towards those scenes.')
)
parser.add_argument(
    '--n-epochs', type=int, default=50,
    help='Number of (pseudo) epochs to train for',
)
parser.add_argument(
    '--desc-dim', type=int, default=128,
    help='Dimensionality of descriptors to produce. 128 by default',
)
parser.add_argument(
    '--load', type=str, default=None,
    help='Path to a checkpoint to resume training from',
)
parser.add_argument(
    '--epoch-offset', type=int, default=0,
    help=('Start counting epochs from this value. Influences the annealing '
          'procedures, and is therefore useful when restarting from a '
          'checkpoint'),
)
args = parser.parse_args()

DEV  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEV}')

random.seed(42)

# create the feature extractor and descriptor. It does not handle matching,
# this will come later
disk = DISK(window=8, desc_dim=args.desc_dim)

# maybe load from a checkpoint
if args.load is not None:
    state_dict = torch.load(args.load, map_location='cpu')['disk'] 
    disk.load_state_dict(state_dict)

disk = disk.to(DEV)

# get training datasets. They will yield Images as defined in
# disk/common/image.py. This structure contains the actual bitmap,
# camera position and intrinsics (focal length, etc) and optionally
# depth maps.
train_chunk_iter, test_iter = get_datasets(
    args.data_path,
    no_depth=args.reward == 'epipolar',
    batch_size=args.batch_size,
    chunk_size=args.chunk_size,
    substep=args.substep,
    n_epochs=args.n_epochs,
    train_limit=args.train_scene_limit,
    test_limit=args.test_scene_limit,

    crop_size=(args.height, args.width),
)

logger = Logger(args.save_dir)

# set up the inference-time matching algorthim and validation metrics
valtime_matcher     = CycleMatcher()
pose_quality_metric = PoseQuality()
disc_quality_metric = DiscreteMetric(th=1.5, lm_kp=-0.01)

if args.reward == 'epipolar':
    reward_class = EpipolarReward
elif args.reward == 'depth':
    reward_class = DepthReward
else:
    raise ValueError(f'Unknown reward mode `{args.reward}`')

optim = torch.optim.Adam(disk.parameters(), lr=1e-4)

for e, chunk in enumerate(train_chunk_iter):
    # this allows us to offset the annealing below, for instance when resuming
    # training from a checkpoint
    e += args.epoch_offset

    # this is an important part: if we start with a random initialization
    # it's pretty bad at first. Therefore if we set penalties for bad matches,
    # the algorithm will quickly converge to the local optimum of not doing
    # anything (which yields 0 reward, still better than negative).
    # Therefore in the first couple of epochs I start with very low (0)
    # penalty and then gradually increase it. The very first epoch can be
    # short, and is controllable by the --warmup switch (default 250)
    if e == 0:
        ramp = 0.
    elif e == 1:
        ramp = 0.1
    else:
        ramp = min(1., 0.1 + 0.2 * e)

    loss_fn = Reinforce(
        reward_class(
            lm_tp=1.,
            lm_fp=-0.25 * ramp,
            th=1.5,
        ),
        lm_kp=-0.001 * ramp
    )

    # this is a module which is used to perform matching. It has a single
    # parameter called θ_M in the paper and `inverse_T` here. It could be
    # learned but I instead anneal it between 15 and 50
    inverse_T = 15 + 35 * min(1., 0.05 * e)
    matcher = ConsistentMatcher(inverse_T=inverse_T).to(DEV)
    matcher.requires_grad_(False)

    # the main training loop
    for i, batch in enumerate(tqdm(chunk, total=args.chunk_size)):
        # get the data onto GPU
        bitmaps, images = batch.to(DEV, non_blocking=True)

        # some reshaping because the image pairs are shaped like
        # [2, batch_size, rgb, height, width] and DISK accepts them
        # as [2 * batch_size, rgb, height, width]
        bitmaps_ = bitmaps.reshape(-1, *bitmaps.shape[2:])

        # extract the features. They are a numpy array of size [2 * batch_size]
        # which contains objects of type disk.common.Features
        features_ = disk.features(bitmaps_, kind='rng')
        # reshape them back to [2, batch_size]
        features = features_.reshape(*bitmaps.shape[:2])

        # normally we'd do something like
        # > matches = matcher(features)
        # > loss, stats = loss_fn(matches, images)
        # > loss.backward()
        # but here I do a trick to squeeze bigger batch sizes in GPU memory
        # (the algorithm is very memory hungry because we create huge feature
        # distance matrices). This is described in the paper in section 4.
        # in "optimization"
        stats = loss_fn.accumulate_grad(images, features, matcher)
        del bitmaps, images, features

        # Make an optimization step. args.substep is there to allow making bigger
        # "batches" by just accumulating gradient across several of those.
        # Again, this is because the algorithm is so memory hungry it can be
        # an issue to have batches bigger than 1.
        if i % args.substep == args.substep - 1:
            optim.step()
            optim.zero_grad()

        for sample in stats.flat:
            logger.add_scalars(sample, prefix='train')

        # first epoch can be cut short after args.warmup optimization steps
        if e == 0 and i == args.warmup:
            break

    torch.save({
        'disk': disk.state_dict(),
    }, f'{args.save_dir}/save-{e}.pth')

    # validation loop
    for i, batch in enumerate(tqdm(test_iter)):
        bitmaps, images = batch.to(DEV, non_blocking=True)
        bitmaps_ = bitmaps.reshape(-1, *bitmaps.shape[2:])
        with torch.no_grad():
            # at validation we use NMS extraction...
            features_ = disk.features(bitmaps_, kind='nms')
            features = features_.reshape(*bitmaps.shape[:2])

            # ...and nearest-neighbor matching
            matches = valtime_matcher.match_pairwise(features)
            d_stats = disc_quality_metric(images, matches)
            p_stats = pose_quality_metric(images, matches)

            for d_stat in d_stats.flat:
                # those are metrics similar to the ones used at training time:
                # number of true/false positives, etc. They are called
                # `discrete` because I compute them after actually performing
                # mutual nearest neighbor (cycle consistent) matching, rather
                # than report the expectations, as I do at trianing time
                logger.add_scalars(d_stat, prefix='test/discrete')
            for p_stat in p_stats.flat:
                # those are metrics related to camera pose estimation: error in
                # camera rotation and translation
                logger.add_scalars(p_stat, prefix='test/pose')

        del bitmaps, images, features

print('Finished')
