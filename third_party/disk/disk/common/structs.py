import torch, abc, sys

from torch_dimcheck import dimchecked

# the class/object below is there for making type annotations like
# def my_function(args) -> NpArray[OutputType]
if sys.version_info >= (3, 7):
    class NpArray:
        def __class_getitem__(self, arg):
            pass
else:
    # 3.6 and below don't support __class_getitem__
    class _NpArray:
        def __getitem__(self, _idx):
            pass
    
    NpArray = _NpArray()

class Features:
    @dimchecked
    def __init__(self, kp: ['N', 2], desc: ['N', 'F'], kp_logp: ['N']):
        assert kp.device == desc.device
        assert kp.device == kp_logp.device

        self.kp      = kp
        self.desc    = desc
        self.kp_logp = kp_logp

    @property
    def n(self):
        return self.kp.shape[0]

    @property
    def device(self):
        return self.kp.device

    def detached_and_grad_(self):
        return Features(
            self.kp,
            self.desc.detach().requires_grad_(),
            self.kp_logp.detach().requires_grad_(),
        )

    def requires_grad_(self, is_on):
        self.desc.requires_grad_(is_on)
        self.kp_logp.requires_grad_(is_on)

    def grad_tensors(self):
        return [self.desc, self.kp_logp]

    def to(self, *args, **kwargs):
        return Features(
            self.kp.to(*args, **kwargs),
            self.desc.to(*args, **kwargs),
            self.kp_logp.to(*args, **kwargs) if self.kp_logp is not None else None,
        )

class MatchDistribution(abc.ABC):
    @abc.abstractmethod
    def sample(self) -> [2, 'K']:
        pass

    @abc.abstractmethod
    def mle(self) -> [2, 'K']:
        pass

    @abc.abstractmethod
    def dense_logp(self):
        pass

    @abc.abstractmethod
    def dense_p(self):
        pass

    @abc.abstractmethod
    def features_1(self) -> Features:
        pass

    @abc.abstractmethod
    def features_2(self) -> Features:
        pass

    @property
    def shape(self):
        return self.features_1().kp.shape[0], self.features_2().kp.shape[1]

    def matched_pairs(self, mle=False):
        matches = self.mle() if mle else self.sample()

        return MatchedPairs(
            self.features_1().kp,
            self.features_2().kp,
            matches,    
        )


class MatchedPairs:
    @dimchecked
    def __init__(self, kps1: ['N', 2], kps2: ['M', 2], matches: [2, 'K']):
        self.kps1    = kps1
        self.kps2    = kps2
        self.matches = matches

    def to(self, *args, **kwargs):
        return MatchedPairs(
            self.kps1.to(*args, **kwargs),
            self.kps2.to(*args, **kwargs),
            self.matches.to(*args, **kwargs),
        )
