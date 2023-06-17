import torch
import numpy as np

from disk import MatchDistribution, Features, NpArray, Image

class Reinforce:
    def __init__(self, reward, lm_kp):
        self.reward = reward
        self.lm_kp   = lm_kp

    def _loss_for_pair(self, match_dist: MatchDistribution, img1: Image, img2: Image):
        elementwise_rewards = self.reward(
            match_dist.features_1().kp,
            match_dist.features_2().kp,
            img1,
            img2,
        )

        with torch.no_grad():
            # we don't want to backpropagate through this
            sample_p = match_dist.dense_p() # [N, M]

        sample_logp = match_dist.dense_logp() # [N, M]

        # [N, M]
        kps_logp    = match_dist.features_1().kp_logp.reshape(-1, 1) \
                    + match_dist.features_2().kp_logp.reshape(1, -1)

        # scalar, used for introducing the lm_kp penalty
        sample_lp_flat = match_dist.features_1().kp_logp.sum() \
                       + match_dist.features_2().kp_logp.sum()
        
        # [N, M], p * logp of sampling a pair
        sample_plogp = sample_p * (sample_logp + kps_logp)

        reinforce  = (elementwise_rewards * sample_plogp).sum()
        kp_penalty = self.lm_kp * sample_lp_flat
        #loss = -((elementwise_rewards * sample_plogp).sum() \
        #         + self.lm_kp * sample_lp_flat.sum())

        loss = -reinforce - kp_penalty

        n_keypoints = match_dist.shape[0] + match_dist.shape[1]
        exp_n_pairs = sample_p.sum().item()
        exp_reward  = (sample_p * elementwise_rewards).sum().item() \
                    + self.lm_kp * n_keypoints

        stats = {
            'reward'     : exp_reward,
            'n_keypoints': n_keypoints,
            'n_pairs'    : exp_n_pairs,
        }

        return loss, stats

    def accumulate_grad(
        self,
        images  : NpArray[Image],    # [N_scenes, N_per_scene]
        features: NpArray[Features], # [N_scenes, N_per_scene]
        matcher,
    ):
        '''
        This method performs BOTH forward and backward pass for the network
        (calling loss.backward() is not necessary afterwards).

        For every pair of covisible images we create a feature match matrix
        which is memory-consuming. In a standard forward -> backward PyTorch
        workflow, those would be all computed (forward pass), then the loss
        would be computed and finally backpropagation would be ran. In our
        case, since we don't need the matrices to stick around, we backprop
        through matching of each image pair on-the-fly, accumulating the
        gradients at Features level. Then, we finally backpropagate from
        Features down to network parameters.
        '''
        assert images.shape == features.shape

        N_scenes, N_per_scene = images.shape
        N_decisions           = ((N_per_scene - 1) * N_per_scene) // 2

        stats = np.zeros((N_scenes, N_decisions), dtype=object)

        # we detach features from the computation graph, so that when we call
        # .backward(), the computation will not flow down to the Unet. We
        # mark them as .requires_grad==True, so they will accumulate the
        # gradients across pairwise matches.
        detached_features = np.zeros(features.shape, dtype=object)
        for i in range(features.size):
            detached_features.flat[i] = features.flat[i].detached_and_grad_()

        # we process each scene in batch independently
        for i_scene in range(N_scenes):
            i_decision = 0
            scene_features = detached_features[i_scene]
            scene_images   = images[i_scene]

            # (N_per_scene choose 2) image pairs
            for i_image1 in range(N_per_scene):
                image1    = scene_images[i_image1]
                features1 = scene_features[i_image1]

                for i_image2 in range(i_image1+1, N_per_scene):
                    image2    = scene_images[i_image2]
                    features2 = scene_features[i_image2]

                    # establish the match distribution and calculate the
                    # gradient estimator
                    match_dist = matcher.match_pair(features1, features2)
                    loss, stats_ = self._loss_for_pair(match_dist, image1, image2)
                    # this .backward() will accumulate in `detached_features`
                    loss.backward()

                    stats[i_scene, i_decision] = stats_
                    i_decision += 1

        # here we "reattach" `detached_features` to the original `features`.
        # `torch.autograd.backward(leaves, grads)` API requires that we have
        # two equal length lists where for each grad-enabled leaf in `leaves`
        # we have a corresponding gradient tensor in `grads`
        leaves = []
        grads  = []
        for feat, detached_feat in zip(features.flat, detached_features.flat):
            leaves.extend(feat.grad_tensors())
            grads.extend([t.grad for t in detached_feat.grad_tensors()])
        #for i in range(features.size):
            #leaves.extend(features.flat[i].grad_tensors())
            #grads.extend([t.grad for t in detached_features.flat[i].grad_tensors()])

        # finally propagate the gradients down to the network
        torch.autograd.backward(leaves, grads)

        return stats
