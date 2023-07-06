# # coding=utf-8
# # Copyright (c) DIRECT Contributors

import math
import warnings
from typing import Callable, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function

from direct.data import transforms as T
from direct.nn.recurrent.recurrent import Conv2dGRU, NormConv2dGRU
from direct.utils.asserts import assert_positive_integer

import matplotlib.pyplot as plt


class StraightThroughPolicy(nn.Module):
    """
    Adaptive sampling policy for generating acquisition trajectories.
    """

    def __init__(
        self,
        budget: int,
        coil_dim: int,
        input_dim: tuple,
        num_actions: int = None,
        sigmoid_slope: float = 10,
        sampling_type: str = 'cartesian'
    ):
        """Inits :class:`StraightThroughPolicy`.

        Parameters
        ----------

        """
        super().__init__()
        
        assert sampling_type in ['cartesian', 'radial'], f"Sampling type should be one of \"cartesian\" or \"radial\". Got {sampling_type}"

        self.input_dim = input_dim
        self.binarizer = ThresholdSigmoidMask.apply
        self.budget = budget
        self.sigmoid_slope = sigmoid_slope        
        self.sampling_type = sampling_type
        self.num_actions = num_actions
        self._coil_dim = coil_dim
        
        # Keep track of which part of sampling space is already explored
        self.actions_taken = torch.zeros((self.num_actions)).cuda()
                
        input_dim_reduced = list(input_dim)
        input_dim_reduced.pop(coil_dim)

        B, M, W, H, C  = input_dim
        
        # For cartesian sampling, num_actions is the amount of vertical lines, so W
        if (self.sampling_type == 'cartesian' and num_actions is None):
            self.num_actions = W
        
        if (self.sampling_type == 'radial'):
            assert num_actions is not None, "Num_actions should be defined for radial sampler"
            self.radial_sampler = RadialSampler(size=(W, H))
            
        self.sampler = ActionSampler(
            input_dim=tuple(input_dim_reduced),
            num_actions=self.num_actions,
            sampling_type=self.sampling_type
        )

    def forward(
        self,
        kspace_pred: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        B, H, W, C = kspace_pred.shape
        flat_prob_mask = self.sampler(kspace_pred, mask, self.actions_taken)
        # self.actions_taken = new_actions
        
        # Take out zero (masked) probabilities, since we don't want to include
        # those in the normalisation
        # Don't squeeze the batch dimension!
        
        # print(mask.shape, mask.squeeze((1,4))[:,0,:].shape)
                
        nonzero_idcs = (mask.squeeze((1,4))[:,0,:].view(B, W) == 0).nonzero(as_tuple=True)
        
        # print('nonzero', len(nonzero_idcs), nonzero_idcs)
        
        probs_to_norm = flat_prob_mask[nonzero_idcs].reshape(B, -1)
                
        # Rescale probabilities to desired sparsity.
        normed_probs = self.rescale_probs(probs_to_norm)
        
        # Reassign to original array
        flat_prob_mask[nonzero_idcs] = normed_probs.flatten()
        
        flat_bin_mask = self.binarizer(
            flat_prob_mask, self.sigmoid_slope, False
        )
        return flat_bin_mask, flat_prob_mask

    def do_acquisition(
        self,
        kspace: torch.Tensor,
        kspace_pred: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ):
        
        print('In do_acquisition, has gradient', kspace.requires_grad)
        
        B, M, H, W, C = kspace.shape  # batch, coils, height, width, complex
        # BMHWC --> BHWC --> BCHW
        
        current_recon = T.reduce_operator(kspace_pred, sens_maps, dim=self._coil_dim)

        # BCHW --> BW --> B11W1
        actions, flat_prob_mask = self(current_recon, mask)
                
        if (self.sampling_type == 'cartesian'):
            acquisitions = actions.reshape(B, 1, 1, W, 1)
            
            print(torch.count_nonzero(acquisitions))
            
            prob_mask = flat_prob_mask.reshape(B, 1, 1, W, 1)

            # B11W1
            
            # print(mask.float() * acquisitions)
            
            # Assert that the intersection of the old and new mask is empty
            # assert torch.all((mask.float() * acquisitions).bool() == False)

            mask = mask.float() + acquisitions
                        
        if (self.sampling_type == 'radial'):                        
            radial_acquisitions = self.radial_sampler(actions)
            mask = mask.float() + radial_acquisitions.reshape(1, 1, H, W, 1)
                        
            # Ensure all values are 1.0 or 0.0
            mask = mask.bool().float()
            
            prob_mask = flat_prob_mask.reshape(B, 1, 1, W, 1)
                        
            # self.actions_taken += actions

                
        # BMHWC        
        masked_kspace = mask * kspace

        fix_sign_leakage_mask = torch.where(
            torch.bitwise_and(kspace < 0.0, mask == 0.0), -1.0, 1.0
        )
        masked_kspace = masked_kspace * fix_sign_leakage_mask
       
        print('mask shape', mask.shape)
        accel = torch.numel(mask) / torch.count_nonzero(mask.float())
        print(f'Acceleration {accel:.2f}')

        return mask, masked_kspace, prob_mask
    
    def rescale_probs(self, batch_x: torch.Tensor):
        """
        Rescale Probability Map
        given a prob map x, rescales it so that it obtains the desired sparsity,
        specified by self.budget and the image size.

        if mean(x) > sparsity, then rescaling is easy: x' = x * sparsity / mean(x)
        if mean(x) < sparsity, one can basically do the same thing by rescaling
                                (1-x) appropriately, then taking 1 minus the result.
        """

        batch_size, W = batch_x.shape
                
        sparsity = self.budget / W
        ret = []
        for i in range(batch_size):
            x = batch_x[i : i + 1]
            xbar = torch.mean(x)
            r = sparsity / (xbar)
            beta = (1 - sparsity) / (1 - xbar)

            # compute adjustment
            le = torch.le(r, 1).float()
            ret.append(le * x * r + (1 - le) * (1 - (1 - x) * beta))

        return torch.cat(ret, dim=0)

class ThresholdSigmoidMask(Function):
    def __init__(self):
        """
        Straight through estimator.
        The forward step stochastically binarizes the probability mask.
        The backward step estimate the non differentiable > operator using sigmoid with large slope (10).
        """
        super(ThresholdSigmoidMask, self).__init__()

    @staticmethod
    def forward(ctx, inputs, slope, clamp):
        batch_size = len(inputs)
        probs = []
        results = []

        for i in range(batch_size):
            x = inputs[i : i + 1]
            count = 0
            while True:
                prob = x.new(x.size()).uniform_()
                result = (x > prob).float()

                if torch.isclose(torch.mean(result), torch.mean(x), atol=1e-3):
                    break
                count += 1
                if count > 10000:
                    raise RuntimeError(
                        "Rejection sampled exceeded number of tries. Probably this means all "
                        "sampling probabilities are 1 or 0 for some reason, leading to divide "
                        "by zero in rescale_probs()."
                    )
            probs.append(prob)
            results.append(result)
        results = torch.cat(results, dim=0)
        probs = torch.cat(probs, dim=0)

        slope = torch.tensor(slope, requires_grad=False)
        ctx.clamp = clamp
        ctx.save_for_backward(inputs, probs, slope)
        return results

    @staticmethod
    def backward(ctx, grad_output):
        input, prob, slope = ctx.saved_tensors
        if ctx.clamp:
            grad_output = F.hardtanh(grad_output)
        # derivative of sigmoid function
        current_grad = (
            slope
            * torch.exp(-slope * (input - prob))
            / torch.pow((torch.exp(-slope * (input - prob)) + 1), 2)
        )
        return current_grad * grad_output, None, None
    
    
class RadialSampler(nn.Module):
    """
    Sampler returning one sampled line going through the origin,
    with variable angle and thickness.
    The default parameters are suitable for a 320x320 image, for
    full sampling coverage.
    """
    def __init__ (
        self,
        num_rays: int = 680,
        eps: float = 0.0031351,
        size: tuple = (320,320)
    ):
        super().__init__()
        
        assert len(size) == 2, 'Size used for sampler should have two dimensions'
        self.size= size
        
        # Equivalent to np.linspace(..., endpoint=False)
        self.angles = torch.linspace(0, 2 * torch.pi, num_rays + 1)[:-1].cuda()
        
        self.eps = eps
    
    def forward(self, actions):
        """
        Parameters:
        ------------
        actions: Binary Tensor of length same as self.num_rays that indicates which sampling actions should be taken
        """
        H, W = self.size
        x = torch.linspace(-1, 1, W)
        y = torch.linspace(-1, 1, H)
        xv, yv = torch.meshgrid(x, y, indexing='xy')
        xv = xv.cuda()
        yv = yv.cuda()
        
        action_indices = torch.nonzero(actions.squeeze(), as_tuple=True)
        sampled_angles = self.angles[action_indices]
        masks = [(torch.abs(xv * torch.cos(angle) + yv * torch.sin(angle)) < self.eps).unsqueeze(-1).float() for angle in sampled_angles]
        masks = torch.cat(masks, axis=-1)
        mask = torch.sum(masks, axis=-1)
        
        circular_mask = ((xv * (W/2)) ** 2 + (yv * (H/2)) ** 2) < (W / 2) ** 2
        mask *= circular_mask.float()
        
        mask = mask.bool()
        return mask
        

class ActionSampler(nn.Module):
    """
    Sampler returning 1d sampling salience, corresponding to salience of performing one of the possible sampling actions.
    Based on current k-space prediction.
    """
    def __init__ (
        self,
        sampling_type: str,
        input_dim: tuple = (1, 320, 320, 2),
        chans: int = 16,
        num_conv_layers: int = 3,
        drop_prob: float = 0,
        sigmoid_slope: float = 10,
        intermediate_fc_size: int = 1024,
        num_fc_layers: int = 3,
        use_adaptive_avg_pool: bool = True,
        adaptive_avg_pool_size: tuple = (32,32),
        num_actions: int = 320,
    ):
        super().__init__()
        
        # TODO: would be nicer to do this somewhere else
        self.permuted_input_dim = tuple([input_dim[0], input_dim[3], input_dim[1], input_dim[2]])
                
        self.in_chans = self.permuted_input_dim[1]
        
        self.sampling_type = sampling_type
        self.num_actions = num_actions     
        self.chans = chans
        self.num_conv_layers = num_conv_layers
        self.sigmoid_slope = sigmoid_slope
        self.intermediate_fc_size = intermediate_fc_size
        self.adaptive_avg_pool_size = adaptive_avg_pool_size
        self.use_adaptive_avg_pool = use_adaptive_avg_pool
        
        if self.use_adaptive_avg_pool:
            assert len(self.adaptive_avg_pool_size) == 2, "Adaptive Avg. Pool size needs to be two-dimensional"
                
        layers = []
        
        # Initial from in_chans to chans
        layers.extend([
            nn.Conv2d(
                self.in_chans,
                chans,
                kernel_size=3,
                padding=1
            ),
            nn.InstanceNorm2d(chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
        ])

        for i in range(num_conv_layers): 
            layers.extend([
                nn.Conv2d(
                    chans * 2 ** i,
                    chans * 2 ** (i + 1),
                    kernel_size=3,
                    padding=1
                ),
                nn.InstanceNorm2d(chans * 2 ** (i + 1)),
                nn.ReLU(),
                nn.Dropout2d(drop_prob),
            ])

        if (self.use_adaptive_avg_pool):
            layers.append(nn.AdaptiveAvgPool2d(self.adaptive_avg_pool_size))
            
        self.feature_extractor = nn.Sequential(*layers)
        
        in_features_fc = None
            
        # If using AdaptiveAveragePool, the output of the conv. layers is fixed based on thhe hyperparameters
        if (self.use_adaptive_avg_pool):
            in_features_fc = self.in_chans ** self.num_conv_layers * self.chans * self.adaptive_avg_pool_size[0] * self.adaptive_avg_pool_size[1]
        else:
            # Dynamically determine size of input to the linear layers
            dummy = torch.randn(self.permuted_input_dim)
            out = self.feature_extractor(dummy)
            in_features_fc = out.flatten().shape[0]
        
        # Edge case: only 1 layer
        if (num_fc_layers == 1):
            fc_out = [nn.Linear(in_features=in_features_fc, out_features=self.num_actions)]
        else:
            fc_out = [nn.Linear(in_features=in_features_fc, out_features=self.intermediate_fc_size)]
            for i in range(num_fc_layers - 1):
                
                # Last layer has out_features equal to number of actions to sample
                if (i == num_fc_layers - 2):
                    fc_out.append(nn.Linear(in_features=self.intermediate_fc_size, out_features=self.num_actions))
                    break
                fc_out.append(nn.Linear(in_features=self.intermediate_fc_size, out_features=self.intermediate_fc_size))

        # Fully connected output layers
        self.fc_out = nn.Sequential(*fc_out)
            
    def forward(self, kspace_pred, mask, actions_taken):
        
        # Channels should be second dimension for nn.Conv2d
        kspace_pred = torch.permute(kspace_pred, (0,3,1,2))
                
        kspace_pred_emb = self.feature_extractor(kspace_pred)
        kspace_pred_emb = kspace_pred_emb.flatten(start_dim=1)
        out = self.fc_out(kspace_pred_emb)
        prob_mask = torch.sigmoid(self.sigmoid_slope * out)
        
        mask = mask.float()        
        
        # Mask out already sampled rows. If cartesian, we can simply look at the mask
        # Do not squeeze the batch dimension! BMWHC
        if (self.sampling_type == 'cartesian'): 
            # print(mask.shape, mask.squeeze((1,4)).shape)
            prob_mask = prob_mask * ( 1 - mask.squeeze((1,4))[:,0,:] )
            
        # Otherwise, we have stored which actions have been taken here
        if (self.sampling_type == 'radial'):
            pass
            # print('nonzero', torch.count_nonzero(actions_taken))
            # print(prob_mask)
#             print('mysterio dos', prob_mask.shape, self.actions_taken.shape)
#             print(prob_mask, self.actions_taken)
#             prob_mask = prob_mask * ( 1 - self.actions_taken )
            
        assert len(prob_mask.shape) == 2
        
        return prob_mask