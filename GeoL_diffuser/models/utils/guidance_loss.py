import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

############## GUIDANCE config ########################
class GuidanceConfig(object):
    def __init__(self, name, weight, params, agents, func=None):
        '''
        - name : name of the guidance function (i.e. the type of guidance), must be in GUIDANCE_FUNC_MAP
        - weight : alpha weight, how much affects denoising
        - params : guidance loss specific parameters
        - agents : agent indices within the scene to apply this guidance to. Applies to ALL if is None.
        - func : the function to call to evaluate this guidance loss value.
        '''
        assert name in GUIDANCE_FUNC_MAP, 'Guidance name must be one of: ' + ', '.join(map(str, GUIDANCE_FUNC_MAP.keys()))
        self.name = name
        self.weight = weight
        self.params = params
        self.agents = agents
        self.func = func

    @staticmethod
    def from_dict(config_dict):
        assert config_dict.keys() == {'name', 'weight', 'params', 'agents'}, \
                'Guidance config must include only [name, weight, params, agt_mask]. agt_mask may be None if applies to all agents in a scene'
        return GuidanceConfig(**config_dict)

    def __repr__(self):
        return '<\n%s\n>' % str('\n '.join('%s : %s' % (k, repr(v)) for (k, v) in self.__dict__.items()))

def verify_guidance_config_list(guidance_config_list):
    '''
    Returns true if there list contains some valid guidance that needs to be applied.
    Does not check to make sure each guidance dict is structured properly, only that
    the list structure is valid.
    '''
    assert len(guidance_config_list) > 0
    valid_guidance = False
    for guide in guidance_config_list:
        valid_guidance = valid_guidance or len(guide) > 0
    return valid_guidance


############## GUIDANCE functions ########################

class GuidanceLoss(nn.Module):
    '''
    Abstract guidance function. This is a loss (not a reward), i.e. guidance will seek to
    MINIMIZE the implemented function.
    '''
    def __init__(self):
        super().__init__()
        self.global_t = 0

    def init_for_batch(self, example_batch):
        '''
        Initializes this loss to be used repeatedly only for the given scenes/agents in the example_batch.
        e.g. this function could use the extents of agents or num agents in each scene to cache information
              that is used while evaluating the loss
        '''
        pass

    def update(self, global_t=None):
        '''
        Update any persistant state needed by guidance loss functions.
        - global_t : the current global timestep of rollout
        '''
        if global_t is not None:
            self.global_t = global_t

    def forward(self, x, data_batch, agt_mask=None):
        '''
        Computes and returns loss value.

        Inputs:
        - x : the current trajectory (B, N, T, 6) where N is the number of samples and 6 is (x, y, vel, yaw, acc, yawvel)
        - data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        - agt_mask : size B boolean list specifying which agents to apply guidance to. Applies to ALL agents if is None.

        Output:
        - loss : (B, N) loss for each sample of each batch index. Final loss will be mean of this.
        '''
        raise NotImplementedError('Must implement guidance function evaluation')




############## GUIDANCE utilities ########################

GUIDANCE_FUNC_MAP = {


    'map_collision' : MapCollisionLoss,
    'target_pos' : TargetPosLoss,
    'target_rotation' : TargetRotationLoss,
    'global_target_pos' : GlobalTargetPosLoss,

}

class DiffuserGuidance(object):
    '''
    Handles initializing guidance functions and computing gradients at test-time.
    '''
    def __init__(self, guidance_config_list, example_batch=None):
        '''
        - example_batch [optional] - if this guidance will only be used on a single batch repeatedly,
                                    i.e. the same set of scenes/agents, an example data batch can
                                    be passed in a used to init some guidance making test-time more efficient.
        '''
        self.num_scenes = len(guidance_config_list)
        assert self.num_scenes > 0, "Guidance config list must include list of guidance for each scene"
        self.guide_configs = [[]]*self.num_scenes
        for si in range(self.num_scenes):
            if len(guidance_config_list[si]) > 0:
                self.guide_configs[si] = [GuidanceConfig.from_dict(cur_cfg) for cur_cfg in guidance_config_list[si]]
                # initialize each guidance function
                for guide_cfg in self.guide_configs[si]:
                    guide_cfg.func = GUIDANCE_FUNC_MAP[guide_cfg.name](**guide_cfg.params)
                    if example_batch is not None:
                        guide_cfg.func.init_for_batch(example_batch)


    def compute_guidance_loss(self, x_loss, data_batch):
        '''
        Evaluates all guidance losses and total and individual values.
        - x_loss: (B, N, T, 6) the trajectory to use to compute losses and 6 is (x, y, vel, yaw, acc, yawvel)
        - data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        '''
        bsize, num_samp, _, _ = x_loss.size()
        guide_losses = dict()
        loss_tot = 0.0
        # NOTE: unique_consecutive is important here to avoid sorting by torch.unique which may shuffle the scene ordering
        #       and breaks correspondence with guide_configs
        _, local_scene_index = torch.unique_consecutive(data_batch['scene_index'], return_inverse=True)
        for si in range(self.num_scenes):
            cur_guide = self.guide_configs[si]
            if len(cur_guide) > 0:
                # mask out non-current current scene
                for gidx, guide_cfg in enumerate(cur_guide):
                    agt_mask = local_scene_index == si
                    if guide_cfg.agents is not None:
                        # mask out non-requested agents within the scene
                        cur_scene_inds = torch.nonzero(agt_mask, as_tuple=True)[0]
                        agt_mask_inds = cur_scene_inds[guide_cfg.agents]
                        agt_mask = torch.zeros_like(agt_mask)
                        agt_mask[agt_mask_inds] = True
                    # compute loss
                    cur_loss = guide_cfg.func(x_loss, data_batch,
                                            agt_mask=agt_mask)
                    indiv_loss = torch.ones((bsize, num_samp)).to(cur_loss.device) * np.nan # return indiv loss for whole batch, not just masked ones
                    indiv_loss[agt_mask] = cur_loss.detach().clone()
                    guide_losses[guide_cfg.name + '_scene_%03d_%02d' % (si, gidx)] = indiv_loss
                    loss_tot = loss_tot + torch.mean(cur_loss) * guide_cfg.weight

        return loss_tot, guide_losses

    def update(self, **kwargs):
        for si in range(self.num_scenes):
            cur_guide = self.guide_configs[si]
            if len(cur_guide) > 0:
                for guide_cfg in cur_guide:
                    guide_cfg.func.update(**kwargs)