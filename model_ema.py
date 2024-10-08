"""This is a script copy from work FDT: https://github.com/AngusDujw/FTD-distillation/tree/main?tab=readme-ov-file"""
""" Exponential Moving Average (EMA) of model updates

Hacked together by / Copyright 2020 Ross Wightman
"""
import logging
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn

_logger = logging.getLogger(__name__)


class ModelEma:
    """ Model Exponential Moving Average (DEPRECATED)

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This version is deprecated, it does not work with scripted models. Will be removed eventually.

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.9999, device='', resume=''):
        # make a copy of the model for accumulating moving average of weights
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if device:
            self.ema.to(device=device)
        self.ema_has_module = hasattr(self.ema, 'module')
        if resume:
            self._load_checkpoint(resume)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert isinstance(checkpoint, dict)
        if 'state_dict_ema' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict_ema'].items():
                # ema model may have been wrapped by DataParallel, and need module prefix
                if self.ema_has_module:
                    name = 'module.' + k if not k.startswith('module') else k
                else:
                    name = k
                new_state_dict[name] = v
            self.ema.load_state_dict(new_state_dict)
            _logger.info("Loaded state_dict_ema")
        else:
            _logger.warning("Failed to find state_dict_ema, starting from loaded model weights")

    def update(self, model):
        # correct a mismatch in state dict keys
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)


class ModelEmaV2(nn.Module):
    """ Model Exponential Moving Average V2

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, params, decay=0.9999, device=None,updating_list=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the params for accumulating moving average of weights
        self.module = deepcopy(params)
        #self.module.eval()
        self.decay = decay
        self.updating_list = updating_list
        self.device = device  # perform ema on different device from params if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, params, update_fn):
        with torch.no_grad():
            count = 0
            for ema_v, params_v in zip(self.module, params):
                if self.updating_list is not None and ema_v.dim() > 1: 
                    count += 1
                    if not count in self.updating_list: 
                        decay = 0
                    else: 
                        decay = self.decay
                    # this is for the updating part EMA use, comment it for using set method 
                    # update_fn = lambda e, m: decay * e + (1. - decay) * m
                if self.device is not None:
                    params_v = params_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, params_v))

    def update(self, params):
        self._update(params, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, params):
        self._update(params, update_fn=lambda e, m: m)
