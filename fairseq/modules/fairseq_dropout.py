# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
import torch

logger = logging.getLogger(__name__)


class FairseqDropout(nn.Module):
    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

        self.dropout_mask = None

    def forward(self, x, inplace: bool = False):
        if self.p > 0 and (self.training or self.apply_during_inference):
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x

    def forward_mask_fixed(self, x, inplace: bool = False):
        if self.p > 0 and (self.training or self.apply_during_inference):
            if self.dropout_mask is None:
                self.dropout_mask = torch.bernoulli(torch.full_like(x, 1 - self.p))

            if self.dropout_mask.size(0) < x.size(0):
                temp_shape = (x.size(0) - self.dropout_mask.size(0),
                              self.dropout_mask.size(1),
                              self.dropout_mask.size(2))
                temp_mask = torch.bernoulli(torch.full(temp_shape, 1 - self.p))
                self.dropout_mask = torch.cat([self.dropout_mask, temp_mask], dim=0)
            if self.dropout_mask.size(1) < x.size(1):
                temp_shape = (self.dropout_mask.size(0),
                              x.size(1) - self.dropout_mask.size(1),
                              self.dropout_mask.size(2))
                temp_mask = torch.bernoulli(torch.full(temp_shape, 1 - self.p))
                self.dropout_mask = torch.cat([self.dropout_mask, temp_mask], dim=1)

            return x * self.dropout_mask[:min(x.size(0), self.dropout_mask.size(0)),
                       :min(x.size(1), self.dropout_mask.size(1)), :]
        else:
            return x

    def reset_dropout_mask(self):
        self.dropout_mask = None

    def make_generation_fast_(
            self,
            name: str,
            retain_dropout: bool = False,
            retain_dropout_modules: Optional[List[str]] = None,
            **kwargs
    ):
        if retain_dropout:
            if retain_dropout_modules is not None and self.module_name is None:
                logger.warning(
                    "Cannot enable dropout during inference for module {} "
                    "because module_name was not set".format(name)
                )
            elif (
                    retain_dropout_modules is None  # if None, apply to all modules
                    or self.module_name in retain_dropout_modules
            ):
                logger.info(
                    "Enabling dropout during inference for module: {}".format(name)
                )
                self.apply_during_inference = True
            else:
                logger.info("Disabling dropout for module: {}".format(name))
