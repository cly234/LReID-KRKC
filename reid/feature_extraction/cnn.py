from __future__ import absolute_import
from collections import OrderedDict
import torch

from ..utils import to_torch

def extract_cnn_feature(model, inputs):
    model.eval()
    with torch.no_grad():
        inputs = to_torch(inputs).cuda()
        outputs = model(inputs)
        outputs = outputs.data.cpu()
        return outputs

