from __future__ import absolute_import

from .triplet import TripletLoss, SoftTripletLoss, SoftTripletLoss_weight
from .crossentropy import CrossEntropyLabelSmooth, CrossEntropyLabelSmooth_weighted

__all__ = [
    'TripletLoss',
    'CrossEntropyLabelSmooth',
    'SoftTripletLoss',
    'CrossEntropyLabelSmooth_weighted',
    'SoftTripletLoss_weight',

]
