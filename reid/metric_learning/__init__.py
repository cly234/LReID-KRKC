from __future__ import absolute_import

from metric_learn import (ITML_Supervised, LMNN, LSML_Supervised,
                          SDML_Supervised, NCA, LFDA, RCA_Supervised)

__factory = {
    'itml': ITML_Supervised,
    'lmnn': LMNN,
    'lsml': LSML_Supervised,
    'sdml': SDML_Supervised,
    'nca': NCA,
    'lfda': LFDA,
    'rca': RCA_Supervised,
}


def get_metric(algorithm, *args, **kwargs):
    if algorithm not in __factory:
        raise KeyError("Unknown metric:", algorithm)
    return __factory[algorithm](*args, **kwargs)
