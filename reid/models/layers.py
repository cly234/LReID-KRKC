import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import Parameter
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import DataParallel as DataParallel_
from collections import OrderedDict
from torch.nn.parallel.scatter_gather import scatter_kwargs


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def param_classifier(self):
        for name, param in self.named_params(self):
            if name == 'classifier.weight':
                yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                if p.requires_grad is not False:
                    yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):

                name_t, param_t = tgt
                grad = src

                if first_order:
                    grad = to_var(grad.detach().data)
                if grad is not None:  # ignore classifier's weight which is not used
                    tmp = param_t - lr_inner * grad

                # print(type(tmp))
                self.set_param(self, name_t, tmp)
        else:
            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)

class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        #self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight)

    def named_leaves(self):
        return [('weight', self.weight)]

class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

import torch
import warnings

from torch.nn import DataParallel as DataParallel_
from collections import OrderedDict

from torch.nn.parallel import parallel_apply
from torch.nn.parallel.scatter_gather import scatter_kwargs
from torch.nn.parallel.replicate import _broadcast_coalesced_reshape


class DataParallel(DataParallel_, MetaModule):
    __doc__ = DataParallel_.__doc__

    def scatter(self, inputs, kwargs, device_ids):
        if not isinstance(self.module, MetaModule):
            return super(DataParallel, self).scatter(inputs, kwargs, device_ids)

        params = kwargs.pop('params', None)
        inputs_, kwargs_ = scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)
        # Add params argument unchanged back in kwargs
        replicas = self._replicate_params(params, inputs_, device_ids,
                                          detach=not torch.is_grad_enabled())
        kwargs_ = tuple(dict(params=replica, **kwarg)
                        for (kwarg, replica) in zip(kwargs_, replicas))
        return inputs_, kwargs_

    def _replicate_params(self, params, inputs, device_ids, detach=False):
        if params is None:
            module_params = OrderedDict(self.module.named_parameters())
        else:
            # Temporarily disable the warning if no parameter with key prefix
            # `module` was found. In that case, the original params dictionary
            # is used.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                module_params = self.get_subdict(params, key='module')
            if module_params is None:
                module_params = params

        replicas = _broadcast_coalesced_reshape(list(module_params.values()),
                                                device_ids[:len(inputs)],
                                                detach)
        replicas = tuple(OrderedDict(zip(module_params.keys(), replica))
                         for replica in replicas)
        return replicas
