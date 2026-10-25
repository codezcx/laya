"""Keep BatchNorm statistics fixed without freezing affine weights or the encoder."""
import torch


def apply_bn_policy(module, policy):
    if policy not in {'updating','frozen'}:
        raise ValueError(f'Unknown BatchNorm policy: {policy}')
    if policy=='frozen':
        for child in module.modules():
            if isinstance(child,torch.nn.modules.batchnorm._BatchNorm):
                child.eval()


def bn_summary(module):
    result={}
    for name,child in module.named_modules():
        if isinstance(child,torch.nn.modules.batchnorm._BatchNorm):
            result[name]=dict(training=child.training,
                             batches=int(child.num_batches_tracked),
                             running_mean_norm=float(child.running_mean.float().norm()),
                             running_var_mean=float(child.running_var.float().mean()),
                             affine_trainable=child.weight.requires_grad)
    return result
