# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

import typing
from pruning.pruned_model import PrunedModel


def vectorize(state_dict: typing.Dict[str, torch.Tensor]):
    """Convert a state dict into a single column Tensor in a repeatable way."""

    return torch.cat([state_dict[k].reshape(-1) for k in sorted(state_dict.keys())])


def unvectorize(vector: torch.Tensor, reference_state_dict: typing.Dict[str, torch.Tensor]):
    """Convert a vector back into a state dict with the same shapes as reference state_dict."""

    if len(vector.shape) > 1: raise ValueError('vector has more than one dimension.')

    state_dict = {}
    for k in sorted(reference_state_dict.keys()):
        if vector.nelement() == 0: raise ValueError('Ran out of values.')

        size, shape = reference_state_dict[k].nelement(), reference_state_dict[k].shape
        this, vector = vector[:size], vector[size:]
        state_dict[k] = this.reshape(shape)

    if vector.nelement() > 0: raise ValueError('Excess values.')
    return state_dict


def perm(N, seed: int = None):
    """Generate a tensor with the numbers 0 through N-1 ordered randomly."""

    gen = torch.Generator()
    if seed is not None: gen.manual_seed(seed)
    perm = torch.normal(torch.zeros(N), torch.ones(N), generator=gen)
    return torch.argsort(perm)


def shuffle_tensor(tensor: torch.Tensor, seed: int = None):
    """Randomly shuffle the elements of a tensor."""

    shape = tensor.shape
    return tensor.reshape(-1)[perm(tensor.nelement(), seed=seed)].reshape(shape)


def shuffle_state_dict(state_dict: typing.Dict[str, torch.Tensor], seed: int = None):
    """Randomly shuffle each of the tensors in a state_dict."""

    output = {}
    for i, k in enumerate(sorted(state_dict.keys())):
        output[k] = shuffle_tensor(state_dict[k], seed=None if seed is None else seed+i)
    return output

def erank(M):
    # U, S, Vt = torch.svd(M, compute_uv=False)
    S = torch.linalg.svdvals(M)
    S[S <= 1e-6] = 0
    normalized = S / S.sum()
    return -(normalized[normalized > 0] * torch.log(normalized[normalized > 0])).sum().item()


def min_singular(M):
    S = torch.linalg.svdvals(M)
    return S[-1].item()


def condition(M):
    U, S, Vt = torch.svd(M, compute_uv=False)
    return (S[0] / S[S >= 1e-6][-1]).item()


def mutual_coherence(M):
    W = torch.nn.functional.normalize(M, dim=1)
    W_adj = W @ W.T - torch.eye(W.size()[0], device=W.device)
    return (W_adj.abs().max()).item()


def weight_erank(state_dict: typing.Dict[str, torch.Tensor]):
    output = {}
    for k, v in state_dict.items():
           output[k] = erank(v)
    return output

def activation(model: PrunedModel, input, conv_layers=False):
    features = model.intermediate(input, conv_layers)
    output = []
    for f in features[0:-1]:
        output.append((f > 1e-8).float().mean().item())
    return output


def activation_mean(model: PrunedModel, input, conv_layers=False):
    features = model.intermediate(input, conv_layers)
    total_activation = 0
    total_size = 0
    for f in features[0:-1]:
        total_activation += (f > 1e-8).float().sum()
        total_size += f.numel()
    return (total_activation / total_size).item()


def gradient_mean(model: PrunedModel, input, labels, conv_layers=False):
    # features = model.intermediate(input, conv_layers)
    l = model.loss_criterion(model(input), labels)
    l.backward()

    grad_mean = []
    for n, p in model.named_parameters():
        if n[6:] in model.prunable_layer_names:
            grad_mean.append(p.grad[p.abs() > 0].abs().mean().item())
    return grad_mean


def features_spectral(model: PrunedModel, input, conv_layers=False, no_activation=False):
    features = model.intermediate(input, conv_layers, no_activation)
    output = []
    for f in features:
        output.append(torch.linalg.norm(f, ord=2).item())
    return output


def features_frobenius(model: PrunedModel, input, conv_layers=False, no_activation=False):
    features = model.intermediate(input, conv_layers, no_activation)
    output = []
    for f in features:
        output.append(torch.linalg.norm(f, ord='fro').item())
    return output


def features_spectral_fro_ratio(model: PrunedModel, input, conv_layers=False, no_activation=False):
    features = model.intermediate(input, conv_layers, no_activation)
    output = []
    for f in features:
        output.append(torch.linalg.norm(f, ord=2).item() / torch.linalg.norm(f, ord='fro').item())
    return output


def generate_mask_active(w, p, seed, input):
    mask = torch.zeros_like(w)
    torch.manual_seed(seed)
    if len(input.shape) > 2:
        input = torch.flatten(input, 1)

    for d in range(w.shape[0]):
        mask[d, :] = ((input * w[d, :]) > 0).sum(dim=0)

    mask_norm = torch.zeros_like(w)
    mask_norm[mask > 0] = mask[mask > 0] * (p / mask[mask > 0].mean())
    rand = torch.rand_like(w)
    return rand < mask_norm



def feature_erank(model: PrunedModel, input, conv_layers=False, no_activation=False):
    features = model.intermediate(input, conv_layers, no_activation)
    output = []
    for f in features:
        output.append(erank(f))
    return output
