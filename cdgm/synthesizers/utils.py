import torch
from torch import nn
import numpy as np
from packaging import version
from torch.nn import functional

from cdgm.constraints_code.compute_sets_of_constraints import compute_sets_of_constraints
from cdgm.constraints_code.parser import parse_constraints_file
from cdgm.constraints_code.feature_orderings import set_ordering
from cdgm.constraints_code.correct_predictions import correct_preds


class LossPert(nn.Module):
    def forward(self, orig, adv):
        return torch.mean(torch.norm(orig - adv, 2, dim=-1))

class LossAdv(nn.Module):
    def forward(self, probs, labels, num_labels):
        onehot_labels = torch.eye(num_labels)[labels]
        real = torch.sum(probs * onehot_labels, dim=1)
        other, _ = torch.max((1 - onehot_labels) * probs, dim=1)
        zeros = torch.zeros_like(other)

        ## Untargeted
        loss = torch.max(real-other, zeros).mean()
        return loss











def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    """Deals with the instability of the gumbel_softmax for older versions of torch.

    For more details about the issue:
    https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

    Args:
        logits […, num_features]:
            Unnormalized log probabilities
        tau:
            Non-negative scalar temperature
        hard (bool):
            If True, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
        dim (int):
            A dimension along which softmax will be computed. Default: -1.

    Returns:
        Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
    """
    if version.parse(torch.__version__) < version.parse('1.2.0'):
        for i in range(10):
            transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard,
                                                    eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed
        raise ValueError('gumbel_softmax returning NaN.')

    return functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)


def apply_activate(transformer, data):
    """Apply proper activation function to the output of the generator."""
    data_t = []
    st = 0
    for column_info in transformer.output_info_list:
        for span_info in column_info:
            if span_info.activation_fn == 'tanh':
                ed = st + span_info.dim
                data_t.append(torch.tanh(data[:, st:ed]))
                st = ed
            elif span_info.activation_fn == 'softmax':
                ed = st + span_info.dim
                transformed = _gumbel_softmax(data[:, st:ed], tau=0.2, hard=True)
                data_t.append(transformed)
                st = ed
            else:
                raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')
    return torch.cat(data_t, dim=1)

def apply_constrained(transformer, data, ordering, sets_of_constr, args):
    inverse = transformer.inverse_transform(data)
    for i in range(inverse.shape[1]):
        inverse[:,i] = round_func_BPDA(inverse[:,i], args.round_decs[i])
    constrained = correct_preds(inverse, ordering, sets_of_constr)
    transformed = transformer.transform(constrained, data)
    return transformed, inverse
        
def get_sets_constraints(model, use_case, label_ordering_choice, constraints_file):
    ordering, constraints = parse_constraints_file(constraints_file)
    ordering = set_ordering(use_case, ordering, label_ordering_choice, model)
    sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)
    return constraints, sets_of_constr, ordering

def adversarial_loss(probs, labels, num_labels, targeted):
    """
    We implement the f6 from "Towards Evaluating the Robustness of Neural Networks", Nicholas Carlini & David Wagner.
    The loss is essential the difference between the maximum of logits of classes other than the target class and logits of the target
    class. Minimizing the loss in turns maximizes the difference which usually means maximize the logits of the target class
    and minimize logits of other classes.

    Inputs:
    model: target model
    x: perturbed input
    target_y: target class
    """
    confidence = 0
    if targeted:
        onehot_labels = torch.eye(num_labels)[1-labels]
    else:
        onehot_labels = torch.eye(num_labels)[labels]
    target = torch.sum(probs * onehot_labels, dim=1)
    #other, _ = torch.max((1 - onehot_labels) * probs, dim=1)
    other, _ = torch.max((1 - onehot_labels) * probs, dim=1)
    zeros = torch.zeros_like(other)

    ## Untargeted
    if not targeted:
        loss = torch.max((target-other)+confidence, zeros).mean()
    else:
        loss = torch.max(other-target, zeros).mean()
    return loss

def round_func_BPDA(input, dec):
    # This is equivalent to replacing round function (non-differentiable) with
    # an identity function (differentiable) only when backward.
    forward_value = torch.round(input, decimals=dec)
    out = input.clone()
    out.data = forward_value.data
    return out


def get_discrete_col(transformer):
    discrete_cols = []
    st = 0
    for column_info in transformer.output_info_list:
        for span_info in column_info:
            if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                #not discrete column
                st += span_info.dim
            else:
                ed = st + span_info.dim
                discrete_cols.append((st,ed))
        st += span_info.dim
    return discrete_cols

def get_modes_idx(transformer):
    mode_idx = []
    st = 0
    for column_info in transformer._column_transform_info_list:
        if column_info.column_type=="continuous":
            ed = st + column_info.output_dimensions
            mode_idx.extend(list(np.arange(st+1,ed)))
            st = ed 
        else:           
            st = st + column_info.output_dimensions
    return mode_idx


def get_not_modif_idx(transformer, not_modifiable):
    not_mod_idx = []
    st = 0
    for idx, column_info in enumerate(transformer._column_transform_info_list):
        if idx in not_modifiable:
            ed = st + column_info.output_dimensions
            not_mod_idx.extend(list(np.arange(st,ed)))
            st = ed 
        else:           
            st = st + column_info.output_dimensions
    return not_mod_idx
