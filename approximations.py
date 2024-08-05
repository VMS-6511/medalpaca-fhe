import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from torch import Tensor, nn

def invert(input_tensor, d=5):
    # print("INSIDE Invert")
    a_i = 2 - input_tensor
    b_i = 1 - input_tensor

    for i in range(d):
        b_i = b_i ** 2
        a_i = a_i * (b_i + 1)

    return a_i

def sign(input_tensor):
    # print("INSIDE Sign")
    # f4_coeffs and g4_coeffs as tensors
    f4_coeffs = torch.tensor([2.4609375, -3.28125, 2.953125, -1.40625, 0.2734375])
    g4_coeffs = torch.tensor([5.712890625, -34.154296875, 94.7412109375, -110.83203125, 45.5302734375])

    def eval_composite(input_tensor, coeffs):
        total_degree = 9
        input_pows = [input_tensor ** (i + 1) for i in range(total_degree)]

        result = input_tensor * coeffs[0]
        for coeff_ind in range(1, len(coeffs)):
            deg_ind = coeff_ind * 2
            result += input_pows[deg_ind] * coeffs[coeff_ind]

        return result

    g4 = eval_composite(input_tensor, g4_coeffs)
    g4g4 = eval_composite(g4, g4_coeffs)
    f4g4g4 = eval_composite(g4g4, f4_coeffs)
    result = eval_composite(f4g4g4, f4_coeffs)

    return result

def exponentiate(input_tensor, r=10):
    # print("INSIDE Exponentiate")
    x_scaled_down = input_tensor * (1.0 / 2**r) + 1
    for i in range(r):
        x_scaled_down = x_scaled_down ** 2

    return x_scaled_down

def quick_operation(input_tensor, n, operation):
    # print(f"INSIDE QuickOperation op = {operation}")
    num_slots = input_tensor.size(0)
    sparse = n != num_slots
    log_n = int(torch.ceil(torch.log2(torch.tensor(n))))
    nearest_pow2 = 2 ** log_n

    if nearest_pow2 == num_slots:
        result = input_tensor.clone()
    else:
        rotated = torch.roll(input_tensor, -nearest_pow2)
        result = input_tensor + rotated

    for i in range(log_n):
        shift = 2 ** i
        rotated_s = torch.roll(result, shift)
        if operation == "MAX":
            res_plus_rot = result + rotated_s
            res_minus_rot = result - rotated_s
            sgn_output = sign(res_minus_rot)
            result = (res_plus_rot + res_minus_rot * sgn_output) * 0.5
        elif operation == "SUM":
            result += rotated_s

    if sparse:
        mask = torch.cat([torch.ones(n), torch.zeros(num_slots - n)])
        result *= mask

    return result

def zero_out_tail(input_tensor, n):
    # print("INSIDE ZeroOutTail")
    num_slots = input_tensor.size(0)
    if n != num_slots:
        mask = torch.cat([torch.ones(n), torch.zeros(num_slots - n)])
        input_tensor *= mask

def softmax(input_tensor, n): # not designed for -inf 
    # print("INSIDE SoftMax")
    scale_for_sm = 1e-2
    input_tensor *= scale_for_sm
    a_max = quick_operation(input_tensor, n, "MAX")
    input_tensor /= scale_for_sm
    a_max /= scale_for_sm
    a_exp = exponentiate(input_tensor - a_max)
    zero_out_tail(a_exp, n)
    sum_exp = quick_operation(a_exp, n, "SUM")
    sum_exp *= 2.0 / n
    inverted = invert(sum_exp, 5)
    inverted *= 2.0 / n

    return a_exp * inverted

def softmax_approx(input, dim):
    slices = torch.unbind(input, dim=dim)
    print(slices[0].ravel().shape)
    processed_slices = [softmax(s.ravel(), s.ravel().shape[0]).reshape(slices[0].shape) for s in slices]
    output_tensor = torch.stack(processed_slices, dim=dim)
    # print(output_tensor.shape)
    return output_tensor

    
class SoftmaxApprox(nn.Module):
    """
    Applies Softmax approximation that is suitable for FHE applications.
    """

    def forward(self, input: Tensor, dim: int) -> Tensor:
        scale_for_sm = 1e-2
        input *= scale_for_sm
        a_max = quick_operation(input, dim, "MAX")
        input /= scale_for_sm
        a_max /= scale_for_sm
        a_exp = exponentiate(input - a_max)
        zero_out_tail(a_exp, dim)
        sum_exp = quick_operation(a_exp, dim, "SUM")
        sum_exp *= 2.0 / dim
        inverted = invert(sum_exp, 5)
        inverted *= 2.0 / dim

        return a_exp * inverted

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention_approx(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    # print(attn_weight.shape)
    attn_weight = softmax_approx(attn_weight, dim=-1)
    return
    print(attn_weight.shape)
    attn_weight = F.dropout(attn_weight, dropout_p)
    return attn_weight @ value

def example():
    x = torch.tensor([0.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, -8.0]+[0 for _ in range(32-11)])
    print(x.shape)
    query = torch.rand(32, 8, 128, 64, dtype=torch.float16)
    key = torch.rand(32, 8, 128, 64, dtype=torch.float16)
    value = torch.rand(32, 8, 128, 64, dtype=torch.float16)
    result = scaled_dot_product_attention_approx(query, key, value)
    # print("Approximation:\n",result)
    # print("Expected:\n",F.scaled_dot_product_attention(query, key, value))    
    # assert torch.allclose(result, F.scaled_dot_product_attention(query, key, value), atol=1e-3)

example()