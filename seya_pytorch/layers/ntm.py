import time

import numpy as np

import torch
from torch.autograd import Variable
from torch import Tensor

from torch.nn import Parameter, Module, LSTMCell, Linear
import torch.nn.functional as F

class NTM(Module):
    r"""Applies a multi-layer controller Neural Turing Machine (NTM) to an input sequence.
    """

    def __init__(self, input_dim, hidden_size, m_length, n_slots, shift_range = 3, **kwargs):

        assert shift_range % 2 == 1, 'Shift range must be odd'

        super(NTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.m_length = m_length
        self.n_slots = n_slots
        self.shift_range = shift_range

        # Initialise the controller
        self.controller = LSTMCell(self.input_dim + self.m_length, self.hidden_size)

        # Initialise the heads

        # For content based addressing and writing
        # Note: pytorch's linear actually acts as a transpose product. It performs x * w^T
        self.head_rk = Linear(self.hidden_size, self.m_length)
        self.head_wk = Linear(self.hidden_size, self.m_length)

        # For beta, g, and gamma
        self.head_rc = Linear(self.hidden_size, 3)
        self.head_wc = Linear(self.hidden_size, 3)
        # For index based addressing (shift)
        self.head_rs = Linear(self.hidden_size, self.shift_range)
        self.head_ws = Linear(self.hidden_size, self.shift_range)

        # For the add and eras vectors
        # We bundle them together for better parallelism
        self.head_w = Linear(self.hidden_size, 2 * self.m_length)

        self.reset_weights()

    def reset_weights(self):
        # Use Xavier/Glorot uniform initialisation for all feed forward weights
        bound = np.sqrt(6.0 / (self.input_dim + self.m_length + self.hidden_size))
        bound_k = np.sqrt(6.0 / (self.hidden_size + self.m_length))
        bound_c = np.sqrt(6.0 / (self.hidden_size + 3))
        bound_s = np.sqrt(6.0 / (self.hidden_size + self.shift_range))
        bound_w = np.sqrt(6.0 / (self.hidden_size + self.m_length))

        self.controller.weight_ih.data.uniform_(bound, -bound)
        self.head_rk.weight.data.uniform_(bound_k, -bound_k)
        self.head_wk.weight.data.uniform_(bound_k, -bound_k)
        self.head_rc.weight.data.uniform_(bound_c, -bound_c)
        self.head_wc.weight.data.uniform_(bound_c, -bound_c)
        self.head_rs.weight.data.uniform_(bound_s, -bound_s)
        self.head_ws.weight.data.uniform_(bound_s, -bound_s)
        self.head_w.weight.data.uniform_(bound_w, -bound_w)

        # Use zero initialisation for all biases apart from the forget gate and k head
        self.controller.bias_ih.data.zero_()
        self.controller.bias_hh.data.zero_()
        self.head_rc.bias.data.zero_()
        self.head_wc.bias.data.zero_()
        self.head_rs.bias.data.zero_()
        self.head_ws.bias.data.zero_()
        self.head_w.bias.data.zero_()

        # We'll use Xavier/Glorot uniform initialisation for them
        self.head_rk.bias.data.uniform_(bound_k, -bound_k)
        self.head_wk.bias.data.uniform_(bound_k, -bound_k)

        # Use ones initialisation for the forget gate
        # Note CuDNN/PyTorch uses 2 biases, even though it's equivalent to just using 1
        # Therefore we will only apply the ones initialisation to one of the forget biases
        self.controller.bias_ih.data[self.hidden_size:2 * self.hidden_size] = 1

        # Use orthogonal initialisation for the recurrent weight so that it has maximal eigenvalue 1
        # First we fill the weights with N(0, 1.0)
        # Then we take the SVD of the weights which will produce 3 matrices U, S and V
        # Therefore we set the weights to U which will be a random orthagonal matrix for the same shape
        self.controller.weight_hh.data.normal_(0, 1.0)
        u, s, v = self.controller.weight_hh.data.svd()
        self.controller.weight_hh.data = u

    @staticmethod
    def get_controller(inp, func_k, func_c, func_s):
        k = F.tanh(func_k(inp))
        c = func_c(inp)
        s = F.softmax(func_s(inp))

        # We want to keep the singleton dimensions so we use slices
        beta = F.relu(c[:, 0:1]) + 1e-4
        g = F.sigmoid(c[:, 1:2])
        gamma = F.relu(c[:, 2:3]) + (1 + 1e-4)

        return (k, c, s, beta, g, gamma)

    def get_address(self, M, head_tm1, k, c, s, beta, g, gamma):
        # First do content based addressing
        # We need to get the cosine distance of each slot from the embedding vector k
        # We can do this using the dot product rule via torch.addbmm
        # Therefore if we bmm (samples, 1, m) and (samples, m, n) this will give us (samples, 1, n)
        # We can then squeeze this to (samples, n)
        dot = torch.bmm(k[:, None], M.permute(0, 2, 1)).squeeze()

        # Get the norms of the slots and the embedding vectors
        # Note: by default, pytorch keeps singleton dimensions after sum
        # Therefore nM and nk are actually (samples, n, 1) and (samples, 1) respectively
        # Also note: pytorch does not do broadcasting, so we have to manually expand dimensions
        nM = (M**2).sum(2).squeeze()
        nk = (k**2).sum(1).expand_as(dot)

        # Finally, get the cosine distance, and turn it into a soft address via softmax
        # We also sharpen via beta
        content = F.softmax(beta.expand_as(dot) * dot / (nM * nk))

        # Apply the interpolation gate
        g = g.expand_as(content)
        inter = content * g + (1 - g) * head_tm1

        # Apply the shift to the content based address
        # Note: the original paper does this use circular convolution
        # Therefore we're gonna hack one out by padding 1D valid convolutions
        inter_expanded = torch.cat([inter[:, -(self.shift_range - 1):], inter], dim = 1)

        out = torch.autograd.Variable(torch.Tensor(*inter.size()))

        if inter_expanded.is_cuda:
          out = out.cuda()
        for i in range(s.size()[0]):
            out[i] = F.conv1d(inter_expanded[i:i + 1, None], s[i:i + 1, None]).squeeze()

        # Now we sharpen with gamma
        out = torch.pow(out, gamma.expand_as(out))
        out /= out.sum(1).expand_as(out)

        return out

    def forward(self, inp, states, heads, memory):
        # Note: Only works for 1 timestep at a time (i.e. must loop by hand)
        # Therefore inputs should be 2D with batch as the first dimension neurons as the second

        # Get the read head
        # First, get the controller output
        read_params = self.get_controller(states[0], self.head_rk, self.head_rc, self.head_rs)
        # Do addressing, content based and then shift. We lump it as a single function
        read_head = self.get_address(memory, heads[0], *read_params)

        # (samples, 1, n) * (samples, n, m)
        M_read = torch.bmm(read_head[:, None], memory).squeeze()

        # Controller is fed the input concatenated with the last read
        in_all = torch.cat([inp, M_read], 1).contiguous()
        out = self.controller(in_all, states)


        # Get the controller output again
        write_params = self.get_controller(out[0], self.head_wk, self.head_wc, self.head_ws)

        # Do addressing, content based and then shift. We lump it as a single function
        write_head = self.get_address(memory, heads[1], *write_params)


        # Get the write vectors (erase and then add)
        write_all = self.head_w(out[0])
        e = F.sigmoid(write_all[:, :self.m_length])
        a = write_all[:, self.m_length:]


        # Write to memory
        # Erase first, then add
        write_weight = write_head[:, :, None].expand_as(memory)
        M_out = memory * (1 - write_weight * e[:, None, :].expand_as(memory))
        M_out += write_weight * a[:, None, :].expand_as(memory)


        # We're done, return all the new results
        return [out[0], out[1]], [read_head, write_head], M_out
