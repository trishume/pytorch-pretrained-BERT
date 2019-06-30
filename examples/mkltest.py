# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from timeit import default_timer as timer


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.zeros(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = w
        self.bias = torch.randn(nf)

    def forward(self, x):
        x = torch.addmm(self.bias, x, self.weight)
        # x = x.matmul(self.weight)
        return x

class Attention(nn.Module):
    def __init__(self, nx, batch_size, n_ctx=30, head_size=64):
        super(Attention, self).__init__()
        self.nx = nx
        self.head_size = head_size
        self.n_heads = nx // head_size
        self.n_ctx = n_ctx
        self.past_k = torch.randn(batch_size, self.n_heads, head_size, n_ctx)
        self.past_v = torch.randn(batch_size, self.n_heads, n_ctx, head_size)

    def forward(self, q, k, v):
        batch_size = q.size(0)
        q = q.view(batch_size, self.n_heads, 1, self.head_size)
        k = k.view(batch_size, self.n_heads, self.head_size, 1)
        k = torch.cat((self.past_k, k), dim=-1)
        # print()
        # print("attn",q.size(), k.size())
        w = torch.matmul(q, k)

        v = v.view(batch_size, self.n_heads, 1, self.head_size)
        v = torch.cat((self.past_v, v), dim=-2)
        # print("attn2",w.size(), v.size())
        res = torch.matmul(w,v)
        return res.view(batch_size, self.nx)

class Layer(nn.Module):
    def __init__(self, nx, batch_size):
        super(Layer, self).__init__()
        self.c1 = Conv1D(nx*3, nx)
        self.attn = Attention(nx, batch_size)
        self.c2 = Conv1D(nx, nx)
        self.c3 = Conv1D(nx*4, nx)
        self.c4 = Conv1D(nx, nx*4)

    def forward(self, x):
        x = self.c1(x)

        q = x[:, 0:nx]
        k = x[:,nx:nx*2]
        v = x[:,nx*2:nx*3]
        x = self.attn(q,k,v)
        # x = q+k+v

        x = self.c2(q+k+v)
        x = self.c3(x)
        x = self.c4(x)
        return x

# nx = 1024
# num_layers = 24
nx = 768
num_layers = 12

batch_size = 1

# Use the nn package to define our model as a sequence of layers. nn.Sequential is a Module which contains other Modules, and applies them in sequence to produce its output.
model = torch.nn.Sequential(
    *[Layer(nx, batch_size) for i in range(num_layers)]
)

x = torch.randn(batch_size, nx)

# model = torch.jit.trace(model, x)

iters = 5
start = timer()
for i in range(iters):
    with torch.no_grad():
        res = model(x)
        print(res.size(), res[0,0])
end = timer()
print("avg: ", (end - start)/iters)
