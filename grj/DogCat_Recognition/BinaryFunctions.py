import torch
import torch.nn as nn

# 先定义函数，再定义层，有自定义的backward过程，可实现较复杂的激活函数
# class BinaryActivationReLU(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, kk):
#         # 在forward中，定义激活函数的计算过程
#         # 同时可以保存任何在后向传播中需要使用的变量值
#         ctx.save_for_backward(input, kk)  # 保存input和kk以便在backward中使用
#         output = torch.tanh(torch.relu(input) * kk)
#         return output
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         input, kk = ctx.saved_tensors
#         # 计算ReLU部分的激活
#         relu_grad = input > 0
#         # 计算tanh部分的导数
#         tanh_output = torch.tanh(relu_grad * input * kk)
#         tanh_grad = 1 - tanh_output.pow(2)
#         # 应用链式法则
#         grad_input = grad_output * tanh_grad * kk * relu_grad
#         return grad_input, None
#
# class BinaryReLU(nn.Module):
#     def __init__(self):
#         super(BinaryReLU, self).__init__()
#         self.kk = torch.tensor([999], requires_grad=False)
#
#     # allow external modification of kk
#     def set_kk(self, kknew):
#         with torch.no_grad():
#             self.kk = self.kk.to(kknew.device)
#             self.kk.copy_(kknew)
#
#         return BinaryActivationReLU.apply(input, self.kk)

#     def forward(self, input):
# 由于激活函数较为简单，即使是torch.autograd的内置激活函数的组合，依然可以自动backward
class BinaryReLU(nn.Module):
    def __init__(self):
        super(BinaryReLU, self).__init__()
        # 初始化kk
        self.kk = torch.tensor([999], requires_grad=False)

    # allow external modification of kk
    def set_kk(self, kknew):
        with torch.no_grad():
            self.kk = self.kk.to(kknew.device)
            self.kk.copy_(kknew)

    def forward(self, input):
        # 定义激活函数的逻辑，torch.autograd自动进行backward
        return torch.tanh(torch.relu(input) * self.kk)

class BinaryHeaviside(nn.Module):
    def __init__(self):
        super(BinaryHeaviside, self).__init__()

    def forward(self, x):
        return torch.heaviside(x,torch.Tensor([1]))