import os

import torch
from torch.autograd import Function

from tinyllama.BinaryTinyLlama.state_storage import Config


# %% Linear Function
class BinaryLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, kk, aa):
        ctx.save_for_backward(input, weight, bias, kk, aa)
        if bias is not None:
            return aa * torch.matmul(input, torch.tanh((weight - weight.mean()) * kk).t()) + bias
        else:
            return aa * torch.matmul(input, torch.tanh((weight - weight.mean()) * kk).t())

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, kk, aa = ctx.saved_tensors

        # kk and aa always doesn't need grad
        grad_input = grad_weight = grad_bias = grad_kk = grad_aa = None

        weight = weight.to(grad_output)
        input = input.to(grad_output)

        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, aa * torch.tanh((weight - weight.mean()) * kk))
        if ctx.needs_input_grad[1]:
            grad_weight = torch.einsum('ijk,ijl->kl', grad_output, input) * kk * aa * (
                        1 - torch.pow(torch.tanh((weight - weight.mean()) * kk), 2)) * (1 - 1/torch.tensor(weight.shape).sum())
        if ctx.needs_input_grad[2] and bias is not None:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, grad_kk, grad_aa


# %% kk learning schedule stage1
def kk_schedule_stage1(sloss_list, kk, kk_lr, aa, aa_lr, prefix, epoch_list, epoch, last_best_epoch,
                       model, optimizer, lr_scheduler):
    # pre-process of sloss_list in case of nan
    s = sloss_list[-1]
    sloss = torch.torch.empty_like(s).copy_(s)

    # calculate start of mean sloss value
    mean_start = -20
    mean_now_start = -15
    mean_sloss = torch.mean(torch.tensor(sloss_list[mean_start:]))
    mean_sloss_now = torch.mean(torch.tensor(sloss_list[mean_now_start:]))

    print(f'Current Validation Loss: {sloss:.4f}')
    print(f'Mean Validation Loss: {mean_sloss:.4f}')
    print(f'Current Mean Validation Loss: {mean_sloss_now:.4f}')

    # check if pushing kk and aa
    if mean_sloss_now > mean_sloss:
        '''dealing with over fitting'''
        sloss_list_temp = torch.tensor(sloss_list[-len(epoch_list):])
        epoch = epoch_list[torch.argmin(sloss_list_temp)]
        last_best_epoch = epoch

        # return back to last best checkpoint (in case of overfitting)
        # rewind model states
        file_path = "%s%.4d.pt" % (prefix, epoch)
        print(f"\nReturn to CheckPoint: {file_path} {'.' * 6}\n")
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        # delete extra files
        file_to_del = ["%s%.4d.pt" % (prefix, i) for i in epoch_list[torch.argmin(sloss_list_temp) + 1:]]
        for file in file_to_del:
            os.remove(file)

        # initialize some parameters
        epoch_list = []
        sloss_list = []

        # push kk and aa
        if kk < 1:
            kk += aa_lr
            aa = 1 / kk
        else:
            kk *= kk_lr
            aa = torch.tensor([1.]).to(kk.device)
        model.set_kk_stage1(torch.tensor([kk, aa]).to(kk.device))

        return kk, aa, sloss_list, mean_sloss, epoch_list, epoch, last_best_epoch

    else:
        return kk, aa, sloss_list, mean_sloss, epoch_list, epoch, last_best_epoch


# %% kk learning schedule stage2
def kk_schedule_stage2(sloss_list, ratio, prefix, epoch_list, epoch, last_best_epoch,
                       model, optimizer, lr_scheduler,
                       unbinary_ratio_threshold, push_step):
    # pre-process of sloss_list in case of nan
    s = sloss_list[-1]
    sloss = torch.torch.empty_like(s).copy_(s)

    # calculate start of mean sloss value
    mean_start = -20
    mean_now_start = -15
    mean_sloss = torch.mean(torch.tensor(sloss_list[mean_start:]))
    mean_sloss_now = torch.mean(torch.tensor(sloss_list[mean_now_start:]))

    print(f'Current Validation Loss: {sloss:.4f}')
    print(f'Mean Validation Loss: {mean_sloss:.4f}')
    print(f'Current Mean Validation Loss: {mean_sloss_now:.4f}')

    # check if pushing kk and aa
    if mean_sloss_now > mean_sloss:
        # update push step and ratio
        push_step += 1
        # ratio = 1 - 1 / (rate ** 2 * push_step + 1 / (1 - rate))

        # find which epoch to rewind
        sloss_list_temp = torch.tensor(sloss_list[-len(epoch_list):])
        epoch = epoch_list[torch.argmin(sloss_list_temp)]
        last_best_epoch = epoch

        # return back to last best checkpoint (in case of overfitting)
        # rewind model states
        file_path = "%s%.4d.pt" % (prefix, epoch)
        print(f"\nReturn to CheckPoint: {file_path} {'.' * 6}\n")
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        # delete extra files
        file_to_del = ["%s%.4d.pt" % (prefix, i) for i in epoch_list[torch.argmin(sloss_list_temp) + 1:]]
        for file in file_to_del:
            os.remove(file)

        # initialize some parameters
        epoch_list = []
        sloss_list = []

        # calculate unbinary ratio
        unbinary_ratio = []
        for name, param in model.named_parameters():
            if name.endswith('.weight') and name != 'src_embed.0.lut.weight' and name != 'tgt_embed.0.lut.weight':
                weight = param.flatten()
                k = model.state_dict()[name[:-7] + '.kk']
                a = model.state_dict()[name[:-7] + '.aa']
                weight_binary = a * torch.tanh(k * (weight - weight.mean())).abs()
                unbinary_ratio.append(weight_binary[weight_binary < 0.99].shape[0] / weight.shape[0])
        max_unbinary_ratio = torch.max(torch.tensor(unbinary_ratio))
        print(f'Remaining Unbinary Weight: {max_unbinary_ratio * 100:.2f} % ')
        unbinary_ratio_threshold = max_unbinary_ratio

        # push kk and aa
        Config.kk_list = []
        model.set_kk_stage2(ratio)

        return sloss_list, epoch_list, epoch, last_best_epoch, unbinary_ratio_threshold, max_unbinary_ratio, push_step

    else:
        # calculate unbinary ratio
        unbinary_ratio = []
        for name, param in model.named_parameters():
            if name.endswith('.weight') and name != 'src_embed.0.lut.weight' and name != 'tgt_embed.0.lut.weight':
                weight = param.flatten()
                k = model.state_dict()[name[:-7] + '.kk']
                a = model.state_dict()[name[:-7] + '.aa']
                weight_binary = a * torch.tanh(k * (weight - weight.mean())).abs()
                unbinary_ratio.append(weight_binary[weight_binary < 0.99].shape[0] / weight.shape[0])
        max_unbinary_ratio = torch.max(torch.tensor(unbinary_ratio))
        print(f'Remaining Unbinary Weight: {max_unbinary_ratio * 100:.2f} % ')

        return sloss_list, epoch_list, epoch, last_best_epoch, unbinary_ratio_threshold, max_unbinary_ratio, push_step
