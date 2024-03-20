import numpy as np
import torch
import os

def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))

def create_F():
    F = np.array(
        [[2.0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
         [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 11, 16, 19, 21, 20, 18, 16, 14, 11, 7, 5, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2],
         [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16, 9, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    for band in range(3):
        div = np.sum(F[band][:])
        for i in range(31):
            F[band][i] = F[band][i] / div;
    return F


def reconstruction(net2, R, MSI, training_size, stride):
    index_matrix = torch.zeros((R.shape[1], MSI.shape[2], MSI.shape[3])).cuda()
    abundance_t = torch.zeros((R.shape[1], MSI.shape[2], MSI.shape[3])).cuda()
    a = []
    for j in range(0, MSI.shape[2] - training_size + 1, stride):
        a.append(j)
    a.append(MSI.shape[2] - training_size)
    b = []
    for j in range(0, MSI.shape[3] - training_size + 1, stride):
        b.append(j)
    b.append(MSI.shape[3] - training_size)
    for j in a:
        for k in b:
            temp_hrms = MSI[:, :, j:j + training_size, k:k + training_size]
            #                temp_hrms=torch.unsqueeze(temp_hrms, 0)
            #                print(temp_hrms.shape)
            with torch.no_grad():
                # print(temp_hrms.shape)
                #                    HSI = net2(R,R_inv,temp_hrms)
                HSI = net2(temp_hrms)
                HSI = HSI.squeeze()
                #                   print(HSI.shape)
                HSI = torch.clamp(HSI, 0, 1)
                abundance_t[:, j:j + training_size, k:k + training_size] = abundance_t[:, j:j + training_size,
                                                                           k:k + training_size] + HSI
                index_matrix[:, j:j + training_size, k:k + training_size] = 1 + index_matrix[:, j:j + training_size,
                                                                                k:k + training_size]

    HSI_recon = abundance_t / index_matrix
    return HSI_recon


class MyarcLoss(torch.nn.Module):
    def __init__(self):
        super(MyarcLoss, self).__init__()

    def forward(self, output, target):
        sum1 = output * target
        sum2 = torch.sum(sum1, dim=0) + 1e-10
        norm_abs1 = torch.sqrt(torch.sum(output * output, dim=0)) + 1e-10
        norm_abs2 = torch.sqrt(torch.sum(target * target, dim=0)) + 1e-10
        aa = sum2 / norm_abs1 / norm_abs2
        aa[aa < -1] = -1
        aa[aa > 1] = 1
        spectralmap = torch.acos(aa)
        return torch.mean(spectralmap)


def warm_lr_scheduler(optimizer, init_lr1, init_lr2, warm_iter, iteraion, lr_decay_iter, max_iter, power):
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer
    if iteraion < warm_iter:
        lr = init_lr1 + iteraion / warm_iter * (init_lr2 - init_lr1)
    else:
        lr = init_lr2 * (1 - (iteraion - warm_iter) / (max_iter - warm_iter)) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
