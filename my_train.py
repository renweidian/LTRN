# -*- coding:utf-8 -*-
import argparse
import math
from math import pi

from torch import nn
from tqdm import tqdm

import metrics
from lyy_dataset import *
from model import *
import time
import os

import pandas as pd

from model3 import HSI_MSI_Data1
from LTRN import UDoubleTransformerCP3
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
#parser.add_argument('--method', type=str, default='mst_plus_plus')
parser.add_argument("--dataset", type=str, default='cave_yu', help='train dataset name')
# train/val/test 数据集路径
parser.add_argument("--train_data_path", type=str, default='/data2/pbm/ssr_code/NTIRE2020/2020_cut32/train', help='train data path')
parser.add_argument("--val_data_path", type=str, default='/data2/pbm/ssr_code/NTIRE2020/2020_cut32/val', help='val data path')
parser.add_argument("--test_data_path", type=str, default='/data2/pbm/ssr_code/harvard_cut/test2', help='test data path')

parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--epoch", type=int, default=1000, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
parser.add_argument("--out_path", type=str, default='save_model', help='path log files')
parser.add_argument("--patch_size", type=int, default=64, help="patch size")
parser.add_argument("--stride", type=int, default=32, help="stride")
parser.add_argument("--gpu_id", type=str, default='0', help='path log files')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
save_path = os.path.join(opt.out_path, opt.dataset)
if not os.path.exists(save_path):
    os.makedirs(save_path)

save_path1 = os.path.join(save_path, 'rmse')
if not os.path.exists(save_path1):
    os.makedirs(save_path1)

save_path2 = os.path.join(save_path, 'sam')
if not os.path.exists(save_path2):
    os.makedirs(save_path2)

save_path3 = os.path.join(save_path, 'loss')
if not os.path.exists(save_path3):
    os.makedirs(save_path3)

save_path4 = os.path.join(save_path, 'psnr')
if not os.path.exists(save_path4):
    os.makedirs(save_path4)


print("文件夹创建完毕")

# 记录结果的的csv文件
df = pd.DataFrame(columns=['epoch', 'lr', 'train_loss', 'val_loss', 'val_rmse', 'val_psnr', 'val_sam'])  # 列名
df.to_csv(os.path.join(save_path, 'val_result_record.csv'), index=False)
df = pd.DataFrame(columns=['epoch', 'lr', 'train_loss', 'test_loss', 'test_rmse', 'test_psnr', 'test_sam'])  # 列名
df.to_csv(os.path.join(save_path, 'test_result_record.csv'), index=False)


def create_F2():
    F = [[0.02353544, 0.02918394, 0.03389103, 0.04048095, 0.0433052, 0.04707087,
          0.04940559, 0.04850183, 0.04518804, 0.03859812, 0.03155631, 0.02402497,
          0.01788693, 0.01223843, 0.00970037, 0.00658992, 0.00470709, 0.00282425,
          0.00235354, 0.00188283, 0.00169455, 0.00141589, 0.00094142, 0.00141213,
          0.00188283, 0.00282425, 0.00470709, 0.0056485, 0.00625101, 0.00625101,
          0.00625101],
         [0.00424014, 0.00376567, 0.00329873, 0.00329496, 0.00329496, 0.00376567,
          0.0056485, 0.01739739, 0.03200819, 0.04993278, 0.0593093, 0.06446827,
          0.06684064, 0.06919795, 0.06919795, 0.06823394, 0.06636993, 0.06356451,
          0.05790094, 0.05271938, 0.04189684, 0.03106678, 0.02447685, 0.01882835,
          0.01506268, 0.01317984, 0.01317984, 0.01430955, 0.01694551, 0.0216526,
          0.02495133],
         [0.00470709, 0.00424014, 0.00329873, 0.00235354, 0.00188283, 0.00164936,
          0.00143095, 0.00143095, 0.00164936, 0.00188283, 0.00235354, 0.00282425,
          0.00376567, 0.00470709, 0.00424014, 0.00329873, 0.00329873, 0.01084513,
          0.03483245, 0.06166661, 0.07390504, 0.0753134, 0.07410838, 0.07388244,
          0.07343056, 0.07248915, 0.07108078, 0.06853519, 0.06307497, 0.0593093,
          0.0593093]]
    return F

R = create_F2()
LR = 1e-4  # 不一定会用
init_lr2 = 2e-4  # 不一定会用
init_lr1 = init_lr2 / 10  # 不一定会用   # 老师原来用的那个学习率策略

rmse_optimal = 10
sam_optimal = 15
val_loss_optimal = 1
psnr_optimal=35
decay_power = 1.5

num =20  #cave 20    harvard 30
#cave
max_iteration = math.ceil(((512 - opt.patch_size) // opt.stride + 1) ** 2 * num / opt.batch_size) * opt.epoch

#icvl
#max_iteration=math.ceil(((1300-opt.patch_size)//opt.stride+1)*((1392-opt.patch_size)//opt.stride+1)*num/opt.batch_size)*opt.epoch

#2020
#max_iteration=math.ceil(((512-opt.patch_size)//opt.stride+1)*((482-opt.patch_size)//opt.stride+1)*num/opt.batch_size)*opt.epoch


#harvard
#max_iteration=math.ceil(((1040-opt.patch_size)//opt.stride+1)*((1392-opt.patch_size)//opt.stride+1)*num/opt.batch_size)*opt.epoch
# maxiteration=math.ceil(((1040-training_size)//stride+1)*((1392-training_size)//stride+1)*num/BATCH_SIZE)*EPOCH
warm_iter = math.floor(max_iteration / 40)
print(max_iteration)


path1= r"F:\STC\STC\LYY\cave\cave_train/"
path2=r'F:\STC\STC\LYY\cave\cave_val/'
imglist1=os.listdir(path1)
imglist2=os.listdir(path2)
train_data=HSI_MSI_Data1(path1,R,64,32,num=20)
train_loader = data.DataLoader(dataset=train_data, batch_size=32, shuffle=True)
print("数据切割完毕")


# train_data = MYDataset(opt.train_data_path)
# train_loader = data.DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True,num_workers=2)
# print(len(train_data))

# val_data = MYDataset(opt.val_data_path)
# val_loader = data.DataLoader(dataset=val_data, batch_size=1, shuffle=False)
# print(len(val_data))

#test_data = MYDataset(opt.test_data_path)
#test_loader = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
#print(len(val_data))

R_inv = np.linalg.pinv(R)
#    R_inv=torch.from_numpy(R_inv)
R_inv = torch.Tensor(R_inv)  # rgb超分不需要这个
R2 = torch.Tensor(R)
R2 = R2.cuda()

cnn = UDoubleTransformerCP3(3, 64, 64, 32, 31, 4, 1, "RTGB3",3).cuda()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iteration, eta_min=1e-6)


loss_func = nn.L1Loss(reduction='mean').cuda()
# loss_func = nn.MSELoss(reduction='mean').cuda()
#    loss_func=LossTrainCSS()
loss1 = MyarcLoss().cuda()  # 计算sam要用

# 模型参数初始化
for m in cnn.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def validate():
    cnn.eval()
    RMSE = AverageMeter()
    SAM = AverageMeter()
    PSNR = AverageMeter()
    val_loss = AverageMeter()
    with torch.no_grad():
        for i in range(0,len(imglist2)):
        # for i,(msi, hsi) in enumerate(data_loader):
            img = loadmat(path2 + imglist2[i])
            img1 = img["b"]
            img1=img1/img1.max()
            hsi = np.transpose(img1, (2, 0, 1))
            MSI = np.tensordot(R, hsi, axes=([1], [0]))
            msi = torch.Tensor(np.expand_dims(MSI, axis=0))
            hsi=torch.Tensor(np.expand_dims(hsi,0))
            prediction = reconstruction(cnn, R2, msi.cuda(), opt.patch_size, 32)
            loss = loss_func(prediction.unsqueeze(0), hsi.cuda())
            val_loss.update(loss.data)
            prediction = torch.clamp(prediction, 0, 1)
            # print("************************************************************************************")
            Fuse = prediction.cpu().detach().numpy()  # 转移至cpu
            sam = loss1(prediction, hsi.cuda().squeeze())
            sam = sam * 180 / pi
            SAM.update(sam.data)

            Fuse = np.squeeze(Fuse)
            Fuse = np.clip(Fuse, 0, 1)
            a, b = metrics.rmse1(Fuse, np.squeeze(hsi.cpu().detach().numpy()))
            RMSE.update(a)
            PSNR.update(b)
    return val_loss.avg, RMSE.avg, PSNR.avg, SAM.avg


step = 0
for epoch in range(opt.epoch):
    time1 = time.time()
    cnn.train()
    tbar = tqdm(train_loader, ncols=100)
    train_loss = AverageMeter()
    for epoch_step, (a1, a2) in enumerate(tbar):
        # 学习率更新设置

        lr = warm_lr_scheduler(optimizer, init_lr1, init_lr2, warm_iter, step,
                               lr_decay_iter=1, max_iter=max_iteration, power=decay_power)
        step = step + 1
        output = cnn(a2.cuda())
        loss = loss_func(output, a1.cuda())
        # assert torch.isnan(loss).sum() == 0, print("...........")
        train_loss.update(loss.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        tbar.set_description('epoch:{}  lr:{}  loss:{}'.format(epoch + 1, lr, train_loss.avg))
        # 验证
    if epoch+1 >=50:
        val_loss, val_rmse, val_psnr, val_sam = validate()
        # 根据验证集筛选模型
        if val_loss < val_loss_optimal or torch.abs(val_loss - val_loss_optimal)<0.1 :
            if val_loss < val_loss_optimal:
                val_loss_optimal = val_loss
            save_checkpoint(save_path3, epoch+1, step, cnn, optimizer)
        if val_sam < sam_optimal or torch.abs(val_sam - sam_optimal)<0.05:
            if val_sam < sam_optimal:
                sam_optimal = val_sam
            save_checkpoint(save_path2, epoch+1, step, cnn, optimizer)
        if val_rmse < rmse_optimal or np.abs(val_rmse - rmse_optimal)<0.05:
            if val_rmse < rmse_optimal:
                rmse_optimal = val_rmse
            save_checkpoint(save_path1, epoch+1, step, cnn, optimizer)
            #torch.save(cnn, save_path + '/' + str(epoch + 1) + 'RMSE_best.pkl')
        if val_psnr > psnr_optimal or np.abs(val_psnr - psnr_optimal)<0.05:
            if val_psnr > psnr_optimal:
                psnr_optimal = val_psnr
            save_checkpoint(save_path4, epoch+1, step, cnn, optimizer)
 
        # 测试
        #test_loss, test_rmse, test_psnr, test_sam = validate(test_loader)
        print("epoch:{:3d} ,  lr:{:.8f},        trainloss:{:.8f},".format(epoch + 1, lr, train_loss.avg))
        print("               val_loss:{:.8f},  val_rmse:{:.8f},   val_psnr:{:.8f},   val_sam:{:.8f}"
              .format(val_loss, val_rmse, val_psnr, val_sam))
        #print("               test_loss:{:.8f}, test_rmse:{:.8f},  test_psnr:{:.8f},  test_sam:{:.8f}"
        #      .format(test_loss, test_rmse, test_psnr, test_sam))

        val_list = [epoch + 1, lr, train_loss.avg, val_loss, val_rmse, val_psnr, val_sam]
        #test_list = [epoch + 1, lr, train_loss.avg, test_loss, test_rmse, test_psnr, test_sam]

        # 由于DataFrame是Pandas库中的一种数据结构，它类似excel，是一种二维表，所以需要将list以二维列表的形式转化为DataFrame
        val_data = pd.DataFrame([val_list])
        val_data.to_csv(os.path.join(save_path, 'val_result_record.csv'),
                        mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了

