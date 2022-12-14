import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from evaluation import compute_rmse

import scipy.io as sio
import numpy as np
import os
import math
import time
import random



seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
model_my="model_3D"

#数据加载  P是类别 L是波段 col是裁剪的图像尺度
dataset = 'jasper'
if dataset == 'samson':
    image_file = r'./data/samson_dataset.mat'
    P, L, col = 3, 156, 95
    LR, EPOCH, batch_size = 2e-3, 480, 1
    beta, delta, gamma = 0.1, 1e-3, 1e-7
    sparse_decay, weight_decay_param = 5e-6, 1e-4
    index = [1,2,0]
    print("beta:",beta)
elif dataset == 'jasper':
    image_file = r'./data/jasper_dataset.mat'
    P, L, col = 4, 198, 100
    LR, EPOCH, batch_size = 8e-3, 70, 1
    beta, delta, gamma = 0.5, 1e-2, 1e-7
    sparse_decay, weight_decay_param = 0, 0
    index = [3,1,2,0]
    print("beta:",beta)
else:
    raise ValueError("错误")


#读取数据
data = sio.loadmat(image_file)
Y = torch.from_numpy(data['Y'])
A = torch.from_numpy(data['A'])
M_true = data['M']

#用VCA把端元初始化
E_VCA_init = torch.from_numpy(data['M1']).unsqueeze(2).unsqueeze(3).float()

Y=torch.reshape(Y,(L,col,col))
A=torch.reshape(A,(P,col,col))

#定义数据集
class MyTrainData(torch.utils.data.Dataset):
  def __init__(self, img, gt, transform=None):
    self.img = img.float()
    self.gt = gt.float()
    self.transform=transform

  def __getitem__(self, idx):
    return self.img,self.gt

  def __len__(self):
    return 1

def Nuclear_norm(inputs):
    _, band, h, w = inputs.shape
    input = torch.reshape(inputs, (band, h*w))
    out = torch.norm(input, p='nuc')
    return out

#定义KL损失函数
class SparseKLloss(nn.Module):
    def __init__(self):
        super(SparseKLloss, self).__init__()

    def __call__(self, input, decay=sparse_decay):
        input = torch.sum(input, 0, keepdim=True)
        loss = Nuclear_norm(input)
        return decay*loss

class NonZeroClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(1e-6,1)
            
            
#定义和为一函数
class SumToOneLoss(nn.Module):
    def __init__(self):
        super(SumToOneLoss, self).__init__()
        self.register_buffer('one', torch.tensor(1, dtype=torch.float))
        self.loss = nn.L1Loss(size_average=False)

    def get_target_tensor(self, input):
        target_tensor = self.one
        return target_tensor.expand_as(input)

    def __call__(self, input, gamma_reg=gamma):
        input = torch.sum(input, 1)
        target_tensor = self.get_target_tensor(input)
        loss = self.loss(input, target_tensor)
        return gamma_reg*loss


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(L, 128,kernel_size=(1,1), stride=1, padding=(0,0)),
            nn.BatchNorm2d(128,momentum=0.9),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Conv2d(128, 64,kernel_size=(1,1), stride=1, padding=(0,0)),
            nn.BatchNorm2d(64,momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(64, P, kernel_size=(1,1), stride=1, padding=(0,0)),
            nn.BatchNorm2d(P,momentum=0.9),
        )

        
        self.decoder1 = nn.Sequential(
            nn.Conv2d(P, L, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(P, L, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(P, L, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
        )
    def forward(self,x):
        abu_est1 = self.encoder(x).clamp_(0,1)
        re_result1 = self.decoder1(abu_est1)
        abu_est2 = self.encoder(re_result1).clamp_(0,1)
        re_result2 = self.decoder2(abu_est2)
        abu_est3 = self.encoder(re_result2).clamp_(0,1)
        re_result3 = self.decoder2(abu_est3)
        return abu_est1, re_result1, abu_est2, re_result2, abu_est3, re_result3

def weights_init(m):
    nn.init.kaiming_normal_(net.encoder[0].weight.data)
    nn.init.kaiming_normal_(net.encoder[4].weight.data)
    nn.init.kaiming_normal_(net.encoder[7].weight.data)


train_dataset= MyTrainData(img=Y,gt=A, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
#自编码网络
net=AutoEncoder()
#网络权重初始化
net.apply(weights_init)
#和为一约束
criterionSumToOne = SumToOneLoss()
#KL散度约束
criterionSparse = SparseKLloss()

model_dict = net.state_dict()
model_dict['decoder1.0.weight'] = E_VCA_init
model_dict['decoder2.0.weight'] = E_VCA_init
net.load_state_dict(model_dict)

loss_func = nn.MSELoss(size_average=True,reduce=True,reduction='mean')
optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=weight_decay_param)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
apply_clamp_inst1 = NonZeroClipper()
time_start = time.time()
for epoch in range(EPOCH):
    for i, (x,y) in enumerate(train_loader):

        #网络训练
        net.train()
        #获取四个阶段的结果
        abu_est1, re_result1, abu_est2, re_result2,abu_est3, re_result3 = net(x)

        #丰度的和为一约束
        loss_sumtoone = criterionSumToOne(abu_est1) + criterionSumToOne(abu_est2)+ criterionSumToOne(abu_est3)
        #稀疏约束
        loss_sparse = criterionSparse(abu_est1) + criterionSparse(abu_est2)+ criterionSparse(abu_est3)
        
        #循环一致性约束
        loss_re = 0.3*loss_func(re_result1,x) + 0.3*loss_func(x,re_result2)+ 0.4*loss_func(x,re_result3)
        loss_abu = delta*loss_func(abu_est1,abu_est2)

        total_loss =loss_re+loss_abu+loss_sumtoone+loss_sparse

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=1)
        optimizer.step()

        net.decoder1.apply(apply_clamp_inst1)
        net.decoder2.apply(apply_clamp_inst1)

        # if epoch % 10 == 0:
        #     print('Epoch:', epoch, '| i:', i,'| train loss: %.4f' % total_loss.data.numpy(),'| abu loss: %.4f' % loss_abu.data.numpy(),'| sumtoone loss: %.4f' % loss_sumtoone.data.numpy(),'| re loss: %.4f' % loss_re.data.numpy())
        
    scheduler.step()
time_end = time.time()

net.eval()
abu_est1, re_result1, abu_est2, re_result2 , abu_est3, re_result3= net(x)
abu_est1 = abu_est1/(torch.sum(abu_est1, dim=1))
abu_est1 = torch.reshape(abu_est1.squeeze(0),(P,col,col)).detach().numpy()

A = A[index,:,:]
A = A.detach().numpy()
Y = Y.detach().numpy()
print('**********************************')
print('RMSE: {:.5f}'.format(compute_rmse(A, abu_est1)))

for i in range(P):
    plt.subplot(2, P, i+1)
    plt.imshow(abu_est1[i,:,:])

for i in range(P):
    plt.subplot(2, P, P+i+1)
    plt.imshow(A[i,:,:])
plt.show()

print('total computational cost:', time_end-time_start)
save_path = str(dataset) +"_"+"_"+model_my+ str(beta)+'_cycunet_result.mat'
sio.savemat(save_path,{'Y':Y,'abu_est':abu_est1, 'A':A, 'M':M_true[:,index]})
