# -*- coding: utf-8 -*-
"""

@author: jia_2
"""


import torch
import torch.nn as nn



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

def Loss_fuction(abu_est1, re_result1, abu_est2, re_result2):
    
    #丰度的和为一约束
    loss_sumtoone = criterionSumToOne(abu_est1) + criterionSumToOne(abu_est2)
    
    #稀疏约束
    loss_sparse = criterionSparse(abu_est1) + criterionSparse(abu_est2)     
    
    #循环一致性约束
    loss_re = beta*loss_func(re_result1,x) + (1-beta)*loss_func(x,re_result2)
    loss_abu = delta*loss_func(abu_est1,abu_est2)
    total_loss =loss_re+loss_abu+loss_sumtoone+loss_sparse
    
    return total_loss