import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def check(a, b):  ##a: pred , b: labels
    a = torch.argmax(a, dim=1)
    b = b.to(device=a.device)
    mask = a == b
    return mask


##DKD loss
class DKDLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=8.0, temperature=4, warmup=20):  ##warmup默认是20
        super(DKDLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.warmup = warmup

    def forward(self, logits_student, logits_teacher, target, epoch):
        target = target.to(device=logits_student.device)
        # print(target.device, logits_student.device)

        def _get_gt_mask(logits, target):
            target = target.reshape(-1)
            mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
            return mask

        def _get_other_mask(logits, target):
            target = target.reshape(-1)
            mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
            return mask

        def cat_mask(t, mask1, mask2):
            t1 = (t * mask1).sum(dim=1, keepdims=True)
            t2 = (t * mask2).sum(dim=1, keepdims=True)
            rt = torch.cat([t1, t2], dim=1)
            return rt

        gt_mask = _get_gt_mask(logits_student, target)
        # print('gt_mask:', gt_mask, gt_mask.shape)             ##torch.Size([4, 100])
        other_mask = _get_other_mask(logits_student, target)
        # print('other_mask:', other_mask, other_mask.shape)    ##torch.Size([4, 100])
        pred_student = F.softmax(logits_student / self.temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=1)
        pred_student = cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
                F.kl_div(log_pred_student, pred_teacher, reduction='batchmean')    ###原代码的mean会报错，改成batchmean就不会
                * (self.temperature ** 2)
                / target.shape[0]
        )
        pred_teacher_part2 = F.softmax(
            logits_teacher / self.temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / self.temperature - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
                F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='batchmean')     ###原代码的mean会报错，改成batchmean就不会
                * (self.temperature ** 2)
                / target.shape[0]
        )

        # 根据 epoch 进行 warmup
        loss_dkd = min(epoch / self.warmup, 1.0) * (self.alpha * tckd_loss + self.beta * nckd_loss)

        return loss_dkd


##################自适应调整难易任务的学习权重###########
def lr(epoch, epochs):
    e = epoch
    E = epochs
    lr = e / E
    lr_values = math.cos((math.pi/2) * lr)**2  # lr_values 0-------->1  #
    # lr_values = lr  #0----->1
    return lr_values

def lr_line(epoch, epochs):
    e = epoch
    E = epochs

    lr = min(epoch / 20, 1.0)  # lr 0-------->1  #
    return lr


###############################错误知识校准模块  交换 #################################
#~mask 返回的是相反的
def calibrate(mask, y_tea, y_stu, labels, T):  ##mask: , y_tea: 教师的预测,   y_stu: 学生的预测
    T = T
    de_mask = ~mask   ##得到相反的mask
    y_tea = y_tea[de_mask]
    y_stu = y_stu[de_mask]
    label = labels[de_mask]  ## tensor([41, 80])

    tea_idx = torch.argmax(y_tea, dim=1)  ###tea  最大下标
    num = label.shape[0]
    for i in range(num):
        l_idx = label[i]
        y_idx = tea_idx[i]
        #######交换了一下，调整成为正确和label相符合的
        y_tea[i][y_idx], y_tea[i][l_idx] = y_tea[i][l_idx].clone(), y_tea[i][y_idx].clone()

    return y_tea, y_stu

###################################################################################
###############################错误知识校准模块2   加权  #################################
#~mask 返回的是相反的    就把错误的下标的类别对应标签强化一下
def calibrate2(mask, y_tea, y_stu, labels, T):  ##mask: , y_tea: 教师的预测,   y_stu: 学生的预测
    T = T
    de_mask = ~mask   ##得到相反的mask
    y_tea = y_tea[de_mask]
    y_stu = y_stu[de_mask]     ##  torch.Size([3, 10])
    label = labels[de_mask]  ## tensor([41, 80])
    num = label.shape[0]
    new_tensor = torch.zeros_like(y_tea)
    for i in range(num):
        index = label[i]
        new_tensor[i][index] = 1

    y_tea = (y_tea + new_tensor) / 2
    return y_tea, y_stu

###################################################################################
###############################错误知识校准模块3   只加权错误的两个类的概率值  ###################
#~mask 返回的是相反的    就把错误的下标的类别对应标签强化一下
def calibrate3(mask, y_tea, y_stu, labels, T):  ##mask: , y_tea: 教师的预测,   y_stu: 学生的预测
    T = T
    de_mask = ~mask   ##得到相反的mask
    # y_tea1 = y_tea[de_mask]
    y_tea = y_tea[de_mask]
    y_stu = y_stu[de_mask]     ##  torch.Size([3, 10])
    label = labels[de_mask]  ## tensor([41, 80])
    tea_idx = torch.argmax(y_tea, dim=1)
    num = label.shape[0]
    for i in range(num):
        l_idx = label[i]
        y_idx = tea_idx[i]
        sum1 = y_tea[i][l_idx] + y_tea[i][y_idx]
        ### new
        y_tea[i][l_idx] = (y_tea[i][l_idx].clone() + 1) / 2
        y_tea[i][y_idx] = (y_tea[i][y_idx].clone() + 0) / 2
        sum2 = y_tea[i][l_idx] + y_tea[i][y_idx]
        ####
        y_tea[i][l_idx] = y_tea[i][l_idx].clone()*(sum1 / sum2)
        y_tea[i][y_idx] = y_tea[i][y_idx].clone()*(sum1 / sum2)

    return y_tea, y_stu

