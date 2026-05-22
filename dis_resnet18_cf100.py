import torch
from dataload_cf100 import train_loader,val_loader,train_dataset,val_dataset
from dis_model import h_resnet50, h_resnet18
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.nn import functional as F
from tool import *
from my_log import save_log
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
tips = '_dis_resnet18_cf100'
save_log(tips)

####################################去错+校正#############################################

if __name__ == '__main__':
    Teacher = h_resnet50(num_classes=100)
    device = torch.device('cuda:0')
    teacherNet = Teacher
    teacherNet.load_state_dict(torch.load("./pth/h_resnet50_cf100.pth"))
    teacherNet.eval()
    # teacherNet.train(mode=False)
    teacherNet = teacherNet.to(device)

    studentNet = h_resnet18(num_classes=100)
    # studentNet.load_state_dict(torch.load("./pth/h_resnet18_cf100.pth", map_location=device))
    studentNet = studentNet.to(device)
    optimizer = optim.Adam(studentNet.parameters(), lr=1e-4)
    ##scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    lossCE = nn.CrossEntropyLoss()
    lossMSE = nn.MSELoss(reduction='mean')
    lossKD = nn.KLDivLoss(reduction='batchmean')
    val_num = len(val_dataset)
    best_acc = 0.0
    save_path = './pth/dis_resnet18_cf100_2.pth'
    train_steps = len(train_loader)

    epochs = 300
    T, lambda_stu = 4.0, 0.5
    for epoch in range(epochs):
        ##train
        studentNet.train()
        running_loss = 0.0
        batch_count = 0
        n = 0
        train_loader = tqdm(train_loader)
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            optimizer.zero_grad()
            Stu = studentNet(images.to(device))

            y_stu = Stu
            loss_student = lossCE(y_stu, labels.to(device))
            Tea = teacherNet(images.to(device))
            # print('Tea.shape:', Tea.shape)
            y_tea = Tea
            mask = check(y_tea, labels)
            y_tea1,y_stu1 = y_tea[mask], y_stu[mask]
            loss_easy = lossKD(F.log_softmax(y_stu1 / T, dim=1),
                                  F.softmax(y_tea1 / T, dim=1))

            ###################################################################################
            ##############################   校准    ###########################################
            y_tea2, y_stu2 = calibrate3(mask, y_tea, y_stu, labels, T)
            loss_hard = lossKD(F.log_softmax(y_stu2 / T, dim=1), F.softmax(y_tea2 / T, dim=1))


            #####################################  模块3    #############################################
            b = lr(epoch, epochs)

            # loss = lambda_stu * loss_student + (1 - lambda_stu) * T * T *(b*loss_teacher + (1-b)*loss_hard)
            loss = lambda_stu * loss_student + (1 - lambda_stu) * T * T *(b*loss_easy + (1-b)*loss_hard)

            ######################################################################################
            loss.backward()
            optimizer.step()
            ##scheduler.step()

            running_loss += loss.item()
            train_loader.desc = "train epoch[{}/{}] loss:{:.4f}".format(epoch + 1, epochs, loss)

        ##validate
        studentNet.eval()
        acc = 0.0
        # top5_acc = 0.0  # accumulate Top-5 accurate number / epoch
        with torch.no_grad():
            val_loader = tqdm(val_loader)
            for val_data in val_loader:
                val_images, val_labels = val_data
                outputs = studentNet(val_images.to(device))

                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                ########top5 acc######################
                # _, top5_pred = outputs.topk(5, 1)  # 获取前五个最大的预测索引
                # top5_pred = top5_pred.t()  # 转置以便与目标标签形状匹配
                #
                # correct = top5_pred.eq((val_labels.to(device)).view(1, -1).expand_as(top5_pred))
                # top5_acc += correct.sum(0).gt(0).sum().item()

                val_loader.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

            val_accurate = acc / val_num
            # top5_accuracy = top5_acc / val_num
            print('[epoch %d] train_loss: %.4f val_accuracy: %.4f ' %
                  (epoch + 1, running_loss / train_steps, val_accurate))
            logger.info('[epoch %d] train_loss: %.4f val_accuracy: %.4f ' %
                        (epoch + 1, running_loss / train_steps, val_accurate))
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(studentNet.state_dict(), save_path)
        print('dis_resnet18_cf100_2 ,Finished Training    betacc = %.4f   ' % (best_acc))
        logger.info('dis_resnet18_cf100_2 ,Finished Training    betacc = %.4f  ' % (best_acc))

