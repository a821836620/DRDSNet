import numpy as np
import torch
from medpy.metric import hd95
from hausdorff import hausdorff_distance
import torch.nn.functional as F

class LossAvarage(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += (val*n)
        self.count += n
        self.avg = round(self.sum/self.count, 4)

class DiceAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = DiceAverage.get_dices(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_dices(logits, targets):
        logits = F.softmax(logits, dim=1)
        dices = []
        for class_index in range(targets.size()[1]):
            inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dices.append(dice.item())
        return np.asarray(dices)

class HDAverage(object):
    def __init__(self, class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0] * self.class_num, dtype='float64')
        self.avg = np.asarray([0] * self.class_num, dtype='float64')
        self.sum = np.asarray([0] * self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = HDAverage.get_HD(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)

    @staticmethod
    def get_HD(logits, targets):
        HDs = []
        for class_index in range(targets.size()[1]):
            HD = hd95(logits[:, class_index, :, :, :].cpu().detach().numpy(), targets[:, class_index, :, :, :].cpu().detach().numpy())
            HDs.append(HD)
        return np.asarray(HDs)


class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):  
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))  # matrix 2*2
        self.SMOOTH = 1e-5

    def _fast_hist(self, label_pred, label_true):  #  computer confusion matrix
        # find where label_true is not equal to 255
        mask = (label_true >= 0) & (label_true < self.num_classes)
        # np.bincount compute the histogram of the array 
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)  ## core code
        return hist

    def add_batch(self, predictions, gts):  # compute confusion matrix
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())  # flatten() multi-dimension to one-dimension
    def evaluate(self):  # simple evaluation
        np.seterr(divide='ignore', invalid='ignore')
        # 6.Accuracy
        Accuracy = (np.diag(self.hist).sum() + self.SMOOTH) / (self.hist.sum() + self.SMOOTH)  # PA = TP / (TP + FP + FN)
        # acc_cls = np.diag(self.hist) / self.hist.sum(axis=0)  # cpa precision

        # 4.ppv/cpa/Precision 
        # cpa = np.diag(self.hist) / (self.hist.sum(axis = 0) + SMOOTH) # cpa precision

        # 3.Sensitivity/Recall
        # Recall = np.diag(self.hist) / (self.hist.sum(axis = 1)+ SMOOTH) #sensivity

        # if return is nan, then replace with 0
        # print("acc_cls", acc_cls)
        # acc_cls = np.nanmean(acc_cls)# nanmean()

        # 1.DSC
        # dsc = 2*(np.diag(self.hist)[1]) / (2*(np.diag(self.hist)[1]) + np.diag(np.fliplr(self.hist)).sum())
        Positive = 0

        if self.hist.sum(axis=1)[1] != 0: 
            Positive = 1
        # TrueFalseNum
        WrongNum = 0
        if np.diag(self.hist).sum() == 0:
            WrongNum = 1


        IoU = (np.diag(self.hist) + self.SMOOTH) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(
            self.hist) + self.SMOOTH)  # IoU 

        # MIoU = sum(IoU)/self.num_classes
        mIoU = np.nanmean(IoU)

        # return acc, recallRate, cpa, Positive, WrongNum, mIoU
        return Accuracy, mIoU, Positive, WrongNum

def get_iou(target, predict, thr, num_classes):
    assert target.size() == predict.size()
    predict[predict < thr] = 0
    predict[predict >= thr] = 1
    target[target < thr] = 0
    target[target >= thr] = 1

    predict = predict.numpy().astype(np.int16)
    target = target.numpy().astype(np.int16)
    Iou = IOUMetric(num_classes)
    Iou.add_batch(predict, target)  # predict 

    # acc, recallRate, cpa, positive, wrongNum, miou= Iou.evaluate()
    Accuracy, mIoU, Positive, WrongNum = Iou.evaluate()
    return Accuracy, mIoU, Positive, WrongNum


# target [batchsize,channel,d,h,w]
def get_mIoU(target, predict, thr, num_classes): # 
    batchsize = target.shape[0]
    m_iou = 0
    PositiveNum = 0
    WrongNum = 0
    acc = 0
    for i in range(batchsize):
        Accuracy, mIoU, Positive, Wrongn = get_iou(target[i,:,:,:],predict[i,:,:,:],thr, num_classes)
        PositiveNum += Positive
        WrongNum += Wrongn
        m_iou += mIoU
        acc += Accuracy
    return m_iou / batchsize, acc / batchsize, PositiveNum, WrongNum

def m_metric(target, predict, thr, num_classes):
    batchsize = target.shape[0]

    mask = target.squeeze(1)
    predict = predict.squeeze(1)

    predict[predict < thr] = 0
    predict[predict >= thr] = 1
    mask[mask < thr] = 0
    mask[mask >= thr] = 1

    predict = predict.numpy().astype(np.int16)
    mask = mask.numpy().astype(np.int16)

    m_dsc = []
    m_acc = []
    m_ppv = []
    m_sen = []
    m_hausdorff_distance = []

    for i in range(batchsize):
        m_dsc.append(dice_coef(predict[i], mask[i]))
        m_acc.append(accuracy(predict[i], mask[i]))
        m_ppv.append(ppv(predict[i], mask[i]))
        m_sen.append(sensitivity(predict[i], mask[i]))
        if len(predict[i].shape) == 3:
            for j in range(predict.shape[1]):
                
                m_hausdorff_distance.append(hausdorff_distance(predict[i][j], mask[i][j]))
        else:
            m_hausdorff_distance.append(hausdorff_distance(predict[i], mask[i]))
    return np.nanmean(m_dsc), np.nanmean(m_acc), np.nanmean(m_ppv), np.nanmean(m_sen), np.nanmean(m_hausdorff_distance)

def dice_coef(output, target, SMOOTH=1e-5):
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    #output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    #target = target.view(-1).data.cpu().numpy()

    intersection = (output * target).sum()

    return (2. * intersection + SMOOTH) / \
        (output.sum() + target.sum() + SMOOTH)

def accuracy(output, target):
    return (output == target).sum() / len(output.flatten())


def ppv(output, target, SMOOTH=1e-5):
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    intersection = (output * target).sum()
    return  (intersection + SMOOTH) / \
           (output.sum() + SMOOTH)

def sensitivity(output, target, SMOOTH=1e-5):
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (intersection + SMOOTH) / \
        (target.sum() + SMOOTH)