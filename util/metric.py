import torch
import numpy as np

# https://blog.csdn.net/qq_21466543/article/details/82936246
class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)  # shape:(num_class, num_class)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self): #MPA
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    # 生成混淆矩阵
    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    # 输入预测和标签，生成混淆矩阵
    def add_batch(self, gt_image, pre_image):
        pre_image = pre_image.squeeze().long().cpu().numpy()
        gt_image = gt_image.squeeze().long().cpu().numpy()
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def export_tensor(self):
        return torch.tensor(self.confusion_matrix)
    
    def set_confusion_matrix(self, confusion_matrix):
        self.confusion_matrix = confusion_matrix.cpu().numpy()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)



if __name__ == '__main__':

    import torch
    eva = Evaluator(3) # 输入类别
    # 一维度
    # x = [2, 1, 0, 1, 2, 0]
    # y = [2, 0, 0, 1, 2, 1]

    # 输入的必须是整形的类别
    x = [[2, 1, 0],[1, 2, 0]]
    y = [[2, 1, 0], [1, 2, 1]]

    x = np.array(x)
    y = np.array(y)
    print(x)
    print(y)
    print('====================')
    # print(torch.min(y),torch.max(y))
    confusion_matrix = eva._generate_matrix(x,y)
    print(confusion_matrix) # 类别一定是class_num * class_num

    eva.add_batch(x, y)  # 评价标准使用
    PA = eva.Pixel_Accuracy()
    print("PA:",PA)

    Acc = eva.Pixel_Accuracy_Class()
    # print(Acc.shape)
    print("Acc:",Acc)

    MIoU = eva.Mean_Intersection_over_Union()
    # print(MIoU.shape)
    print("MIoU:",MIoU)

    print('====================')
    FWIoU = eva.Frequency_Weighted_Intersection_over_Union()
    # print(FWIoU.shape)
    print("FWIoU:",FWIoU)
