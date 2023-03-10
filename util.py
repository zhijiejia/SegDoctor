import os
import cv2
import torch 
import random
import numpy as np
from PIL import Image
import torch.distributed as dist
from collections import defaultdict
from matplotlib import pyplot as plt
from torch.nn import functional as F

from queue import Queue
from sklearn.cluster import KMeans

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)

def gen_pred_label(preds, patch_size=8):
    bs, h, w = preds.shape       # bs, h, w
    assert h % patch_size == 0, 'h should be divided by patch_size'
    assert w % patch_size == 0, 'h should be divided by patch_size'

    nrow = h // patch_size
    ncol = w // patch_size
    num_patch = nrow * ncol

    res = []
    for idx in range(bs):

        border = torch.zeros(size=(num_patch, 4)).long()        
        border.fill_(-1)
        
        pred = preds[idx].cpu().numpy()

        for i_idx in range(nrow):
            for j_idx in range(ncol):
                patch_idx = i_idx * ncol + j_idx
                y_slice = slice(i_idx * patch_size, (i_idx + 1) * patch_size)
                x_slice = slice(j_idx * patch_size, (j_idx + 1) * patch_size)
                patch_i_j = pred[y_slice, x_slice]
                
                vals = np.unique(patch_i_j)
                if len(vals) > 1:
                    border[patch_idx][:] = 1
        
        res.append(border)
    
    return torch.stack(res, dim=0)

def gen_link_gt(masks, patch_size=8):
    bs, h, w = masks.shape       # bs, h, w
    assert h % patch_size == 0, 'h should be divided by patch_size'
    assert w % patch_size == 0, 'h should be divided by patch_size'

    nrow = h // patch_size
    ncol = w // patch_size
    num_patch = nrow * ncol

    res = []
    for idx in range(bs):

        link_graph = torch.zeros(size=(num_patch, 4)).long()        
        link_graph.fill_(-1)
        
        mask = masks[idx].cpu().numpy()

        for i_idx in range(nrow):
            for j_idx in range(ncol):
                patch_idx = i_idx * ncol + j_idx
                y_slice = slice(i_idx * patch_size, (i_idx + 1) * patch_size)
                x_slice = slice(j_idx * patch_size, (j_idx + 1) * patch_size)
                patch_i_j = mask[y_slice, x_slice]
                
                if np.sum(patch_i_j) <= 0:    # 背景, 背景和其他都没有边
                    continue

                elif np.sum(patch_i_j) > 0:    # 前景, 前景和背景没有边, 但是和前景有边

                    vals, counts = np.unique(patch_i_j.flatten(), return_counts=True)
                    max_count_idx = np.argmax(counts)
                    cls_id = vals[max_count_idx]

                    if i_idx + 1 < nrow:   # down
                        y_down_slice = slice((i_idx + 1) * patch_size, (i_idx + 2) * patch_size)
                        x_down_slice = x_slice
                        patch_i_j_down = mask[y_down_slice, x_down_slice]
                        
                        vals, counts = np.unique(patch_i_j_down.flatten(), return_counts=True)    # vals: patch_i_j 中元素按照增序排列, counts和vals对应, 记录对应的val出现的次数
                        max_count_idx = np.argmax(counts)                                        # max_count_idx: patch_i_j 中出现次数最多的类别id
                        
                        link_graph[patch_idx][1] = 1.0 * (vals[max_count_idx] == cls_id)

                    if i_idx - 1 >= 0:   # up
                        y_up_slice = slice((i_idx - 1) * patch_size, i_idx * patch_size)
                        x_up_slice = x_slice
                        patch_i_j_up = mask[y_up_slice, x_up_slice]
                        
                        vals, counts = np.unique(patch_i_j_up.flatten(), return_counts=True)    # vals: patch_i_j 中元素按照增序排列, counts和vals对应, 记录对应的val出现的次数
                        max_count_idx = np.argmax(counts)                                        # max_count_idx: patch_i_j 中出现次数最多的类别id
                        
                        link_graph[patch_idx][0] = 1.0 * (vals[max_count_idx] == cls_id)

                    if j_idx - 1 >= 0:   # left
                        y_left_slice = y_slice
                        x_left_slice = slice((j_idx - 1) * patch_size, j_idx * patch_size)
                        patch_i_j_left = mask[y_left_slice, x_left_slice]
                        
                        vals, counts = np.unique(patch_i_j_left.flatten(), return_counts=True)    # vals: patch_i_j 中元素按照增序排列, counts和vals对应, 记录对应的val出现的次数
                        max_count_idx = np.argmax(counts)     
                        
                        link_graph[patch_idx][2] = 1.0 * (vals[max_count_idx] == cls_id)

                    if j_idx + 1 < ncol:   # right
                        y_right_slice = y_slice
                        x_right_slice = slice(j_idx * patch_size, (j_idx + 1) * patch_size)
                        patch_i_j_right = mask[y_right_slice, x_right_slice]
                        
                        vals, counts = np.unique(patch_i_j_right.flatten(), return_counts=True)    # vals: patch_i_j 中元素按照增序排列, counts和vals对应, 记录对应的val出现的次数
                        max_count_idx = np.argmax(counts)    
                        
                        link_graph[patch_idx][3] = 1.0 * (vals[max_count_idx] == cls_id)

        res.append(link_graph)
        
    res = torch.stack(res, dim=0)
    return res

def link_loss(gts, outs, feats, ps=8):
    # gts [bs, H, W]  , 注意有255
    # outs [bs, H, W]  
    # feats [bs, 2048, H, W]
    bs, c, h, w = feats.shape
    feats = feats.permute((0, 2, 3, 1)) # [bs, H, W, 2048]
    new_line_h = torch.zeros((bs, 1, w, c)).cuda()
    new_line_w = torch.zeros((bs, h, 1, c)).cuda()
    
    feats_up = torch.cat((new_line_h, feats), dim=1)[:, :h, :, :]    # [bs, 1 + H, W, 2048] -> [bs, H, W, 2048]
    feats_down = torch.cat((feats, new_line_h), dim=1)[:, 1:, :, :]  # [bs, H + 1, W, 2048] -> [bs, H, W, 2048]
    feats_left = torch.cat((new_line_w, feats), dim=2)[:, :, :w, :]  # [bs, H, 1 + W, 2048] -> [bs, H, W, 2048]
    feats_right = torch.cat((feats, new_line_w), dim=2)[:, :, 1:, :] # [bs, H, W + 1, 2048] -> [bs, H, W, 2048]
    
    feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-5)                    # 归一化: 将向量变成单位向量 
    feats_up = feats_up / (feats_up.norm(dim=-1, keepdim=True) + 1e-5)           # 归一化: 将向量变成单位向量 
    feats_down = feats_down / (feats_down.norm(dim=-1, keepdim=True) + 1e-5)     # 归一化: 将向量变成单位向量 
    feats_left = feats_left / (feats_left.norm(dim=-1, keepdim=True) + 1e-5)     # 归一化: 将向量变成单位向量 
    feats_right = feats_right / (feats_right.norm(dim=-1, keepdim=True) + 1e-5)  # 归一化: 将向量变成单位向量 
    
    simms = [
        torch.sum(feats * feats_up, dim=-1),       # [bs, H, W]
        torch.sum(feats * feats_down, dim=-1),
        torch.sum(feats * feats_left, dim=-1),
        torch.sum(feats * feats_right, dim=-1),
    ]
    
    pred = torch.stack(simms, dim=3).reshape((bs, -1, 4))   # [bs, H, W, 4]  -> [bs, H*W, 4]
    pred = torch.clamp((pred + 1) * 0.5, 0, 1)              # [bs, H*W, 4]
    crition = torch.nn.BCELoss(reduction="none")
    
    link_gt = gen_link_gt(gts, ps).cuda()     # [bs, H*W, 4]
    border = gen_pred_label(outs, ps).cuda()  # [bs, H*W, 4]
    link_gt[link_gt == 255] = -1
    
    border_num = torch.sum(1. * (border != -1))
    if border_num > 0:
        conditation = (link_gt != -1) & (border != -1)
    else:
        conditation = (link_gt != -1)
        
    pred = pred[conditation]                     
    link_gt = link_gt[conditation]                 

    loss = 1.0 * focal_loss(pred.float(), link_gt.float(), criterion=crition)
    
    if torch.isnan(loss):
        print(torch.sum(1. * (border != -1)))
    
    return loss

def gen_correct_map(gts, preds, patch_size=8):

    bs, h, w = preds.shape       # bs, h, w
    assert h % patch_size == 0, 'h should be divided by patch_size'
    assert w % patch_size == 0, 'h should be divided by patch_size'

    nrow = h // patch_size
    ncol = w // patch_size

    pred_ps = torch.zeros((bs, nrow, ncol)).cuda()
    gt_ps = torch.zeros((bs, nrow, ncol)).cuda()
    
    for idx in range(bs):
        
        pred = preds[idx]
        gt = gts[idx]
        for i_idx in range(nrow):
            for j_idx in range(ncol):
                y_slice = slice(i_idx * patch_size, (i_idx + 1) * patch_size)
                x_slice = slice(j_idx * patch_size, (j_idx + 1) * patch_size)
                pred_patch_i_j = pred[y_slice, x_slice]
                gt_patch_i_j = gt[y_slice, x_slice]
                
                pred_vals, pred_counts = torch.unique(pred_patch_i_j, return_counts=True)
                pred_index = torch.argmax(pred_counts)
                pred_ps[idx][i_idx][j_idx] = pred_vals[pred_index]
             
                gt_vals, gt_counts = torch.unique(gt_patch_i_j, return_counts=True)
                gt_index = torch.argmax(gt_counts)
                gt_ps[idx][i_idx][j_idx] = gt_vals[gt_index]
                
    return gt_ps, pred_ps

def next_border(preds, patch_size):
    bs, h, w = preds.shape       # bs, h, w
    assert h % patch_size == 0, 'h should be divided by patch_size'
    assert w % patch_size == 0, 'h should be divided by patch_size'

    nrow = h // patch_size
    ncol = w // patch_size

    res = []
    for idx in range(bs):

        border = torch.zeros(size=(nrow, ncol)).long()
        
        pred = preds[idx].cpu().numpy()

        for i_idx in range(nrow):
            for j_idx in range(ncol):
                y_slice = slice(i_idx * patch_size, (i_idx + 1) * patch_size)
                x_slice = slice(j_idx * patch_size, (j_idx + 1) * patch_size)
                patch_i_j = pred[y_slice, x_slice]
                
                vals = np.unique(patch_i_j)
                if len(vals) > 1:

                    border[i_idx][j_idx] = 1   # self
                    if i_idx - 1 >= 0:
                        border[i_idx - 1][j_idx] = 1   # top
                    if i_idx + 1 < nrow:
                        border[i_idx + 1][j_idx] = 1   # down
                    if j_idx - 1 >= 0:
                        border[i_idx][j_idx - 1] = 1   # left
                    if j_idx + 1 < ncol:
                        border[i_idx][j_idx + 1] = 1   # right
        
        res.append(border)
    
    return torch.stack(res, dim=0).cuda()

def link_loss2(gts, outs, feats, ps=8):
    # gts [bs, H, W]  , 注意有255
    # outs [bs, H, W]  
    # feats [bs, 2048, H, W]
    bs, c, h, w = feats.shape
    feats = feats.permute((0, 2, 3, 1)) # [bs, H, W, 2048]
    borders = next_border(outs, ps)                              # [bs, nrow, ncol]
    gts_ps, preds_ps = gen_correct_map(gts, outs, ps)            # [bs, nrow, ncol]
    
    record = defaultdict(int)
    nums = defaultdict(int)
    
    condition = (gts_ps != 255) & (borders != -1)
    correct = ((gts_ps == preds_ps) & condition) * 1      # & borders != -1                     # [bs, nrow, ncol]
    idxi, idxj, idxk = torch.where(correct == 1)
    # select_idx = torch.rand(size=(len(idxi),))
    # select_idx[select_idx >= 0.8] = True
    # select_idx[select_idx < 0.8] = False
    # select_idx = select_idx.bool()
    # idxi = idxi[select_idx]#[:1000]
    # idxj = idxj[select_idx]#[:1000]
    # idxk = idxk[select_idx]#[:1000]
    
    leni = len(idxi)
    # print('success', leni)
    for i in range(leni):
        ii, jj, kk = idxi[i], idxj[i], idxk[i]
        cls_gt = gts_ps[ii][jj][kk].item()
        record[cls_gt] += feats[ii][jj][kk][:]
        nums[cls_gt] += 1
     
    loss = 0
    fail = ((gts_ps != preds_ps) & condition) * 1                            # [bs, nrow, ncol]
    idxi, idxj, idxk = torch.where(fail == 1)
    leni = len(idxi)
    empty = 0
    # print('fail', leni)
    for i in range(leni):
        ii, jj, kk = idxi[i], idxj[i], idxk[i]
        cls_gt = gts_ps[ii][jj][kk].item()
        if nums[cls_gt] == 0:
            empty += 1
            continue
        feat = feats[ii][jj][kk][:]
        feat = feat / feat.norm()
        target = (record[cls_gt] / nums[cls_gt])
        target = target / target.norm()
        loss = loss + 1 - F.cosine_similarity(feat, target, dim=0)
        # loss = loss + F.kl_div(feat.log_softmax(dim=0), target.softmax(dim=0))
    loss = loss / leni
    
    del record, nums
    # print('empty', empty)
    return loss


def link_loss3(gts, outs, feats, ps=8):
    # gts [bs, H, W]  , 注意有255
    # outs [bs, H, W]  
    # feats [bs, 2048, H, W]
    bs, c, h, w = feats.shape
    feats = feats.permute((0, 2, 3, 1)) # [bs, H, W, 2048]
    gts_ps = F.interpolate(gts.unsqueeze(1).float(), size=(h, w)).squeeze(1)             # nearst
    outs = F.interpolate(outs.unsqueeze(1).float(), size=(h, w)).squeeze(1)              # nearst
    
    condition = (gts_ps != 255)
    correct = ((gts_ps == outs) & condition) * 1      # [bs, nrow, ncol]
    fail = ((gts_ps != outs) & condition) * 1      # [bs, nrow, ncol]
    idxi, idxj, idxk = torch.where(correct == 1)

    leni = len(idxi)
    record = defaultdict(list)
    
    # print('success', leni)
    for i in range(leni):
        ii, jj, kk = idxi[i], idxj[i], idxk[i]
        cls_gt = gts_ps[ii][jj][kk].item()
        record[cls_gt].append(feats[ii][jj][kk][:].detach().cpu().numpy())
            
    centers = defaultdict(int)
    estimator = KMeans(n_clusters=1) # 构造聚类器
    
    for cls in record.keys():
        estimator.fit(record[cls])               # 聚类
        x = torch.tensor(estimator.cluster_centers_).cuda()   # [1, 1024]
        centers[cls] = x / x.norm(dim=1, keepdim=True)
    
    loss = 0
    elems = torch.sum(fail * 1).item() + 1
    feat = feats.reshape(-1, c)      # feat: [bs*h*w, c]
    feat = feat / feat.norm(dim=1, keepdim=True)
    for cls in record.keys():
        fail_cls = fail & (gts_ps == cls)       # [bs, nrow, ncol] 
        target = centers[cls]   #  [1, c]
        # target = target / target.norm(dim=1, keepdim=True)
        simm_loss = 1 - F.cosine_similarity(feat, target, dim=1)
        simm_loss = simm_loss.reshape(bs, h, w) * fail_cls
        loss = loss + simm_loss
        
    return loss.sum() / elems

def focal_loss(pred, gt, criterion, gamma=2, reduction="mean"):
    loss = criterion(pred, gt)
    pt = torch.exp(-loss)
    weights = (1 - pt) ** gamma
    focal_loss = weights * loss

    if reduction == "mean":
        focal_loss = focal_loss.mean()
    elif reduction == "sum":
        focal_loss = focal_loss.sum()
    else:
        raise ValueError(f"reduction {reduction} is not vaild")

    return focal_loss


def metric(pred, mask, num_classes):
    crossList = np.array([0 for _ in range(num_classes)])
    unionList = np.array([0 for _ in range(num_classes)])

    pred = pred.squeeze().long()
    mask = mask.squeeze().long()

    assert pred.shape == mask.shape, 'pred and mask should be same in shape'

    for class_id in range(num_classes):
        pred_one_class = (pred == class_id) * 1.0
        mask_one_class = (mask == class_id) * 1.0

        pred_one_class = pred_one_class.cpu().numpy()
        mask_one_class = mask_one_class.cpu().numpy()

        cross = np.logical_and(pred_one_class, mask_one_class).sum()
        union = np.logical_or(pred_one_class, mask_one_class).sum()

        crossList[class_id] = cross
        unionList[class_id] = union

    return crossList, unionList

def collate_fn(batch):
    '''
        batch: list, lens = bs
        batch[idx]: dict
        batch[idx]['im']: image tensor
        batch[idx]['gt_semantic_seg']: mask tensor
    '''

    bs = len(batch)
    ims = [batch[idx][0] for idx in range(bs)]
    msks = [batch[idx][1] for idx in range(bs)]
    paths = [batch[idx][2] for idx in range(bs)]
    patch_size = batch[0][3]                                 # patch_size
    batch_img = torch.stack(ims, dim=0)
    batch_mask = torch.stack(msks, dim=0)

    link_graphs = gen_link_gt(batch_mask, patch_size=patch_size)   # patch_size
    return batch_img, batch_mask, link_graphs, paths

def plotfig(data_list, title, xlabel='Epoch', ylabel=None, savepath=None):
    plt.plot(range(len(data_list)), data_list)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(savepath)
    plt.clf()
    

def reduce_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= dist.get_world_size()
    return rt  


def is_master():
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def setup_for_distributed():
    """
    This function disables printing when not in master process
    """
    import builtins
    import datetime

    builtin_print = builtins.print

    def print(*args, **kwargs):   # 将非主进程的print功能取消, 只保留主进程的print功能, 并设置打印格式
        if is_master():           
            now = datetime.datetime.now()
            builtin_print(f'[{now.month}-{now.day:02} {now.hour:02}:{now.minute:02}]', end=' ')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print
    

# 颜色画板
palette = np.array([                          
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128]
            ], dtype='uint8').flatten()


def setcolor(img):
    img = Image.fromarray(img.astype(np.uint8))
    img.putpalette(palette)                           # putpalette函数没有返回值
    return img
