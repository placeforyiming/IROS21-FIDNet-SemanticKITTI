
import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.distributed
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from network.ResNet import *
import argparse
from torchvision.transforms import functional as FF
from dataloader.Dataset_semanticKITTI import *
from torch.autograd import Variable
import torch.optim.lr_scheduler as toptim

try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse as ifilterfalse


def NN_filter(proj_range,semantic_pred,k_size=5):
    semantic_pred=semantic_pred.double()
    H,W=np.shape(proj_range)
    
    proj_range_expand=torch.unsqueeze(proj_range,axis=0)
    proj_range_expand=torch.unsqueeze(proj_range_expand,axis=0)
    
    semantic_pred_expand=torch.unsqueeze(semantic_pred,axis=0)
    semantic_pred_expand=torch.unsqueeze(semantic_pred_expand,axis=0)
    
    pad = int((k_size - 1) / 2)

    proj_unfold_range = Func.unfold(proj_range_expand,kernel_size=(k_size, k_size),padding=(pad, pad))
    proj_unfold_range = proj_unfold_range.reshape(-1, k_size*k_size, H, W)
        
    proj_unfold_pre = Func.unfold(semantic_pred_expand,kernel_size=(k_size, k_size),padding=(pad, pad))
    proj_unfold_pre = proj_unfold_pre.reshape(-1, k_size*k_size, H, W)
    
    return proj_unfold_range,proj_unfold_pre
        


class CE_Weight:
	@staticmethod
	def get_weight():
		return [0.0,
	1.0/(0.040818519255974316+0.001789309418528068+0.001),
	1.0/(0.00016609538710764618+0.001),
	1.0/(0.00039838616015114444+0.001),
	1.0/(0.0020633612104619787+0.00010157861367183268+0.001),
	1.0/(2.7879693665067774e-05+0.0016218197275284021+0.00011351574470342043+4.3840131989471124e-05+0.001),
	1.0/(0.00017698551338515307+0.00016059776092534436+0.001),
	1.0/(1.1065903904919655e-08+0.00012709999297008662+0.001),
	1.0/(5.532951952459828e-09+3.745553104802113e-05+0.001),
	1.0/(0.1987493871255525+4.7084144280367186e-05+0.001),
	1.0/(0.014717169549888214+0.001),
	1.0/(0.14392298360372+0.001),
	1.0/(0.0039048553037472045+0.001),
	1.0/(0.1326861944777486+0.001),
	1.0/(0.0723592229456223+0.001),
	1.0/(0.26681502148037506+0.001),
	1.0/(0.006035012012626033+0.001),
	1.0/(0.07814222006271769+0.001),
	1.0/(0.002855498193863172+0.001),
	1.0/(0.0006155958086189918+0.001)]

	@staticmethod
	def get_bin_weight(bin_num):
		weight_list=[]
		for i in range(bin_num+1):
			weight_list.append(abs(i/float(bin_num)-0.5)*2+0.2)
		return weight_list

def get_semantic_segmentation(sem):
	# map semantic output to labels
	if sem.size(0) != 1:
		raise ValueError('Only supports inference for batch size = 1')
	sem = sem.squeeze(0)
	predict_pre=torch.argmax(sem, dim=0, keepdim=True)
	'''
	sem_prob=Func.softmax(sem,dim=0)
	change_mask_motorcyclist=torch.logical_and(predict_pre==7,sem_prob[8:9,:,:]>0.1)
	predict_pre[change_mask_motorcyclist]=8
	'''
	return predict_pre





def get_lr_manually_1(current_epoch):
    current_epoch=int(current_epoch/2)
    if current_epoch==0:
        return 0.0004
    if current_epoch==1:
        return 0.0008
    if current_epoch==2:
        return 0.0012
    if current_epoch==3:
        return 0.0016
    if current_epoch==4:
        return 0.002
    if current_epoch==5:
        return 0.0016
    if current_epoch==6:
        return 0.0012
    if current_epoch==7:
        return 0.0008
    if current_epoch==8:
        return 0.0012
    if current_epoch==9:
        return 0.0006
    if current_epoch==10:
        return 0.0008
    if current_epoch==11:
        return 0.0004
    if current_epoch==12:
        return 0.0002



def get_lr_manually_2(current_epoch):
	current_epoch=int(current_epoch/2)
	if current_epoch==0:
		return 0.0004
	if current_epoch==1:
		return 0.0008
	if current_epoch==2:
		return 0.0012
	if current_epoch==3:
		return 0.0016
	if current_epoch==4:
		return 0.002
	if current_epoch==5:
		return 0.0018
	if current_epoch==6:
		return 0.0016
	if current_epoch==7:
		return 0.0014
	if current_epoch==8:
		return 0.0012
	if current_epoch==9:
		return 0.001
	if current_epoch==10:
		return 0.0008
	if current_epoch==11:
		return 0.0006
	if current_epoch==12:
		return 0.0004
	if current_epoch==13:
		return 0.0002
	if current_epoch>13:
		return 0.0002
	




def prepare_input_label_semantic(sample,if_remission=False, if_range=False):
    data_num=len(sample)
    input_tensor=[]
    semantic_label=[]
    semantic_label_mask=[]
    for i in range(data_num):
        if if_remission and not if_range:
            each_input=[sample[i]['xyz'],sample[i]['remission']]
            input_tensor.append(torch.cat(each_input,axis=0))
        if if_remission and if_range:
            each_input=[sample[i]['xyz'],sample[i]['remission'],sample[i]['range_img']]
            input_tensor.append(torch.cat(each_input,axis=0))
        if not if_remission and not if_range:
            input_tensor.append(sample[i]['xyz'])
        semantic_label.append(sample[i]['semantic_label'][0,:,:])
        semantic_label_mask.append(sample[i]['xyz_mask'][0,:,:])
    return input_tensor,semantic_label,semantic_label_mask




def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                    for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    #vprobas = probas[valid.nonzero().squeeze()]
    vprobas = probas[torch.squeeze(torch.nonzero(valid))]
    vlabels = labels[valid]
    return vprobas, vlabels


class Lovasz_softmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore=None):
        super(Lovasz_softmax, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, probas, labels):
        return lovasz_softmax(probas, labels, self.classes, self.per_image, self.ignore)



def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp