import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import copy
EPS = 1E-20







class PullNoisePerturb(object):
    def __init__(self, model, proxy):
        super(AdvWeightPerturb, self).__init__()
        self.model = model
        self.proxy = proxy

    def calc_pertube(self, pull, scale, num_layer_modify):
        with torch.no_grad():
                conv_layer_cter = 0
                conv_para_noise = copy.deepcopy()
                for para in conv_para_noise():
                    if len(para.shape) == 4:
                        if num_layer_modify > conv_layer_cter:
                            para_mean = torch.mean(para)
                            noise = torch.randn(para.size()) * scale * para_mean
                            for s1 in range(len(para)):
                                for s2 in range(len(para[s1])):
                                    for s3 in range(len(para[s1][s2])):
                                        for s4 in range(len(para[s1][s2][s3])):
                                            if para[s1][s2][s3][s4] > para_mean:
                                                if pull == "away":
                                                    noise[s1][s2][s3][s4] = abs(pull_in_noise[s1][s2][s3][s4])
                                                elif pull == "in":
                                                    noise[s1][s2][s3][s4] = -abs(pull_in_noise[s1][s2][s3][s4])
                                                elif pull == "none":
                                                    noise[s1][s2][s3][s4] = pull_in_noise[s1][s2][s3][s4]
                                            else:
                                                if pull == "away":
                                                    noise[s1][s2][s3][s4] = -abs(pull_in_noise[s1][s2][s3][s4])
                                                elif pull == "in":
                                                    noise[s1][s2][s3][s4] = abs(pull_in_noise[s1][s2][s3][s4])
                                                elif pull == "none":
                                                    noise[s1][s2][s3][s4] = pull_in_noise[s1][s2][s3][s4]
                            para.add_((noise).to(device))
                        conv_layer_cter = conv_layer_cter + 1

        return conv_para_noise







