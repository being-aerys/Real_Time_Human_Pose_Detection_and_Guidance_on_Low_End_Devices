import torch
from torch import nn

from modules.conv import conv, conv_dw, conv_dw_no_bn


class Cpm(nn.Module):
    def __init__(self, in_channels, out_channels):#--------512,128
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv_dw_no_bn(out_channels, out_channels),  #kernel_size=3, padding=1, stride=1, dilation=1
            conv_dw_no_bn(out_channels, out_channels),  #kernel_size=3, padding=1, stride=1, dilation=1
            conv_dw_no_bn(out_channels, out_channels)   #kernel_size=3, padding=1, stride=1, dilation=1
        )
        self.conv = conv(out_channels, out_channels, bn=False)

    def forward(self, x):
        x = self.align(x)
        x = self.conv(x + self.trunk(x))#------------------------------------------------concatenation from mobilenet and trunk
        return x


# class InitialStage(nn.Module):
#     def __init__(self, num_channels, num_heatmaps, num_pafs):
#         super().__init__()
#         self.trunk = nn.Sequential(#----------------------------------------Original OPENPOSE contained two copies of these conv() layers, for PAF and HMs
#             conv(num_channels, num_channels, bn=False),#--------------------Here just 1 in the original lightweight openpose
#             conv(num_channels, num_channels, bn=False),
#             conv(num_channels, num_channels, bn=False)
#         )
#         #--------------------------------------------------------------------Then, give the op to both paf and HM branches
#         self.heatmaps = nn.Sequential(
#             conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
#             conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
#         )
#         self.pafs = nn.Sequential(
#             conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
#             conv(512, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
#         )
#
#     def forward(self, x):
#         trunk_features = self.trunk(x)
#         heatmaps = self.heatmaps(trunk_features)
#         pafs = self.pafs(trunk_features)
#         return [heatmaps, pafs]
#
#
# class RefinementStageBlock(nn.Module):#---------------------------------Each refinement stage block should be a [3*3 , 3*3 , 1*1] with residual connection
#     def __init__(self, in_channels, out_channels):#---------------------for each such block
#         super().__init__()
#         self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
#         self.trunk = nn.Sequential(
#             conv(out_channels, out_channels),
#             conv(out_channels, out_channels, dilation=2, padding=2)
#         )
#
#     def forward(self, x):
#         initial_features = self.initial(x)
#         trunk_features = self.trunk(initial_features)
#         return initial_features + trunk_features
#
#
# class RefinementStage(nn.Module):#------------------------------------------original lightweight contains a set of
#     def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs):
#         super().__init__()
#         self.trunk = nn.Sequential(#----------------------------------------------------In OpenPose five 7 * 7,
#             RefinementStageBlock(in_channels, out_channels),#---------------------------here replaced with  five time [3*3 , 3*3 , 1*1]
#             RefinementStageBlock(out_channels, out_channels),
#             RefinementStageBlock(out_channels, out_channels),
#             RefinementStageBlock(out_channels, out_channels),
#             RefinementStageBlock(out_channels, out_channels)
#         )
#         self.heatmaps = nn.Sequential(
#             conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
#             conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
#         )
#         self.pafs = nn.Sequential(
#             conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
#             conv(out_channels, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
#         )
#
#     def forward(self, x):
#         trunk_features = self.trunk(x)
#         heatmaps = self.heatmaps(trunk_features)
#         pafs = self.pafs(trunk_features)
#         return [heatmaps, pafs]


#--------------------------------------------------------------------------------New Initial Stages-------------------------------------------------

class InitialStage(nn.Module):
    def __init__(self, num_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False),
            #conv(num_channels, num_channels, bn=False)#-----------------------------------remove one convolution layer
        )
        self.heatmaps = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]
#----------------------------------------------------------------------------------New Refinement Stage block------------------------------------------

class RefinementStageBlock(nn.Module):#------------------------------------------------This one block contains
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features


#-------------------------------------------------------------------------------New Refinement Stage------------------------------------------------------
class RefinementStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            #RefinementStageBlock(out_channels, out_channels),#-----------------remove two blocks from the single refinements stage
            #RefinementStageBlock(out_channels, out_channels)
        )
        self.heatmaps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]
#------------------------------------------------------------------------------------------------------------------------------------------------------------


class PoseEstimationWithMobileNet(nn.Module):
    def __init__(self, num_refinement_stages=1, num_channels=128, num_heatmaps=19, num_pafs=38):#--------------why 38

        super().__init__()
        self.model = nn.Sequential(#---------------------------------------------------MobileNet-----------------------------------first
            conv(     3,  32, stride=2, bias=False),
            conv_dw( 32,  64),#---------------------------------------------dw means depthwise convolution
            conv_dw( 64, 128, stride=2),
            conv_dw(128, 128),
            conv_dw(128, 256, stride=2),
            conv_dw(256, 256),
            conv_dw(256, 512),  # Stride of this conv4_2 removed from MobileNetv1 to preserve the receptive field. conv4_2 means kernel 4 * 4, padding 2
            conv_dw(512, 512, dilation=2, padding=2),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512)   # conv5_5 The last layer used from MobileNet is this one.
        )
        self.cpm = Cpm(512, num_channels)#-------------------------------------------------What is this CPM?--------------second

        self.initial_stage = InitialStage(num_channels, num_heatmaps, num_pafs)#--------------------------------------------third



        self.refinement_stages = nn.ModuleList()#------------------------------------------------------------------------fourth
        for idx in range(num_refinement_stages):
            self.refinement_stages.append(RefinementStage(num_channels + num_heatmaps + num_pafs, num_channels,
                                                          num_heatmaps, num_pafs))

    def forward(self, x):

        backbone_features = self.model(x)
        backbone_features = self.cpm(backbone_features)#----------------------get backbone features

        stages_output = self.initial_stage(backbone_features)#----------------get output of the initial stage from the backbone features

        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(torch.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)))#----------refinement stages

        return stages_output
