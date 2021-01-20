import torch
import torch.nn as nn
from torchvision.models import resnet50


class RepNetPeriodEstimator(nn.Module):

    def __init__(self,
                 num_frames = 64,
                 image_size = 112,
                 base_model_layer_name = 'conv4_block3_out',
                 temperature = 13.544,
                 dropout_rate = 0.25,
                 l2_reg_weight = 1e-6,
                 temporal_conv_channels = 512,
                 temporal_conv_kernel_size = 3,
                 temporal_conv_dilation_rate = 3,
                 conv_channels = 32,
                 conv_kernel_size = 3,
                 transformer_layers_config = ((512, 4, 512),),
                 transformer_dropout_rate = 0.0,
                 transformer_reorder_ln = True,
                 period_fc_channels = (512, 512),
                 within_period_fc_channels = (512, 512)):
        super(RepNetPeriodEstimator, self).__init__()

        # model parameters
        self.num_frames = num_frames
        self.image_size = image_size

        self.base_model_layer_name = base_model_layer_name

        self.temperature = temperature

        self.dropout_rate = dropout_rate
        self.l2_reg_weight = l2_reg_weight

        self.temporal_conv_channels = temporal_conv_channels
        self.temporal_conv_kernel_size = temporal_conv_kernel_size
        self.temporal_conv_dilation_rate = temporal_conv_dilation_rate

        self.conv_channels = conv_channels
        self.conv_kernel_size = conv_kernel_size
        # Transformer config in form of (channels, heads, bottleneck channels).
        self.transformer_layers_config = transformer_layers_config
        self.transformer_dropout_rate = transformer_dropout_rate
        self.transformer_reorder_ln = transformer_reorder_ln

        self.period_fc_channels = period_fc_channels
        self.within_period_fc_channels = within_period_fc_channels

        # get resnet50 backbone
        