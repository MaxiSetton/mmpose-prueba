import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import yaml
import json
import tensorflow as tf
from itertools import groupby
import pickle

class AttPositionalEncoding(nn.Module):

    def __init__(self, channel, joint_num, time_len, domain):
        super(AttPositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len

        self.domain = domain

        if domain == "temporal":
            # temporal embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(t)
        elif domain == "spatial":
            # spatial embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(j_id)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()
        pe = torch.zeros(self.time_len * self.joint_num, channel)

        div_term = torch.exp(torch.arange(0, channel, 2).float() *
                             -(math.log(10000.0) / channel))  # channel//2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # nctv
        x = x + self.pe[:, :, :x.size(2)]
        return x

class STAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels, num_subset=2, num_node=27, num_frame=400,
                 kernel_size=1, stride=1, t_kernel=3, glo_reg_s=True, att_s=True, glo_reg_t=False, att_t=False,
                 use_temporal_att=False, use_spatial_att=True, attentiondrop=0., use_pes=True, use_pet=False):
        super(STAttentionBlock, self).__init__()
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset = num_subset
        self.glo_reg_s = glo_reg_s
        self.att_s = att_s
        self.glo_reg_t = glo_reg_t
        self.att_t = att_t
        self.use_pes = use_pes
        self.use_pet = use_pet
        pad = int((kernel_size - 1) / 2)
        self.use_spatial_att = use_spatial_att
        if use_spatial_att:
            atts = torch.zeros((1, num_subset, num_node, num_node))
            self.register_buffer('atts', atts)
            self.pes = AttPositionalEncoding(in_channels, num_node, num_frame, 'spatial')
            self.ff_nets = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if att_s:
                self.in_nets = nn.Conv2d(in_channels, 2 * num_subset * inter_channels, 1, bias=True)
                self.alphas = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
            if glo_reg_s:
                self.attention0s = nn.Parameter(torch.ones(1, num_subset, num_node, num_node) / num_node,
                                                requires_grad=True)

            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels * num_subset, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1), bias=True, stride=1),
                nn.BatchNorm2d(out_channels),
            )
        padd = int(t_kernel / 2)
        self.out_nett = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (t_kernel, 1), padding=(padd, 0), bias=True, stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
        )

        if in_channels != out_channels or stride != 1:
            if use_spatial_att:
                self.downs1 = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downs2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if use_temporal_att:
                self.downt1 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 1, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
        else:
            if use_spatial_att:
                self.downs1 = lambda x: x
            self.downs2 = lambda x: x
        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)

    def forward(self, x):
        N, C, T, V = x.size()
        if self.use_spatial_att:
            attention = self.atts
            if self.use_pes:
                y = self.pes(x)
            else:
                y = x
            if self.att_s:
                q, k = torch.chunk(self.in_nets(y).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2,
                                   dim=1)  # nctv -> n num_subset c'tv [4,16,t,v]
                attention = attention + self.tan(torch.einsum('nsctu,nsctv->nsuv', [q, k]) / (self.inter_channels * T)) * self.alphas
            if self.glo_reg_s:
                attention = attention + self.attention0s.repeat(N, 1, 1, 1)
            attention = self.drop(attention)
            y = torch.einsum('nctu,nsuv->nsctv', [x, attention]).contiguous() \
                .view(N, self.num_subset * self.in_channels, T, V)
            y = self.out_nets(y)  # nctv
            y = self.relu(self.downs1(x) + y)
            y = self.ff_nets(y)
            y = self.relu(self.downs2(x) + y)
        else:
            y = self.out_nets(x)
            y = self.relu(self.downs2(x) + y)
        z = self.out_nett(y)
        z = self.relu(y + z)

        return z

class AttentionStage(nn.Module):
    def __init__(self, in_channels, out_channels, t_kernel_sizes, num_blocks, num_frame=400, dropout=0.1):
        super(AttentionStage, self).__init__()
        num_pathways=len(in_channels)
        self.num_frame=num_frame
        self.num_blocks = num_blocks
        self.num_pathways = 2

        num_frame = self.num_frame
        for pathway in range(num_pathways):
            num_frame = self.num_frame
            for i in range(num_blocks):
                attention_block = STAttentionBlock(
                    in_channels[pathway] if i == 0 else out_channels[pathway],
                    out_channels[pathway], out_channels[pathway]//4, stride=1, num_node=79,
                                    t_kernel=t_kernel_sizes[pathway], num_frame=num_frame)
                self.add_module("pathway{}_att{}".format(pathway, i), attention_block)
        self.drop_out = nn.Dropout(dropout)

    def forward(self, inputs):
        for pathway in range(self.num_pathways):
            x = inputs[pathway]
            for i in range(self.num_blocks):
                m = getattr(self, 'pathway{}_att{}'.format(pathway,i))
                x = m(x)
            if pathway==0:
                x_s=x
            else:
                x_f=x
        return x_s, x_f

class FuseBiAdd(nn.Module):
    """
    Fuses the information from each pathway to other pathway through addition.
    Given the tensors from Slow pathway and Fast pathway, fuse information in bidirectional,
    then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        fusion_kernel,
        alpha,
        beta_inv,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
        norm_module=nn.BatchNorm2d,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(FuseBiAdd, self).__init__()
        self.conv_f2s = nn.Conv2d(
            dim_in,
            dim_in * beta_inv,
            kernel_size=[fusion_kernel, 1],
            stride=[alpha, 1],
            padding=[fusion_kernel // 2, 0],
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_in * beta_inv,
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu = nn.ReLU(inplace_relu)
        self.conv_s2f = nn.Conv2d(
            dim_in * beta_inv,
            dim_in,
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0],
            bias=False,
        )
        self.bn2 = norm_module(
            num_features=dim_in,
            eps=eps,
            momentum=bn_mmt
        )
        self.alpha = alpha
        self.weight_s = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.weight_f = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]

        fuse1 = self.conv_f2s(x_f)
        fuse1 = self.bn(fuse1)
        fuse1 = self.relu(fuse1)
        x_s_fuse = x_s + self.weight_s * fuse1
        fuse2 = self.conv_s2f(x_s)
        fuse2 = self.bn2(fuse2)
        fuse2 = self.relu(fuse2)
        fuse2 = nn.functional.interpolate(fuse2, x_f.shape[2:])
        x_f_fuse = x_f + self.weight_f * fuse2
        return [x_s_fuse, x_f_fuse]

class SlowFastLoader(nn.Module):
    def __init__(self, alpha):
        super(SlowFastLoader, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        assert(len(x.shape) == 4)

        x_f = x[:]
        x_s = x[:,:,::self.alpha]
        return x_s, x_f

class SlowFast(nn.Module):
    def __init__(self, cfg):
        super(SlowFast, self).__init__()
        self.cfg = cfg
        self.num_pathways = 2
        self.alpha=cfg['Alpha']
        self.loader = SlowFastLoader(self.alpha)
        t_kernel_sizes = cfg['temporal_kernels']
        fusion_kernel_size= cfg['fusion_kernel_size']
        beta=cfg['Beta']
        input_channels = cfg['input_channels']
        out_channels = input_channels*4

        self.slow_input_map = nn.Sequential(
            nn.Conv2d(3, input_channels, 1),
            nn.BatchNorm2d(input_channels),
            nn.LeakyReLU(0.1),
        )
        self.fast_input_map = nn.Sequential(
            nn.Conv2d(3, input_channels//4, 1),
            nn.BatchNorm2d(input_channels//4),
            nn.LeakyReLU(0.1),
        )

        self.stage1 = AttentionStage(
            [input_channels, input_channels//4],
            [out_channels, out_channels//beta],
            t_kernel_sizes[0],
            num_blocks= 2
        )

        self.s1_fuse = FuseBiAdd(
            out_channels // beta,
            fusion_kernel_size,
            self.alpha,
            beta
        )

        self.stage2 = AttentionStage(
            [input_channels*4, input_channels*4//beta],
            [out_channels*2, out_channels*2//beta],
            t_kernel_sizes[1],
            num_blocks= 2
        )
        self.s2_fuse = FuseBiAdd(
            out_channels*2 // beta,
            fusion_kernel_size,
            self.alpha,
            beta
        )

        self.stage3 = AttentionStage(
            [input_channels*8, input_channels*8//beta],
            [out_channels*4, out_channels*4//beta],
            t_kernel_sizes[2],
            num_blocks= 2
        )

        self.s3_fuse = FuseBiAdd(
            out_channels*4 // beta,
            fusion_kernel_size,
            self.alpha,
            beta,
        )

        self.stage4 = AttentionStage(
            [input_channels*16, input_channels*16//beta],
            [out_channels*8, out_channels*8//beta],
            t_kernel_sizes[3],
            num_blocks= 2
        )

        self.dropout=nn.Dropout(0.3)

    def forward(self, x):
        x_s, x_f = self.loader(x)
        x_s = self.slow_input_map(x_s)
        x_f = self.fast_input_map(x_f)
        x = (x_s, x_f)
        x = self.stage1(x)
        x = self.s1_fuse(x)
        x = self.stage2(x)
        x = self.s2_fuse(x)
        x = self.stage3(x)
        x = self.s3_fuse(x)
        x = self.stage4(x)
        x_s = x[0].mean(3)
        x_s = self.dropout(nn.functional.interpolate(x_s, scale_factor=(self.alpha)).permute(0,2,1))
        x_f = self.dropout(x[1].permute(0,2,1,3).mean(3))

        return (x_s,x_f)

class PositionalEncoding(nn.Module):
    """
    Pre-compute position encodings (PE).
    In forward pass, this adds the position-encodings to the
    input for as many time steps as necessary.
    Implementation based on OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, size: int = 0, max_len: int = 5000):
        """
        Positional Encoding with maximum length max_len
        :param size:
        :param max_len:
        :param dropout:
        """
        if size % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(size)
            )
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, size, 2, dtype=torch.float) * -(math.log(10000.0) / size))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, size, max_len]
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pe", pe)
        self.dim = size

    def forward(self, emb):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
        """
        # Add position encodings
        return emb + self.pe[:, : emb.size(1)]

class MaskedNorm(nn.Module):
    """
        Original Code from:
        https://discuss.pytorch.org/t/batchnorm-for-different-sized-samples-in-batch/44251/8
    """

    def __init__(self, num_features=512, norm_type='sync_batch', num_groups=1):
        super().__init__()
        self.norm_type = norm_type
        if self.norm_type == "batch":
            # raise ValueError("Please use sync_batch")
            self.norm = nn.BatchNorm1d(num_features=num_features)
        elif self.norm_type == 'sync_batch':
            self.norm = nn.SyncBatchNorm(num_features=num_features)
        elif self.norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
        elif self.norm_type == "layer":
            self.norm = nn.LayerNorm(normalized_shape=num_features)
        else:
            raise ValueError("Unsupported Normalization Layer")

        self.num_features = num_features

    def forward(self, x):
        if self.training:
            mask = (x.abs().sum(dim=-1) != 0).unsqueeze(-1)  # (batch, time, 1)
            reshaped = x.reshape([-1, self.num_features])
            reshaped_mask = mask.reshape([-1, 1]) > 0
            selected = torch.masked_select(reshaped, reshaped_mask).reshape(
                [-1, self.num_features]
            )
            batch_normed = self.norm(selected)
            scattered = reshaped.masked_scatter(reshaped_mask, batch_normed)
            return scattered.reshape([x.shape[0], -1, self.num_features])
        else:
            reshaped = x.reshape([-1, self.num_features])
            batched_normed = self.norm(reshaped)
            return batched_normed.reshape([x.shape[0], -1, self.num_features])

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, input_size, ff_size, dropout=0.1, kernel_size=1,
        skip_connection=True):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super(PositionwiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.kernel_size = kernel_size
        if type(self.kernel_size)==int:
            conv_1 = nn.Conv1d(input_size, ff_size, kernel_size=kernel_size, stride=1, padding='same')
            conv_2 = nn.Conv1d(ff_size, input_size, kernel_size=kernel_size, stride=1, padding='same')
            self.pwff_layer = nn.Sequential(
                conv_1,
                nn.ReLU(),
                nn.Dropout(dropout),
                conv_2,
                nn.Dropout(dropout),
            )
        elif type(self.kernel_size)==list:
            pwff = []
            first_conv = nn.Conv1d(input_size, ff_size, kernel_size=kernel_size[0], stride=1, padding='same')
            pwff += [first_conv, nn.ReLU(), nn.Dropout(dropout)]
            for ks in kernel_size[1:-1]:
                conv = nn.Conv1d(ff_size, ff_size, kernel_size=ks, stride=1, padding='same')
                pwff += [conv, nn.ReLU(), nn.Dropout(dropout)]
            last_conv = nn.Conv1d(ff_size, input_size, kernel_size=kernel_size[-1], stride=1, padding='same')
            pwff += [last_conv, nn.Dropout(dropout)]

            self.pwff_layer = nn.Sequential(
                *pwff
            )
        else:
            raise ValueError
        self.skip_connection=skip_connection
        if not skip_connection:
            print('Turn off skip_connection in PositionwiseFeedForward')

    def forward(self, x):
        x_norm = self.layer_norm(x)
        x_t = x_norm.transpose(1,2)
        x_t = self.pwff_layer(x_t)
        if self.skip_connection:
            return x_t.transpose(1,2)+x
        else:
            return x_t.transpose(1,2)


class VisualHead(torch.nn.Module):
    def __init__(self,
        cls_num, input_size, hidden_size=1024, ff_size=2048, pe=True,
        ff_kernelsize=[3,3], pretrained_ckpt=None, is_empty=False, frozen=False):
        super().__init__()
        self.is_empty = is_empty
        if is_empty==False:
            self.frozen = frozen
            self.hidden_size = hidden_size

            if input_size is None:
                self.fc1 = nn.Identity()
            else:
                self.fc1 = torch.nn.Linear(input_size, self.hidden_size)
            self.bn1 = MaskedNorm(num_features=self.hidden_size, norm_type='batch')
            self.relu1 = torch.nn.ReLU()
            self.dropout1 = torch.nn.Dropout(p=0.1)

            if pe:
                self.pe = PositionalEncoding(self.hidden_size)
            else:
                self.pe = torch.nn.Identity()

            self.feedforward = PositionwiseFeedForward(input_size=self.hidden_size,
                ff_size=ff_size,
                dropout=0.1, kernel_size=ff_kernelsize, skip_connection=True)

            self.layer_norm = torch.nn.LayerNorm(self.hidden_size, eps=1e-6)

            self.gloss_output_layer = torch.nn.Linear(self.hidden_size, cls_num)

            if self.frozen:
                self.frozen_layers = [self.fc1, self.bn1, self.relu1,  self.pe, self.dropout1, self.feedforward, self.layer_norm]
                for layer in self.frozen_layers:
                    for name, param in layer.named_parameters():
                        param.requires_grad = False
                    layer.eval()
        else:
            self.gloss_output_layer = torch.nn.Linear(input_size, cls_num)
        if pretrained_ckpt:
            self.load_from_pretrained_ckpt(pretrained_ckpt)

    def load_from_pretrained_ckpt(self, pretrained_ckpt):
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')['model_state']
        load_dict = {}
        for k,v in checkpoint.items():
            if 'recognition_network.visual_head.' in k:
                load_dict[k.replace('recognition_network.visual_head.','')] = v
        self.load_state_dict(load_dict)

    def forward(self, x):
        B, Tin, D = x.shape
        if self.is_empty==False:
            if not self.frozen:
                #projection 1
                x = self.fc1(x)
                x = self.bn1(x)
                x = self.relu1(x)
                #pe
                x = self.pe(x)
                x = self.dropout1(x)

                #feedforward
                x = self.feedforward(x)
                x = self.layer_norm(x)
            else:
                with torch.no_grad():
                    for ii, layer in enumerate(self.frozen_layers):
                        layer.eval()
                        x = layer(x)

        #classification
        logits = self.gloss_output_layer(x) #B,T,V
        gloss_probabilities_log = logits.log_softmax(2)
        gloss_probabilities = logits.softmax(2)

        return {'gloss_feature': x,
                'gloss_feature_norm': F.normalize(x, dim=-1),
                'gloss_logits':logits,
                'gloss_probabilities_log':gloss_probabilities_log,
                'gloss_probabilities': gloss_probabilities}

def ctc_decode_func(tf_gloss_logits, input_lengths, beam_size):
    ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
        inputs=tf_gloss_logits,
        sequence_length=input_lengths.cpu().detach().numpy(),
        beam_width=beam_size,
        top_paths=1,
    )
    ctc_decode = ctc_decode[0]
    tmp_gloss_sequences = [[] for i in range(input_lengths.shape[0])]
    for (value_idx, dense_idx) in enumerate(ctc_decode.indices):
        tmp_gloss_sequences[dense_idx[0]].append(
            ctc_decode.values[value_idx].numpy() + 1
        )
    decoded_gloss_sequences = []
    for seq_idx in range(0, len(tmp_gloss_sequences)):
        decoded_gloss_sequences.append(
            [x[0] for x in groupby(tmp_gloss_sequences[seq_idx])]
        )
    return decoded_gloss_sequences

class SLRModel(nn.Module):
    def __init__(self, num_cls, cfg):
        super(SLRModel, self).__init__()
        self.feature_extractor = SlowFast(cfg['RecognitionNetwork']['SlowFast'])
        self.slowhead = VisualHead(cls_num=num_cls, **cfg['RecognitionNetwork']['slow_visual_head'])
        self.fasthead = VisualHead(cls_num=num_cls, **cfg['RecognitionNetwork']['fast_visual_head'])
        self.fusehead = VisualHead(cls_num=num_cls, **cfg['RecognitionNetwork']['fuse_visual_head'])
    def forward(self, x):
        x=self.feature_extractor(x)
        x_s, x_f=x
        x_fuse = torch.cat((x_s,x_f), dim=2)
        if self.training:
          slow_outputs=self.slowhead(x_s)
          fast_outputs=self.fasthead(x_f)
        else:
          slow_outputs=[]
          fast_outputs=[]
        fuse_outputs=self.fusehead(x_fuse)
        return {'slow_outputs': slow_outputs,
                'fast_outputs': fast_outputs,
                'fuse_outputs': fuse_outputs}
    def decode(self, gls_logits, beam_size, input_lengths):
        gls_logits = gls_logits.permute(1, 0, 2) #T,B,V  [10,1,1124]
        gls_logits = gls_logits.cpu().detach().numpy()
        tf_gloss_logits = np.concatenate(
            (gls_logits[:, :, 1:], gls_logits[:, :, 0, None]),
            axis=-1,
        )
        decoded_gloss_sequences = ctc_decode_func(
            tf_gloss_logits=tf_gloss_logits,
            input_lengths=input_lengths,
            beam_size=beam_size
        )
        return decoded_gloss_sequences

from collections import defaultdict
class BaseTokenizer(object):
    def __init__(self, tokenizer_cfg):
        self.tokenizer_cfg = tokenizer_cfg
    def __call__(self, input_str):
        pass
class BaseGlossTokenizer(BaseTokenizer):
    def __init__(self, tokenizer_cfg):
        super().__init__(tokenizer_cfg)
        with open(tokenizer_cfg['gloss2id_file'], 'rb') as f:
            self.gloss2id = pickle.load(f)  #
        self.gloss2id = defaultdict(lambda: self.gloss2id['<unk>'], self.gloss2id)
        self.id2gloss = {}
        for gls, id_ in self.gloss2id.items():
            self.id2gloss[id_] = gls
        self.lower_case = tokenizer_cfg.get('lower_case', True)

    def convert_tokens_to_ids(self, tokens):
        if type(tokens) == list:
            return [self.convert_tokens_to_ids(t) for t in tokens]
        else:
            return self.gloss2id[tokens]

    def convert_ids_to_tokens(self, ids):
        if type(ids) == list:
            return [self.convert_ids_to_tokens(i) for i in ids]
        else:
            return self.id2gloss[ids]

    def __len__(self):
        return len(self.id2gloss)


class GlossTokenizer_S2G(BaseGlossTokenizer):
    def __init__(self, tokenizer_cfg):
        super().__init__(tokenizer_cfg)
        if '<s>' in self.gloss2id:
            self.silence_token = '<s>'
            self.silence_id = self.convert_tokens_to_ids(self.silence_token)
        elif '<si>' in self.gloss2id:
            self.silence_token = '<si>'
            self.silence_id = self.convert_tokens_to_ids(self.silence_token)
        else:
            raise ValueError
        assert self.silence_id == 0, (self.silence_id)

        self.pad_token = '<pad>'
        self.pad_id = self.convert_tokens_to_ids(self.pad_token)

    def __call__(self, batch_gls_seq):
        max_length = max([len(gls_seq.split()) for gls_seq in batch_gls_seq])
        gls_lengths, batch_gls_ids = [], []
        for ii, gls_seq in enumerate(batch_gls_seq):
            gls_ids = [self.gloss2id[gls.lower() if self.lower_case else gls] for gls in gls_seq.split()]
            gls_lengths.append(len(gls_ids))
            gls_ids = gls_ids + (max_length - len(gls_ids)) * [self.pad_id]
            batch_gls_ids.append(gls_ids)
        gls_lengths = torch.tensor(gls_lengths)
        batch_gls_ids = torch.tensor(batch_gls_ids)
        return {'gls_lengths': gls_lengths, 'gloss_labels': batch_gls_ids}

class Predictor():
    def __init__(self,cfg_path, ckpt_path, device):
        self.device = device
        with open(cfg_path, encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        self.tokenizer=GlossTokenizer_S2G(self.config['gloss'])
        self.model = SLRModel(len(self.tokenizer), cfg=self.config['model'])
        #Ruta al checkpoint guardado
        checkpoint = torch.load(ckpt_path, map_location=device)  # o 'cuda' si estás usando GPU

        # Cargar el estado del modelo
        self.model.load_state_dict(checkpoint['model'])
        
    def eval(self):
        self.model.eval().to(self.device)
    def __call__(self,keypoints):
        t_dim=len(keypoints)
        body_indexes=self.config['model']['RecognitionNetwork']['SlowFast']['body']
        keypoints=torch.tensor(keypoints[:,body_indexes]).permute(2,0,1).unsqueeze(0)
        output = self.model(keypoints.to(self.device))
        for k, gls_logits in output['fuse_outputs'].items():
                ctc_decode_output = self.model.decode(gls_logits,
                                                1,
                                                t_dim)

                batch_pred_gls = self.tokenizer.convert_ids_to_tokens(ctc_decode_output)
        return batch_pred_gls