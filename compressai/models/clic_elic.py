import math
import torch
import torch.nn as nn

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from compressai.models.utils import conv, deconv, update_registered_buffers
from compressai.ops import ste_round
from compressai.layers import conv3x3, subpel_conv3x3,conv1x1,CoCs_BasicLayer,MergingProj,UnshuffleProj
from compressai.models.base import CompressionModel

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
class CheckboardMaskedConv2d(nn.Conv2d):
    """
    if kernel_size == (5, 5)
    then mask:
        [[0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.]]
    0: non-anchor
    1: anchor
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        self.mask[:, :, 0::2, 1::2] = 1
        self.mask[:, :, 1::2, 0::2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        out = super().forward(x)

        return out

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))
class Quantizer():
    def quantize(self, inputs, quantize_type="noise"):
        if quantize_type == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs
        elif quantize_type == "ste":
            return torch.round(inputs) - inputs.detach() + inputs
        else:
            return torch.round(inputs)

class CLIC_base(CompressionModel):
    """clic base model"""

    def __init__(self, N=192, M=320,num_slices = 5, **kwargs):
        super().__init__(**kwargs)
        # self.max_support_slices = self.num_slices // 2
        self.N = int(N)
        self.M = int(M)
        self.groups = [0, 16, 16, 32, 64, 192]  # support depth
        self.g_a = nn.Sequential(
            conv(5, N//2, kernel_size=5, stride=2),
            CoCs_BasicLayer(N//2,2,fold_w=1,fold_h=1,heads = 6,head_dim=32),
            MergingProj(N//2,N),
            CoCs_BasicLayer(N, 2, fold_w=1, fold_h=1, heads=8, head_dim=32),
            MergingProj(N, 3*N//2),
            CoCs_BasicLayer(3*N//2, 3, fold_w=1, fold_h=1, heads=10, head_dim=32),
            MergingProj(3*N//2,M),
            CoCs_BasicLayer(M, 3, fold_w=1, fold_h=1, heads=12, head_dim=32),
        )
        self.g_s = nn.Sequential(
            CoCs_BasicLayer(M, 3, fold_w=1, fold_h=1, heads=12, head_dim=32),
            UnshuffleProj(M,3*N//2),
            CoCs_BasicLayer(3*N//2, 3, fold_w=1, fold_h=1, heads=10, head_dim=32),
            UnshuffleProj(3*N//2,N),
            CoCs_BasicLayer(N, 2, fold_w=1, fold_h=1, heads=8, head_dim=32),
            UnshuffleProj( N,N//2),
            CoCs_BasicLayer(N//2, 2, fold_w=1, fold_h=1, heads=6, head_dim=32),
            deconv(N//2, 3, kernel_size=5, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(320, 320),
            nn.GELU(),
            conv3x3(320, 288),
            nn.GELU(),
            conv3x3(288, 256, stride=2),
            nn.GELU(),
            conv3x3(256, 224),
            nn.GELU(),
            conv3x3(224, 192, stride=2),
        )

        self.h_mean_s = nn.Sequential(
            conv3x3(192, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, 320),
        )

        self.h_scale_s = nn.Sequential(
            conv3x3(192, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, 320),
        )


        self.cc_transforms = nn.ModuleList(
            nn.Sequential(
                conv(self.groups[min(1, i) if i > 0 else 0] + self.groups[i if i > 1 else 0], 224, stride=1,
                     kernel_size=5),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, self.groups[i + 1]*2, stride=1, kernel_size=3),
            ) for i in range(1,  num_slices)
        )

        self.context_prediction = nn.ModuleList(
            CheckboardMaskedConv2d(
                self.groups[i + 1], 2 * self.groups[i + 1], kernel_size=5, padding=2, stride=1
            ) for i in range(num_slices)
        )  ##

        self.ParamAggregation = nn.ModuleList(
            nn.Sequential(
                conv1x1(640 + self.groups[i+1 if i > 0 else 0] * 2 + self.groups[
                        i + 1] * 2, 640),
                nn.GELU(),
                conv1x1(640, 512),
                nn.GELU(),
                conv1x1(512, self.groups[i + 1]*2),
            ) for i in range(num_slices))



        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)
        self.quantizer = Quantizer()



    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated


    def forward(self, x,noisequant = False):
        _,_,img_h,img_w = x.shape
        range_w = torch.arange(0, img_w, step=1) / (img_w - 1.0)
        range_h = torch.arange(0, img_h, step=1) / (img_h - 1.0)
        fea_pos = torch.stack(torch.meshgrid(range_w, range_h, indexing='ij'), dim=-1).float()
        fea_pos = fea_pos - 0.5
        pos = fea_pos.permute(2, 0, 1).unsqueeze(dim=0).expand(x.shape[0], -1, -1, -1)#.cuda()
        x = torch.cat((x,pos),dim=1)
        y = self.g_a(x)
        B, C, H, W = y.size() ## The shape of y to generate the mask

        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        if not noisequant:
            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        anchor = torch.zeros_like(y).to(x.device)
        non_anchor = torch.zeros_like(y).to(x.device)

        anchor[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
        anchor[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]
        non_anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        non_anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]


        y_slices = torch.split(y, self.groups[1:], 1)
        anchor_split = torch.split(anchor, self.groups[1:], 1)
        non_anchor_split = torch.split(non_anchor, self.groups[1:], 1)
        ctx_params_anchor_split = torch.split(torch.zeros(B, C * 2, H, W).to(x.device),
                                              [2 * i for i in self.groups[1:]], 1)

        y_hat_slices = []
        y_hat_slices_for_gs = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            else:
                support_slices = torch.concat([y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1)
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)

            support = torch.concat([latent_means, latent_scales], dim=1) if slice_index == 0 else torch.concat(
                [support_slices_ch_mean, support_slices_ch_scale, latent_means, latent_scales], dim=1)

            ### checkboard process 1
            y_anchor = anchor_split[slice_index]
            means_anchor, scales_anchor, = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)

            scales_hat_split = torch.zeros_like(y_anchor).to(x.device)
            means_hat_split = torch.zeros_like(y_anchor).to(x.device)

            scales_hat_split[:, :, 0::2, 0::2] = scales_anchor[:, :, 0::2, 0::2]
            scales_hat_split[:, :, 1::2, 1::2] = scales_anchor[:, :, 1::2, 1::2]
            means_hat_split[:, :, 0::2, 0::2] = means_anchor[:, :, 0::2, 0::2]
            means_hat_split[:, :, 1::2, 1::2] = means_anchor[:, :, 1::2, 1::2]
            if noisequant:
                y_anchor_quantilized = self.quantizer.quantize(y_anchor, "noise")
                y_anchor_quantilized_for_gs = self.quantizer.quantize(y_anchor , "ste")
            else:
                y_anchor_quantilized = self.quantizer.quantize(y_anchor - means_anchor, "ste") + means_anchor
                y_anchor_quantilized_for_gs = self.quantizer.quantize(y_anchor - means_anchor, "ste") + means_anchor
            y_anchor_quantilized[:, :, 0::2, 1::2] = 0
            y_anchor_quantilized[:, :, 1::2, 0::2] = 0
            y_anchor_quantilized_for_gs[:, :, 0::2, 1::2] = 0
            y_anchor_quantilized_for_gs[:, :, 1::2, 0::2] = 0
            ### checkboard process 2
            masked_context = self.context_prediction[slice_index](y_anchor_quantilized)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)).chunk(2, 1)

            scales_hat_split[:, :, 0::2, 1::2] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_hat_split[:, :, 1::2, 0::2] = scales_non_anchor[:, :, 1::2, 0::2]
            means_hat_split[:, :, 0::2, 1::2] = means_non_anchor[:, :, 0::2, 1::2]
            means_hat_split[:, :, 1::2, 0::2] = means_non_anchor[:, :, 1::2, 0::2]
            # entropy estimation
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scales_hat_split, means=means_hat_split)

            y_non_anchor = non_anchor_split[slice_index]

            if noisequant:
                y_non_anchor_quantilized = self.quantizer.quantize(y_non_anchor, "noise")
                y_non_anchor_quantilized_for_gs = self.quantizer.quantize(y_non_anchor, "ste")
            else:
                y_non_anchor_quantilized = self.quantizer.quantize(y_non_anchor - means_non_anchor,
                                                                   "ste") + means_non_anchor
                y_non_anchor_quantilized_for_gs = self.quantizer.quantize(y_non_anchor - means_non_anchor,
                                                                          "ste") + means_non_anchor

            y_non_anchor_quantilized[:, :, 0::2, 0::2] = 0
            y_non_anchor_quantilized[:, :, 1::2, 1::2] = 0
            y_non_anchor_quantilized_for_gs[:, :, 0::2, 0::2] = 0
            y_non_anchor_quantilized_for_gs[:, :, 1::2, 1::2] = 0

            y_hat_slice = y_anchor_quantilized + y_non_anchor_quantilized
            y_hat_slice_for_gs = y_anchor_quantilized_for_gs + y_non_anchor_quantilized_for_gs
            y_hat_slices.append(y_hat_slice)
            ### ste for synthesis model
            y_hat_slices_for_gs.append(y_hat_slice_for_gs)
            y_likelihood.append(y_slice_likelihood)

        y_likelihoods = torch.cat(y_likelihood, dim=1)
        y_hat = torch.cat(y_hat_slices_for_gs, dim=1)
        x_hat = self.g_s(y_hat)


        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        # N = state_dict["g_a.0.weight"].size(0)
        # M = state_dict["g_a.6.weight"].size(0)
        # net = cls(N, M)
        net = cls(192, 320)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        _, _, img_w,img_h  = x.shape
        range_w = torch.arange(0, img_w, step=1) / (img_w - 1.0)
        range_h = torch.arange(0, img_h, step=1) / (img_h - 1.0)
        fea_pos = torch.stack(torch.meshgrid(range_w, range_h, indexing='ij'), dim=-1).float()
        fea_pos = fea_pos - 0.5
        pos = fea_pos.permute(2, 0, 1).unsqueeze(dim=0).expand(x.shape[0], -1, -1, -1).cuda()
        x = torch.cat((x, pos), dim=1)
        y = self.g_a(x)
        B, C, H, W = y.size()  ## The shape of y to generate the mask
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)
        y_slices = torch.split(y, self.groups[1:], 1)
        ctx_params_anchor_split = torch.split(torch.zeros(B, C * 2, H, W).to(x.device), [2 * i for i in self.groups[1:]], 1)
        y_strings = []
        y_hat_slices = []
        for slice_index, y_slice in enumerate(y_slices):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)

            else:
                support_slices = torch.concat([y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1)
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            ##support mean and scale
            support = torch.concat([latent_means, latent_scales], dim=1) if slice_index == 0 else torch.concat(
                [support_slices_ch_mean, support_slices_ch_scale, latent_means, latent_scales], dim=1)
            ### checkboard process 1
            y_anchor = y_slices[slice_index].clone()
            means_anchor, scales_anchor, = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)

            B_anchor, C_anchor, H_anchor, W_anchor = y_anchor.size()

            y_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
            means_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
            scales_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
            y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(x.device)

            y_anchor_encode[:, :, 0::2, :] = y_anchor[:, :, 0::2, 0::2]
            y_anchor_encode[:, :, 1::2, :] = y_anchor[:, :, 1::2, 1::2]
            means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]

            indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor_encode)
            anchor_strings = self.gaussian_conditional.compress(y_anchor_encode, indexes_anchor,
                                                                means=means_anchor_encode)
            anchor_quantized = self.gaussian_conditional.decompress(anchor_strings, indexes_anchor,
                                                                    means=means_anchor_encode)
            y_anchor_decode[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]

            ### checkboard process 2
            masked_context = self.context_prediction[slice_index](y_anchor_decode)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)).chunk(2, 1)

            y_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
            means_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
            scales_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)

            non_anchor = y_slices[slice_index].clone()
            y_non_anchor_encode[:, :, 0::2, :] = non_anchor[:, :, 0::2, 1::2]
            y_non_anchor_encode[:, :, 1::2, :] = non_anchor[:, :, 1::2, 0::2]
            means_non_anchor_encode[:, :, 0::2, :] = means_non_anchor[:, :, 0::2, 1::2]
            means_non_anchor_encode[:, :, 1::2, :] = means_non_anchor[:, :, 1::2, 0::2]
            scales_non_anchor_encode[:, :, 0::2, :] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_non_anchor_encode[:, :, 1::2, :] = scales_non_anchor[:, :, 1::2, 0::2]

            indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_non_anchor_encode)
            non_anchor_strings = self.gaussian_conditional.compress(y_non_anchor_encode, indexes_non_anchor,
                                                                    means=means_non_anchor_encode)

            non_anchor_quantized = self.gaussian_conditional.decompress(non_anchor_strings, indexes_non_anchor,
                                                                        means=means_non_anchor_encode)

            y_non_anchor_quantized = torch.zeros_like(means_anchor)
            y_non_anchor_quantized[:, :, 0::2, 1::2] = non_anchor_quantized[:, :, 0::2, :]
            y_non_anchor_quantized[:, :, 1::2, 0::2] = non_anchor_quantized[:, :, 1::2, :]

            y_slice_hat = y_anchor_decode + y_non_anchor_quantized
            y_hat_slices.append(y_slice_hat)

            y_strings.append([anchor_strings, non_anchor_strings])

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = torch.max(scales, torch.tensor(0.11))
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        B, _, _, _ = z_hat.size()
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)
        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        y_strings = strings[0]
        ctx_params_anchor = torch.zeros((B, self.M * 2, z_hat.shape[2] * 4, z_hat.shape[3] * 4)).to(z_hat.device)
        ctx_params_anchor_split = torch.split(ctx_params_anchor, [2 * i for i in self.groups[1:]], 1)
        y_hat_slices = []
        for slice_index in range(len(self.groups) - 1):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)

            else:
                support_slices = torch.concat([y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1)
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            ##support mean and scale
            support = torch.concat([latent_means, latent_scales], dim=1) if slice_index == 0 else torch.concat(
                [support_slices_ch_mean, support_slices_ch_scale, latent_means, latent_scales], dim=1)
            ### checkboard process 1
            means_anchor, scales_anchor, = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)

            B_anchor, C_anchor, H_anchor, W_anchor = means_anchor.size()

            means_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(z_hat.device)
            scales_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(z_hat.device)
            y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(z_hat.device)

            means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]

            indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor_encode)
            anchor_strings = y_strings[slice_index][0]
            anchor_quantized = self.gaussian_conditional.decompress(anchor_strings, indexes_anchor,
                                                                    means=means_anchor_encode)

            y_anchor_decode[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]

            ### checkboard process 2
            masked_context = self.context_prediction[slice_index](y_anchor_decode)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)).chunk(2, 1)

            means_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(z_hat.device)
            scales_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(z_hat.device)

            means_non_anchor_encode[:, :, 0::2, :] = means_non_anchor[:, :, 0::2, 1::2]
            means_non_anchor_encode[:, :, 1::2, :] = means_non_anchor[:, :, 1::2, 0::2]
            scales_non_anchor_encode[:, :, 0::2, :] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_non_anchor_encode[:, :, 1::2, :] = scales_non_anchor[:, :, 1::2, 0::2]

            indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_non_anchor_encode)
            non_anchor_strings = y_strings[slice_index][1]
            non_anchor_quantized = self.gaussian_conditional.decompress(non_anchor_strings, indexes_non_anchor,
                                                                        means=means_non_anchor_encode)

            y_non_anchor_quantized = torch.zeros_like(means_anchor)
            y_non_anchor_quantized[:, :, 0::2, 1::2] = non_anchor_quantized[:, :, 0::2, :]
            y_non_anchor_quantized[:, :, 1::2, 0::2] = non_anchor_quantized[:, :, 1::2, :]

            y_slice_hat = y_anchor_decode + y_non_anchor_quantized
            y_hat_slices.append(y_slice_hat)
        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)


        return {"x_hat": x_hat}

if __name__ == '__main__':
    model = CLIC_base()
    # print(model)
    print("params:",sum(param.numel() for param in model.parameters()))

    x = torch.randn((1, 3, 256, 256))
    x = model(x)
    print(x["x_hat"].shape)




