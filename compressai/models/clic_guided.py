import math
import torch
import torch.nn as nn

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
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
    def clip_by_tensor(self,t, t_min, t_max):
        """
        clip_by_tensor
        :param t: tensor
        :param t_min: min
        :param t_max: max
        :return: cliped tensor
        """
        t = t.float()
        t_min = t_min.float()
        t_max = t_max.float()

        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result
    def quantize(self, inputs, quantize_type="noise",**params):
        if quantize_type == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs
        elif quantize_type == "ste":
            return torch.round(inputs) - inputs.detach() + inputs
        elif quantize_type == "Identity":
            return inputs
        elif quantize_type == "U-Q":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)[
                    :, 0:1, 0:1, 0:1
                    ]
            outputs = torch.round(inputs + noise) - noise
            outputs = outputs-inputs.detach()+inputs
            return outputs
        elif quantize_type =="sigmoid":
            T = params["T"]
            diff = inputs - torch.floor(inputs)
            temp = diff * T
            temp = self.clip_by_tensor(temp, -10.0, 10.0)
            outputs = torch.sigmoid(temp) + torch.floor(inputs)
            return outputs
        elif quantize_type =="DS-Q":
            # k = params["k"]
            k = torch.Tensor([0.1])
            y_floor = torch.floor(inputs)
            diff = inputs-y_floor
            phi = torch.tanh((diff-0.5)*k)/torch.tanh(0.5*k)
            y_phi = (1+phi)/2+y_floor
            outputs = torch.round(inputs)
            outputs = outputs-y_phi.detach()+y_phi
            return outputs
        elif quantize_type in {"ST-Q","SRA-Q"}:
            diff = inputs-torch.floor(inputs)
            if quantize_type =="ST-Q":
                prob = diff
            else:
                tau = params["tau"]
                likelihood_up = torch.exp(-torch.atanh(diff)/tau)
                likelihood_down = torch.exp(-torch.atanh(1-diff)/tau)
                prob = likelihood_down/(likelihood_up+likelihood_down)

            prob = prob.type(torch.float32)
            delta = torch.where(prob>torch.empty_like(prob).uniform_(),torch.tensor([1.]),torch.tensor([0.]))
            outputs = torch.floor(inputs)+delta
            outputs = outputs-inputs.detach()+inputs
            return outputs

        elif quantize_type == "SGA-Q":
            tau = params["tau2"]
            epsilon = 1e-5
            y_floor = torch.floor(inputs)
            y_ceil = torch.ceil(inputs)
            y_bds = torch.stack([y_floor,y_ceil],dim = -1)
            ry_logits = torch.stack(
                [
                    -torch.atanh(
                        self.clip_by_tensor(inputs-y_floor,-1+epsilon,1-epsilon)
                    )/tau,
                    -torch.atanh(
                        self.clip_by_tensor(y_ceil-inputs,-1+epsilon,1-epsilon)
                    )/tau,
                ],
                dim=-1
            )
            ry_dist =  torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(tau, logits=ry_logits)
            ry_sample = ry_dist.sample()
            outputs = torch.sum(y_bds*ry_sample,dim=-1)
            return outputs




            # binary_test = tf.cast(binary > 0.5, tf.float32)
            # binary_test = binary.type(torch.float32)
            # binary_test = torch.where(binary_test > 0.5, torch.tensor([1.]), torch.tensor([0.]))





        else:
            return torch.round(inputs)

def quantize(inputs, input_1, scale=100):
    return (torch.round(inputs * scale) / scale + input_1) / 2
class CLIC(CompressionModel):

    def __init__(self, N=192, M=320,num_slices = 5,pqf = True,A = 2, **kwargs):
        super().__init__(**kwargs)
        # self.max_support_slices = self.num_slices // 2
        self.N = int(N)
        self.M = int(M)
        self.pqf = pqf
        self.A = A
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
            CoCs_BasicLayer(M*2, 3, fold_w=1, fold_h=1, heads=12, head_dim=32),
            UnshuffleProj(M*2,3*N//2),
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

        self.h_latent_s = nn.Sequential(
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


        self.context_prediction = nn.ModuleList(nn.Sequential(
            CheckboardMaskedConv2d(
                self.groups[i + 1], int(1.5 * self.groups[i + 1]), kernel_size=5, padding=2, stride=1
            ),
            nn.GELU(),
            CheckboardMaskedConv2d(
                int(1.5 * self.groups[i + 1]), int(1.5 * self.groups[i + 1]), kernel_size=5, padding=2, stride=1
            ),
            nn.GELU(),
            CheckboardMaskedConv2d(
                int(1.5 * self.groups[i + 1]), 2 * self.groups[i + 1], kernel_size=5, padding=2, stride=1
            ),
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
        if self.pqf:
            self.Guided = nn.ModuleList(
                nn.Sequential(
                    conv3x3(self.groups[i + 1], self.groups[i + 1] * 3),
                    nn.GELU(),
                    conv3x3(self.groups[i + 1] * 3, self.groups[i + 1] * 3),
                    nn.GELU(),
                    conv3x3(self.groups[i + 1] * 3, int(self.groups[i + 1] * 2.4)),
                    nn.GELU(),
                    conv3x3(int(self.groups[i + 1] * 2.4), self.groups[i + 1]*self.A),
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


    def forward(self, x, noisequant=False):
        loss_for_guided = 0
        _, _, img_h, img_w = x.shape
        range_w = torch.arange(0, img_w, step=1) / (img_w - 1.0)
        range_h = torch.arange(0, img_h, step=1) / (img_h - 1.0)
        fea_pos = torch.stack(torch.meshgrid(range_w, range_h, indexing='ij'), dim=-1).float()
        fea_pos = fea_pos - 0.5
        pos = fea_pos.permute(2, 0, 1).unsqueeze(dim=0).expand(x.shape[0], -1, -1, -1)#.cuda()
        x = torch.cat((x, pos), dim=1)
        y = self.g_a(x)
        B, C, H, W = y.size()  ## The shape of y to generate the mask

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

        latent_feature = self.h_latent_s(z_hat)

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
                y_anchor_quantilized_for_gs = self.quantizer.quantize(y_anchor, "ste")
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
            ### ste for synthesis model and not update
            if self.pqf:
                # batch_implementation
                B, C, H, W = y_hat_slice_for_gs.shape
                r_out = self.Guided[slice_index](y_hat_slice_for_gs).reshape(B*C,self.A,H,W).contiguous().permute(0, 2, 3, 1).flatten(1,2)
                y_hat_slice_for_gs_ = y_hat_slice_for_gs.reshape(B*C,1,H,W).contiguous()
                y_slice_ = y_slice.reshape(B*C,1,H,W).contiguous()
                b = torch.subtract(y_slice_, y_hat_slice_for_gs_)
                b = b.permute(0, 2, 3, 1).flatten(1,2)
                A = torch.matmul(
                    torch.matmul(torch.inverse(torch.matmul(r_out.permute(0, 2, 1), r_out)), r_out.permute(0, 2, 1)),
                    b)
                b = b.permute(0, 2, 1)
                loss_for_guided += torch.sum(-(torch.matmul(torch.matmul(b, r_out), A)))
                y_hat_slice_for_gs =(torch.sum(A.unsqueeze(-1)*r_out.permute(0,2,1).unflatten(2,[H,W]),dim = 1,keepdim=True)+y_slice_).reshape(B,C,H,W).contiguous()
            y_hat_slices_for_gs.append(y_hat_slice_for_gs)
            y_likelihood.append(y_slice_likelihood)

        y_likelihoods = torch.cat(y_likelihood, dim=1)
        y_hat = torch.cat(y_hat_slices_for_gs, dim=1)

        y_hat = torch.cat((y_hat, latent_feature), dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            'loss_for_guided': loss_for_guided
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
        net = cls(192, 320)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        string_lenth = 0
        _, _, img_w,img_h  = x.shape
        range_w = torch.arange(0, img_w, step=1) / (img_w - 1.0)
        range_h = torch.arange(0, img_h, step=1) / (img_h - 1.0)
        fea_pos = torch.stack(torch.meshgrid(range_w, range_h, indexing='ij'), dim=-1).float()
        fea_pos = fea_pos - 0.5
        pos = fea_pos.permute(2, 0, 1).unsqueeze(dim=0).expand(x.shape[0], -1, -1, -1)#.cuda()
        x = torch.cat((x, pos), dim=1)
        y = self.g_a(x)
        B, C, H, W = y.size()  ## The shape of y to generate the mask
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        string_lenth = string_lenth + len(z_strings[0])

        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = torch.split(y, self.groups[1:], 1)
        ctx_params_anchor_split = torch.split(torch.zeros(B, C * 2, H, W).to(x.device), [2 * i for i in self.groups[1:]], 1)
        y_strings = []
        y_hat_slices = []
        A_list = []
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
            string_lenth =string_lenth+len(anchor_strings[0])
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
            string_lenth =string_lenth+len(non_anchor_strings[0])



            non_anchor_quantized = self.gaussian_conditional.decompress(non_anchor_strings, indexes_non_anchor,
                                                                        means=means_non_anchor_encode)

            y_non_anchor_quantized = torch.zeros_like(means_anchor)
            y_non_anchor_quantized[:, :, 0::2, 1::2] = non_anchor_quantized[:, :, 0::2, :]
            y_non_anchor_quantized[:, :, 1::2, 0::2] = non_anchor_quantized[:, :, 1::2, :]

            y_slice_hat = y_anchor_decode + y_non_anchor_quantized
            if self.pqf:
                B, C, H, W = y_slice_hat.shape
                R = self.Guided[slice_index](y_slice_hat).reshape(B * C,self.A, H,W).permute(0,2, 3, 1).flatten(1,2).contiguous()
                r = y_slice - y_slice_hat
                r = r.reshape(B * C,-1,  H,W).permute(0,2, 3, 1).flatten(1,2).contiguous()
                R_transpose = torch.transpose(R, -1, -2)
                A = torch.inverse(R_transpose.matmul(R)).matmul(R_transpose).matmul(r).unsqueeze(-1)
                A_list.append(A)
                #A nead to be transmitted
            y_hat_slices.append(y_slice_hat)

            y_strings.append([anchor_strings, non_anchor_strings])

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:],"A":A_list}

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

    def decompress(self, strings, shape,A):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        B, _, _, _ = z_hat.size()
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)
        latent_feature = self.h_latent_s(z_hat)
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

        if self.pqf:
            #whenever use A, please calculate the rate
            for index,y_slice_hat in enumerate(y_hat_slices):
                B, C, H, W = y_slice_hat.shape
                R = self.Guided[index](y_slice_hat).reshape(B * C, self.A, H, W)
                y_hat_slices[index] = (torch.sum(A[index] * R, dim=1, keepdim=True)).reshape(B, C, H, W).contiguous() + y_slice_hat
        y_hat = torch.cat(y_hat_slices, dim=1)

        y_hat = torch.cat((y_hat, latent_feature), dim=1)

        x_hat = self.g_s(y_hat).clamp_(0, 1)


        return {"x_hat": x_hat}

if __name__ == '__main__':
    model = CLIC()
    # print(model)
    print("params:",sum(param.numel() for param in model.parameters()))
    x = torch.randn((1, 3, 64, 64))
    x = model(x)
    print(x["x_hat"].shape)
    x = torch.randn((1, 3, 64, 64))
    model.update()
    x = model.compress(x)
    out = model.decompress(x["strings"], x["shape"],x["A"])




