from typing import Union, Type, List, Tuple

import torch
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
import torch.nn.functional as F
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from nnunetv2.utilities.unet_decoder import UNetDecoder, SAMDecoder
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from nnunetv2.paths import nnUNet_raw
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from mobile_sam.modeling.tiny_vit_sam import TinyViT
import requests
import os
from typing import Optional

def download_model(url,destination):

    chunk_size = 8192

    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)
        print("Weights downloaded successfully.")
    else:
        print("Failed to download file. Status code:", response.status_code)

class PlainConvUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have "\
                                                  f"resolution stages. here: {n_stages}. "\
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries "\
                                                                f"as we have resolution stages. here: {n_stages} "\
                                                                f"stages, so it should have {n_stages - 1} entries. "\
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or "\
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). "\
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

class SAMConvUNet_3D(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[nn.Module],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[nn.Dropout]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False,
                 ):
        super().__init__()

        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        assert len(n_conv_per_stage) == n_stages
        assert len(n_conv_per_stage_decoder) == (n_stages - 1)

        self.encoder = PlainConvEncoder(
            input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
            n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
            dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True, nonlin_first=nonlin_first
        )
        self.decoder = SAMDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision, nonlin_first=nonlin_first)

        save_path = nnUNet_raw
        model_weight_path = os.path.join(save_path, "mobile_sam.pt")

        if not os.path.exists(model_weight_path):
            download_model(url = 'https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt', destination= model_weight_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        mobile_sam = sam_model_registry["vit_t"](checkpoint=model_weight_path)
        mobile_sam.to(device=device)
        self.sam_image_encoder = mobile_sam.image_encoder
        self.device = device
        self.to(device)

        for param in self.sam_image_encoder.parameters():
            param.requires_grad = False

    def _process_axial_slices(self, x: torch.Tensor, chunk_size=96) -> torch.Tensor:
        B, C, D, H, W = x.shape
        slices = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        if slices.shape[1] == 1:
            slices = slices.repeat(1, 3, 1, 1)
        slices = F.interpolate(slices, size=(1024, 1024), mode='bilinear', align_corners=True)

        embeddings = []
        slices = slices.to(self.device)

        with torch.no_grad():
            sam_embed = self.sam_image_encoder(slices)

        sam_embed = sam_embed.view(B, D, 256, 64, 64).permute(0, 2, 1, 3, 4)

        target_h, target_w = H // 16, W // 16
        sam_embed = F.interpolate(sam_embed, size=(D, target_h, target_w), mode='trilinear', align_corners=True)
        return sam_embed

    def forward(self, x):
        skips = self.encoder(x)
        sam_input = x.detach()
        sam_embed_3d = self._process_axial_slices(sam_input)

        target_shape = skips[3].shape[2:]
        sam_embed_3d = F.interpolate(sam_embed_3d, size=target_shape, mode='trilinear', align_corners=True)

        skips[3] = torch.cat([skips[3], sam_embed_3d.to(skips[3].device)], dim=1)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.conv_op),\
            "Input size should be (x, y, z) for 3D or (x, y) for 2D"
        return (
            self.encoder.compute_conv_feature_map_size(input_size) +
            self.decoder.compute_conv_feature_map_size(input_size)
        )

class SMDNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[nn.Module],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[nn.Dropout]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False,
                 transformer_nhead: int = 8,
                 transformer_layers: int = 4,
                 transformer_ffn_dim: int = 1024,
                 transformer_dim: int = 256,
                 ):
        super().__init__()

        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        assert len(n_conv_per_stage) == n_stages
        assert len(n_conv_per_stage_decoder) == (n_stages - 1)

        self.transformer_dim = transformer_dim
        self.num_classes = num_classes

        self.encoder = PlainConvEncoder(
            input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
            n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
            dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True, nonlin_first=nonlin_first
        )

        self.decoder = SAMDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                  nonlin_first=nonlin_first)

        self._init_sam_encoder()

        self.slice_token_norm = nn.LayerNorm(transformer_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_nhead,
            dim_feedforward=transformer_ffn_dim,
            batch_first=True,
            activation='gelu',
            dropout=0.0,
            norm_first=True,
        )
        self.slice_transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)
        self.slice_cross_attn = nn.MultiheadAttention(
            embed_dim=transformer_dim, num_heads=transformer_nhead, batch_first=True
        )
        self.slice_cross_attn_norm = nn.LayerNorm(transformer_dim)

        bottleneck_ch = self.encoder.output_channels[-1]
        self.bottleneck_q_proj = nn.Conv3d(bottleneck_ch, transformer_dim, kernel_size=1, bias=True)
        self.bottleneck_out_proj = nn.Conv3d(transformer_dim, bottleneck_ch, kernel_size=1, bias=True)
        self.bottleneck_cross_attn = nn.MultiheadAttention(
            embed_dim=transformer_dim, num_heads=transformer_nhead, batch_first=True
        )
        self.bottleneck_norm = nn.LayerNorm(transformer_dim)

        self.coarse_head = nn.Sequential(
            nn.Conv3d(transformer_dim, 128, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm3d(128, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm3d(64, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(64, 2, kernel_size=1, padding=0, bias=True),
        )

        self.prior_proj = nn.Sequential(
            nn.Conv3d(2, transformer_dim, kernel_size=1, padding=0, bias=True),
            nn.LeakyReLU(inplace=True),
        )
        self.vessel_token_proj = nn.Linear(1, transformer_dim)

    def _init_sam_encoder(self):
        save_path = nnUNet_raw
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._sam_device = device
        sam_encoder = os.environ.get('SAM_ENCODER', 'vit_b').lower()

        if sam_encoder in ('vit_t', 'mobile', 'mobilesam'):
            model_weight_path = os.path.join(save_path, "mobile_sam.pt")
            if not os.path.exists(model_weight_path):
                download_model(
                    url='https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt',
                    destination=model_weight_path,
                )
            mobile_sam = sam_model_registry["vit_t"](checkpoint=model_weight_path)
            mobile_sam.to(device=device)
            self.sam_image_encoder = mobile_sam.image_encoder
        else:

            from segment_anything import sam_model_registry as sam_vit_registry
            model_weight_path = os.environ.get(
                'SAM_VIT_B_WEIGHTS',
                os.path.join(save_path, "sam_vit_b_01ec64.pth"),
            )
            if not os.path.exists(model_weight_path):
                raise FileNotFoundError(
                    f"SAM ViT-B weights not found at {model_weight_path}. "
                    f"Place sam_vit_b_01ec64.pth under nnUNet_raw or set SAM_VIT_B_WEIGHTS."
                )
            sam = sam_vit_registry["vit_b"](checkpoint=model_weight_path)
            sam.to(device=device)
            self.sam_image_encoder = sam.image_encoder

        self.sam_image_encoder.eval()
        for p in self.sam_image_encoder.parameters():
            p.requires_grad = False

    def _sam_embed_3d_from_axial_slices(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        slices = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        if slices.shape[1] == 1:
            slices = slices.repeat(1, 3, 1, 1)
        slices = F.interpolate(slices, size=(1024, 1024), mode='bilinear', align_corners=True)
        slices = slices.to(self._sam_device)

        chunk_size = max(1, int(os.environ.get('SAM_CHUNK_SIZE', '2')))
        embeds = []
        with torch.no_grad():
            for i in range(0, slices.shape[0], chunk_size):
                emb = self.sam_image_encoder(slices[i:i + chunk_size])
                embeds.append(emb)
        sam_embed = torch.cat(embeds, dim=0)
        sam_embed = sam_embed.view(B, D, 256, sam_embed.shape[-2], sam_embed.shape[-1]).permute(0, 2, 1, 3, 4)

        target_h, target_w = max(1, H // 16), max(1, W // 16)
        sam_embed = F.interpolate(sam_embed, size=(D, target_h, target_w), mode='trilinear', align_corners=True)
        return sam_embed

    def _cross_slice_branch(self, sam_embed_3d: torch.Tensor):
        B, C, D, h, w = sam_embed_3d.shape
        assert C == self.transformer_dim

        tokens = sam_embed_3d.mean(dim=(-1, -2)).permute(0, 2, 1)
        tokens = self.slice_token_norm(tokens)
        tokens = self.slice_transformer(tokens)

        spatial = sam_embed_3d.permute(0, 2, 3, 4, 1).reshape(B, D * h * w, C)
        attn_out, _ = self.slice_cross_attn(spatial, tokens, tokens, need_weights=False)
        spatial = self.slice_cross_attn_norm(spatial + attn_out)
        enhanced = spatial.view(B, D, h, w, C).permute(0, 4, 1, 2, 3).contiguous()
        return enhanced, tokens

    def _bottleneck_cross_attention(self, bottleneck: torch.Tensor, slice_tokens: torch.Tensor) -> torch.Tensor:
        B, Cb, db, hb, wb = bottleneck.shape
        q = self.bottleneck_q_proj(bottleneck)
        q_tokens = q.flatten(2).transpose(1, 2)
        kv = slice_tokens
        attn_out, _ = self.bottleneck_cross_attn(q_tokens, kv, kv, need_weights=False)
        attn_out = self.bottleneck_norm(q_tokens + attn_out)
        attn_out = attn_out.transpose(1, 2).view(B, self.transformer_dim, db, hb, wb)
        return bottleneck + self.bottleneck_out_proj(attn_out)

    def forward(self, x, return_coarse: bool = False):
        skips = self.encoder(x)

        sam_embed_3d = self._sam_embed_3d_from_axial_slices(x.detach()).to(skips[-1].device)
        sam_embed_3d, slice_tokens = self._cross_slice_branch(sam_embed_3d)

        coarse_logits = self.coarse_head(sam_embed_3d)
        coarse_soft = torch.softmax(coarse_logits, dim=1)
        prior_feat = self.prior_proj(coarse_soft)

        vessel_score = coarse_soft[:, 1].mean(dim=(-1, -2)).unsqueeze(-1)
        vessel_tokens = slice_tokens + self.vessel_token_proj(vessel_score)

        skips[-1] = self._bottleneck_cross_attention(skips[-1], vessel_tokens)

        fusion_skip_idx = 3
        if fusion_skip_idx >= len(skips) - 1:
            fusion_skip_idx = max(0, len(skips) - 2)
        target_shape = skips[fusion_skip_idx].shape[2:]
        prior_feat = F.interpolate(prior_feat, size=target_shape, mode='trilinear', align_corners=True)
        skips[fusion_skip_idx] = torch.cat([skips[fusion_skip_idx], prior_feat], dim=1)

        fine = self.decoder(skips)
        if return_coarse:
            return fine, coarse_logits
        return fine

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op),\
            "just give the image size without color/feature channels or batch channel."
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

class SAMConvUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have "\
                                                  f"resolution stages. here: {n_stages}. "\
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries "\
                                                                f"as we have resolution stages. here: {n_stages} "\
                                                                f"stages, so it should have {n_stages - 1} entries. "\
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        self.decoder = SAMDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)

        save_path = nnUNet_raw
        model_weight_path = os.path.join(save_path, "mobile_sam.pt")

        if not os.path.exists(model_weight_path):
            download_model(url = 'https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt', destination= model_weight_path)

        model_type = "vit_t"

        device = "cuda" if torch.cuda.is_available() else "cpu"

        mobile_sam = sam_model_registry[model_type](checkpoint=model_weight_path)
        mobile_sam.to(device=device)

        self.sam_image_encoder = mobile_sam.image_encoder

        for param in self.sam_image_encoder.parameters():
            param.requires_grad = False

    def forward(self, x):

        sam_input = x.detach()
        if sam_input.shape[1] == 1:
            sam_input = sam_input.repeat(1, 3, 1, 1)

        sam_input = F.interpolate(sam_input, size=(1024, 1024), mode='bilinear', align_corners=True)

        sam_embed = self.sam_image_encoder(sam_input)

        skips = self.encoder(x)

        sam_embed = F.interpolate(sam_embed, size=(skips[3].shape[2], skips[3].shape[3]), mode='bilinear', align_corners=True)
        skips[3] = torch.cat((skips[3], sam_embed), dim=1)

        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or "\
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). "\
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

class ResidualEncoderUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None
                 ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have "\
                                                  f"resolution stages. here: {n_stages}. "\
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries "\
                                                                f"as we have resolution stages. here: {n_stages} "\
                                                                f"stages, so it should have {n_stages - 1} entries. "\
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or "\
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). "\
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

if __name__ == '__main__':
    data = torch.rand((1, 4, 128, 128, 128))

    model = PlainConvUNet(4, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4,
                                (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True)

    if False:
        import hiddenlayer as hl

        g = hl.build_graph(model, data,
                           transforms=None)
        g.save("network_architecture.pdf")
        del g

    print(model.compute_conv_feature_map_size(data.shape[2:]))

    data = torch.rand((1, 4, 512, 512))

    model = PlainConvUNet(4, 8, (32, 64, 125, 256, 512, 512, 512, 512), nn.Conv2d, 3, (1, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2, 2, 2), 4,
                                (2, 2, 2, 2, 2, 2, 2), False, nn.BatchNorm2d, None, None, None, nn.ReLU, deep_supervision=True)

    if False:
        import hiddenlayer as hl

        g = hl.build_graph(model, data,
                           transforms=None)
        g.save("network_architecture.pdf")
        del g

    print(model.compute_conv_feature_map_size(data.shape[2:]))
