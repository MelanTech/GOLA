# Modified by Zekai Shao
# Licensed under Apache-2.0: http://www.apache.org/licenses/LICENSE-2.0
# Add support for RGB-T dataset

from typing import Tuple, Mapping, Any

import safetensors
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from trackit.models.backbone.dinov2 import DinoVisionTransformer, interpolate_pos_encoding
from .modules.patch_embed import PatchEmbedNoSizeCheck
from .modules.head.mlp import MlpAnchorFreeHead, Mlp
from .modules.gola.merge import lora_merge_state_dict


class GOLABaseline_DINOv2(nn.Module):
    def __init__(self, vit: DinoVisionTransformer,
                 template_feat_size: Tuple[int, int],
                 search_region_feat_size: Tuple[int, int]):
        super().__init__()
        assert template_feat_size[0] <= search_region_feat_size[0] and template_feat_size[1] <= search_region_feat_size[
            1]
        self.z_size = template_feat_size
        self.d_size = template_feat_size
        self.x_size = search_region_feat_size

        # assert isinstance(vit, DinoVisionTransformer)
        self.patch_embed = PatchEmbedNoSizeCheck.build(vit.patch_embed)
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.embed_dim = vit.embed_dim

        self.pos_embed = nn.Parameter(torch.empty(1, self.x_size[0] * self.x_size[1], self.embed_dim))
        self.pos_embed.data.copy_(interpolate_pos_encoding(vit.pos_embed.data[:, 1:, :],
                                                           self.x_size,
                                                           vit.patch_embed.patches_resolution,
                                                           num_prefix_tokens=0, interpolate_offset=0))

        self.token_type_embed = nn.Parameter(torch.empty(5, self.embed_dim))
        trunc_normal_(self.token_type_embed, std=.02)

        # self.fuse_search = Mlp(self.embed_dim * 2, out_features=self.embed_dim, num_layers=3)

        self.head = MlpAnchorFreeHead(self.embed_dim, self.x_size)

    def forward(self, z: torch.Tensor, x: torch.Tensor, d: torch.Tensor,
                z_feat_mask: torch.Tensor, d_feat_mask: torch.Tensor):
        z_feat_v, z_feat_i = self._z_feat(z, z_feat_mask)
        d_feat_v, d_feat_i = self._d_feat(d, d_feat_mask)
        x_feat_v, x_feat_i = self._x_feat(x)
        x_feat = self._fusion(z_feat_v, x_feat_v, z_feat_i, x_feat_i, d_feat_v, d_feat_i)
        return self.head(x_feat)

    def _z_feat(self, z: torch.Tensor, z_feat_mask: torch.Tensor):
        z_v = self.patch_embed(z[:, :3])
        z_i = self.patch_embed(z[:, 3:])

        z_W, z_H = self.z_size
        z_v = z_v + self.pos_embed.view(1, self.x_size[1], self.x_size[0], self.embed_dim)[:, : z_H, : z_W, :].reshape(
            1, z_H * z_W, self.embed_dim)
        z_v = z_v + self.token_type_embed[:2][z_feat_mask.flatten(1)]

        z_i = z_i + self.pos_embed.view(1, self.x_size[1], self.x_size[0], self.embed_dim)[:, : z_H, : z_W, :].reshape(
            1, z_H * z_W, self.embed_dim)
        z_i = z_i + self.token_type_embed[:2][z_feat_mask.flatten(1)]

        return z_v, z_i

    def _d_feat(self, d: torch.Tensor, d_feat_mask: torch.Tensor):
        d_v = self.patch_embed(d[:, :3])
        d_i = self.patch_embed(d[:, 3:])

        d_W, d_H = self.d_size
        d_v = d_v + self.pos_embed.view(1, self.x_size[1], self.x_size[0], self.embed_dim)[:, : d_H, : d_W, :].reshape(
            1, d_H * d_W, self.embed_dim)
        d_v = d_v + self.token_type_embed[2:4][d_feat_mask.flatten(1)]

        d_i = d_i + self.pos_embed.view(1, self.x_size[1], self.x_size[0], self.embed_dim)[:, : d_H, : d_W, :].reshape(
            1, d_H * d_W, self.embed_dim)
        d_i = d_i + self.token_type_embed[2:4][d_feat_mask.flatten(1)]

        return d_v, d_i

    def _x_feat(self, x: torch.Tensor):
        x_v = self.patch_embed(x[:, :3])
        x_i = self.patch_embed(x[:, 3:])

        x_v = x_v + self.pos_embed
        x_v = x_v + self.token_type_embed[4].view(1, 1, self.embed_dim)
        x_i = x_i + self.pos_embed
        x_i = x_i + self.token_type_embed[4].view(1, 1, self.embed_dim)
        return x_v, x_i

    def _fusion(self, z_feat_v: torch.Tensor, x_feat_v: torch.Tensor, z_feat_i: torch.Tensor, x_feat_i: torch.Tensor,
                d_feat_v: torch.Tensor, d_feat_i: torch.Tensor):
        fusion_feat = torch.cat((z_feat_v, x_feat_v, z_feat_i, x_feat_i, d_feat_v, d_feat_i), dim=1)
        for i in range(len(self.blocks)):
            fusion_feat = self.blocks[i](fusion_feat)
        fusion_feat = self.norm(fusion_feat)
        return self._fuse_search(fusion_feat, z_feat_v.shape[1], x_feat_v.shape[1])

    def _fuse_search(self, feat, z_len, x_len):
        # search_v = feat[:, z_len:z_len + x_len, :]
        search_i = feat[:, 2 * z_len + x_len:2 * (z_len + x_len), :]
        # search = torch.cat([search_v, search_i], dim=2)
        # return self.fuse_search(search)
        return search_i

    def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs):
        state_dict = lora_merge_state_dict(self, state_dict)
        return super().load_state_dict(state_dict, **kwargs)

    def load_state_dict_from_file(self, path: str, **kwargs):
        state_dict = safetensors.torch.load_file(path)
        return self.load_state_dict(state_dict, strict=False)
