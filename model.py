import torch
import torch.nn as nn

from utils import to_2tuple


class OverlapPatchEmbeddings(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        padding = patch_size[0] // 2, patch_size[1] // 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class DropPath(nn.Module):
    """
    Stoch. Depth, a regularization method used in training residual networks, drops entire residual blocks during training.
    Paper: https://arxiv.org/pdf/1603.09382
    """

    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        drop_prob = self.drop_prob
        if drop_prob == 0.0 or not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random = torch.rand(shape, dtype=x.dtype, device=x.device)
        mask = (random > drop_prob).float()
        out = x * mask / (1.0 - drop_prob)
        return out


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        qkv_scale=None,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert dim % num_heads == 0, (
            f"dim {dim} should be divisible by num_heads {num_heads}"
        )
        self.scale = qkv_scale if qkv_scale is not None else head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )  # B, num_heads, tokens, head_dim

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)  # B, head_dim, height, width
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)  # B, (H//sr)*(W//sr), C
            x_ = self.norm(x_)
            kv = (
                self.kv(x_)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )  # 2, B, num_heads, tokens, head_dim
        else:
            kv = (
                self.kv(x)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )  # 2, B, num_heads, tokens, head_dim
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn * v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        act_layer=nn.GELU,
        drop_rate=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio,
        qkv_bias=True,
        qkv_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads,
            qkv_bias=qkv_bias,
            qkv_scale=qkv_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            sr_ratio=sr_ratio,
        )

        self.drop_path = DropPath(drop_path_rate)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            dim, dim * mlp_ratio, dim, act_layer=act_layer, drop_rate=drop_rate
        )

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class MixTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chals=3,
        num_classes=1000,
        embed_dims=[32, 64, 160, 256],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        qkv_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        inp_reductions = [1, 4, 8, 16]
        assert len(inp_reductions) == len(depths)
        assert len(num_heads) == len(depths)
        assert len(sr_ratios) == len(depths)
        assert len(embed_dims) == len(depths)

        patch_embeds, blocks, norms = [], [], []
        dims = [in_chals] + embed_dims
        depth_drop_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        cur = 0
        for l_i in range(len(depths)):
            patch_embeds.append(
                OverlapPatchEmbeddings(
                    img_size=img_size // inp_reductions[l_i],
                    patch_size=7 if l_i == 0 else 3,
                    stride=4 if l_i == 0 else 2,
                    in_chans=dims[0],
                    embed_dim=dims[1],
                )
            )
            blocks.append(
                nn.ModuleList(
                    [
                        Block(
                            dim=embed_dims[l_i],
                            num_heads=num_heads[l_i],
                            mlp_ratio=mlp_ratios[l_i],
                            qkv_bias=qkv_bias,
                            qkv_scale=qkv_scale,
                            drop_rate=drop_rate,
                            attn_drop_rate=attn_drop_rate,
                            drop_path_rate=depth_drop_rates[cur + b_i],
                            norm_layer=norm_layer,
                            sr_ratio=sr_ratios[l_i],
                        )
                        for b_i in range(depths[l_i])
                    ]
                )
            )
            cur += depths[l_i]
            norms.append(norm_layer(embed_dims[l_i]))

        self.patch_embeds = patch_embeds
        self.blocks = blocks
        self.norms = norms

        if self.num_classes > 0:
            self.head = nn.Linear(embed_dims[3], num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out = fan_out / m.groups
            nn.init.normal_(m.weight, mean=0, std=2 / (fan_out**0.5))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for l_i in range(len(self.blocks)):
            x, H, W = self.patch_embeds[l_i](x)
            for blk in self.blocks[l_i]:
                x = blk(x, H, W)
            x = self.norms[l_i](x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # B, C, H, W
            outs.append(x)

        if self.num_classes > 0:
            return self.head(outs[-1])

        return outs


if __name__ == "__main__":
    model = MixTransformer(num_classes=0)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    for x in output:
        print(x.shape)
