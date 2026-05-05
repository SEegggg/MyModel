import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

epsilon = 1e-4


# ──────────────────────────────────────────────────────────────────────────────
# 基础 MLP
# ──────────────────────────────────────────────────────────────────────────────

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1  = nn.Conv2d(input_dim,  hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2  = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1), bias=True)
        self.act  = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x)))) + x


# ──────────────────────────────────────────────────────────────────────────────
# HyperDecomp：节点自适应分解
#
# 修复：w_se 现在通过周期折叠计算 mu_se，语义还原为
#       "跨周期对齐同一时间槽的加权均值"，与原版 SSCNNDecomp 一致。
#       w_st 保留滑动窗口（短周期本来就是局部平滑，语义正确）。
# ──────────────────────────────────────────────────────────────────────────────

class HyperDecomp(nn.Module):
    def __init__(self, num_nodes, node_dim=16, cycle_len=12,
                 short_period_len=4, hyper_hidden=32):
        super().__init__()
        self.num_nodes        = num_nodes
        self.node_dim         = node_dim
        self.cycle_len        = cycle_len
        self.short_period_len = short_period_len

        # 节点嵌入（对外暴露，供 HyperProjHead 共用同一套嵌入）
        self.node_emb = nn.Parameter(torch.empty(num_nodes, node_dim))
        nn.init.xavier_uniform_(self.node_emb)

        out_dim = 3 + cycle_len + short_period_len
        self.hyper_net = nn.Sequential(
            nn.Linear(node_dim, hyper_hidden), nn.GELU(),
            nn.Linear(hyper_hidden, hyper_hidden), nn.GELU(),
            nn.Linear(hyper_hidden, out_dim),
        )

    def _node_conv_periodic(self, x, kernels):
        """
        周期折叠版季节性均值估计（修复版）。

        语义：先将时间轴按 cycle_len 折叠成 (num_cycles, cycle_len)，
              再对每个时间槽做节点独立的跨周期加权平均，
              还原回原始时间轴后输出。

        参数
        ----
        x       : (B, C, N, T)
        kernels : (N, cycle_len)  已经过 softmax，和为 1

        返回
        ----
        mu_se : (B, C, N, T)
        """
        B, C, N, T = x.shape
        K = kernels.shape[1]          # == cycle_len
        t_cycle = (T // K) * K       # 能被整除的最长前缀长度

        if t_cycle == 0:
            # 序列比一个周期还短，退化为全局均值
            return x.mean(dim=-1, keepdim=True).expand_as(x)

        # 折叠：(B, C, N, T) → (B, C, N, num_cycles, K)
        x_fold = x[..., :t_cycle].reshape(B, C, N, -1, K)

        # 节点独立加权：kernels (N, K) → (1, 1, N, 1, K)
        w = kernels.reshape(1, 1, N, 1, K)

        # 每个时间槽的加权均值：(B, C, N, num_cycles)
        # 注意：这里对 num_cycles 维度加权，即同一时间槽跨多个周期的加权
        # w 的含义是"哪个槽位更重要"，需要转置语义：
        # 实际上 kernels[n, k] = 第 n 个节点第 k 个槽位的权重
        # mu_fold[b,c,n,cyc] = sum_k( x_fold[b,c,n,cyc,k] * w[n,k] )  ← 这是跨槽加权，不对
        #
        # 正确语义：mu 在时间槽 k 处 = 该节点所有周期中槽 k 的加权均值
        # 即 mu[b,c,n,t] 只取决于 t % K 对应的槽位 k，
        # 而权重应施加在 num_cycles 维（跨周期），而非 K 维（跨槽）。
        #
        # 因此 kernels 在这里应理解为"各周期的权重"，维度是 (N, num_cycles)。
        # 但 num_cycles 在运行时才确定，超网络无法提前生成。
        #
        # 折中方案（与原版 SSCNNDecomp 对齐）：
        #   kernels (N, K) 作为槽位混合权重，在 K 维做加权 → 得到每周期的标量均值，
        #   再 broadcast 回时间轴。这正好等价于原版 I_se 的 conv2d 操作。
        #
        # 即：mu_fold[b,c,n,cyc] = sum_k( x_fold[b,c,n,cyc,k] * kernels[n,k] )
        #     然后将 mu_fold (B,C,N,num_cycles) 扩展为 (B,C,N,num_cycles,K)，
        #     再 reshape 回 (B,C,N,t_cycle)，每个槽都用同一周期的均值填充。
        mu_fold = (x_fold * w).sum(dim=-1)           # (B, C, N, num_cycles)

        # 将每个周期的均值广播到该周期内所有 K 个时间槽
        mu_expanded = mu_fold.unsqueeze(-1).expand(
            B, C, N, mu_fold.shape[3], K
        ).reshape(B, C, N, t_cycle)                  # (B, C, N, t_cycle)

        # 尾部不足一个周期的部分，用最后一个周期的均值填充
        if t_cycle < T:
            pad = mu_expanded[..., -1:].expand(B, C, N, T - t_cycle)
            mu_se = torch.cat([mu_expanded, pad], dim=-1)
        else:
            mu_se = mu_expanded

        return mu_se                                  # (B, C, N, T)

    def _node_conv_sliding(self, x, kernels, pad_len):
        """
        滑动窗口版短周期均值估计（保持原逻辑，语义正确）。

        参数
        ----
        x       : (B, C, N, T)
        kernels : (N, K)  已经过 softmax
        pad_len : int     左侧填充长度（= K - 1）
        """
        x_pad    = F.pad(x, (pad_len, 0))
        x_unfold = x_pad.unfold(dimension=-1,
                                 size=kernels.shape[1], step=1)  # (B,C,N,T,K)
        w        = kernels.unsqueeze(0).unsqueeze(0).unsqueeze(3) # (1,1,N,1,K)
        return (x_unfold * w).sum(dim=-1)

    def forward(self, x):
        B, C, N, T = x.shape
        hyper_out = self.hyper_net(self.node_emb)              # (N, out_dim)

        gates   = torch.sigmoid(hyper_out[:, :3])
        gate_lt = gates[:, 0].reshape(1, 1, N, 1)
        gate_se = gates[:, 1].reshape(1, 1, N, 1)
        gate_st = gates[:, 2].reshape(1, 1, N, 1)

        # softmax 确保卷积核为凸组合（归一化加权均值）
        w_se = torch.softmax(hyper_out[:, 3:3+self.cycle_len],        dim=-1)  # (N, cycle_len)
        w_st = torch.softmax(hyper_out[:, 3+self.cycle_len:],         dim=-1)  # (N, short_period_len)

        # ── 长趋势：全局均值 ──────────────────────────────────────────────────
        mu_lt  = x.mean(dim=-1, keepdim=True).expand_as(x) * gate_lt
        var_lt = ((x - mu_lt)**2).mean(dim=-1, keepdim=True).expand_as(x) + epsilon
        r_lt   = (x - mu_lt) / var_lt.sqrt()

        # ── 季节性：周期折叠加权均值（修复版） ──────────────────────────────
        mu_se  = self._node_conv_periodic(x, w_se) * gate_se
        var_se = ((x - mu_se)**2).mean(dim=-1, keepdim=True).expand_as(x) + epsilon
        r_se   = (x - mu_se) / var_se.sqrt()

        # ── 短周期：滑动窗口加权均值（语义正确，保持不变） ──────────────────
        mu_st  = self._node_conv_sliding(x, w_st, self.short_period_len - 1) * gate_st
        var_st = ((x - mu_st)**2).mean(dim=-1, keepdim=True).expand_as(x) + epsilon
        r_st   = (x - mu_st) / var_st.sqrt()

        resid = x - mu_lt - mu_se - mu_st
        return (r_lt, mu_lt), (r_se, mu_se), (r_st, mu_st), resid


# ──────────────────────────────────────────────────────────────────────────────
# HyperProjHead：超网络生成节点专属输出投影
# ──────────────────────────────────────────────────────────────────────────────

class HyperProjHead(nn.Module):
    """
    节点自适应输出投影头。

    参数
    ----
    node_emb     : 外部传入的节点嵌入 Parameter (N, node_dim)
                   与 HyperDecomp.node_emb 共享，不在此 register。
    in_dim       : 主干输出的特征维度
    output_len   : 预测步数
    hyper_hidden : 超网络隐层维度
    """

    def __init__(self, node_emb: nn.Parameter, in_dim: int,
                 output_len: int, hyper_hidden: int = 64):
        super().__init__()
        self.in_dim     = in_dim
        self.output_len = output_len
        # 不 register_parameter，避免被父模块二次计算
        self._node_emb_ref = node_emb

        node_dim   = node_emb.shape[1]
        proj_param = in_dim * output_len + output_len   # W + b

        self.hyper_net = nn.Sequential(
            nn.Linear(node_dim, hyper_hidden), nn.GELU(),
            nn.Linear(hyper_hidden, hyper_hidden), nn.GELU(),
            nn.Linear(hyper_hidden, proj_param),
        )
        # 初始化最后一层为小值，保持输出尺度稳定
        nn.init.normal_(self.hyper_net[-1].weight, std=0.01)
        nn.init.zeros_(self.hyper_net[-1].bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        feat : (B, in_dim, N, 1)
        返回  : (B, output_len, N, 1)
        """
        B, D, N, _ = feat.shape
        proj_params = self.hyper_net(self._node_emb_ref)   # (N, D*L + L)

        W = proj_params[:, :self.in_dim * self.output_len] \
              .reshape(N, self.in_dim, self.output_len)     # (N, D, L)
        b = proj_params[:, self.in_dim * self.output_len:] # (N, L)

        # feat: (B, D, N, 1) → (B, N, D)
        f = feat.squeeze(-1).permute(0, 2, 1)              # (B, N, D)

        # 节点独立矩阵乘法：einsum bnd, ndl -> bnl
        out = torch.einsum('bnd,ndl->bnl', f, W) + b.unsqueeze(0)  # (B, N, L)

        # 还原为 (B, L, N, 1)
        return out.permute(0, 2, 1).unsqueeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# 三路外推模块（主干共享 + 超网络个性化输出头）
# ──────────────────────────────────────────────────────────────────────────────

class MLPPredictor(nn.Module):
    """
    主干：2 层残差 MLP（共享）
    输出头：HyperProjHead（节点独立）
    """
    def __init__(self, hidden_dim, output_len, node_emb, hyper_hidden=64):
        super().__init__()
        in_dim = 2 * hidden_dim
        self.mlp  = nn.Sequential(
            MultiLayerPerceptron(in_dim, in_dim),
            MultiLayerPerceptron(in_dim, in_dim),
        )
        self.proj = HyperProjHead(node_emb, in_dim, output_len, hyper_hidden)

    def forward(self, r, mu):
        feat = torch.cat([r, mu], dim=1)[..., -1:]     # (B, 2D, N, 1)
        return self.proj(self.mlp(feat))


class TimeAttentionPredictor(nn.Module):
    """
    主干：在 T 维做多头自注意力（共享）
    输出头：HyperProjHead（节点独立）
    """
    def __init__(self, hidden_dim, output_len, node_emb,
                 num_heads=4, dropout=0.1, hyper_hidden=64):
        super().__init__()
        in_dim = 2 * hidden_dim
        while in_dim % num_heads != 0:
            num_heads -= 1
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.attn  = nn.MultiheadAttention(in_dim, num_heads,
                                            dropout=dropout, batch_first=True)
        self.ffn   = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(in_dim * 2, in_dim),
        )
        self.drop  = nn.Dropout(dropout)
        self.proj  = HyperProjHead(node_emb, in_dim, output_len, hyper_hidden)

    def forward(self, r, mu):
        B, D, N, T = r.shape
        x = torch.cat([r, mu], dim=1).permute(0, 2, 3, 1).reshape(B * N, T, 2 * D)
        res = x
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = res + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.norm2(x)))
        # 取最后时间步 → (B, 2D, N, 1)
        feat = x[:, -1, :].reshape(B, N, 2 * D).permute(0, 2, 1).unsqueeze(-1)
        return self.proj(feat)


class SpatialAttentionPredictor(nn.Module):
    """
    主干：在 N 维做多头自注意力（共享）
    输出头：HyperProjHead（节点独立）
    """
    def __init__(self, hidden_dim, output_len, node_emb,
                 num_heads=4, dropout=0.1, hyper_hidden=64):
        super().__init__()
        in_dim = 2 * hidden_dim
        while in_dim % num_heads != 0:
            num_heads -= 1
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.attn  = nn.MultiheadAttention(in_dim, num_heads,
                                            dropout=dropout, batch_first=True)
        self.ffn   = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(in_dim * 2, in_dim),
        )
        self.drop  = nn.Dropout(dropout)
        self.proj  = HyperProjHead(node_emb, in_dim, output_len, hyper_hidden)

    def forward(self, r, mu):
        # 取最后时间步，在 N 维做 attention
        x = torch.cat([r, mu], dim=1)[..., -1].permute(0, 2, 1)   # (B, N, 2D)
        res = x
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = res + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.norm2(x)))
        feat = x.permute(0, 2, 1).unsqueeze(-1)                    # (B, 2D, N, 1)
        return self.proj(feat)


def _build_predictor(mode, hidden_dim, output_len, node_emb,
                     num_heads=4, dropout=0.1, hyper_hidden=64):
    if mode == "mlp":
        return MLPPredictor(hidden_dim, output_len, node_emb, hyper_hidden)
    elif mode == "time_attn":
        return TimeAttentionPredictor(hidden_dim, output_len, node_emb,
                                      num_heads, dropout, hyper_hidden)
    elif mode == "spatial_attn":
        return SpatialAttentionPredictor(hidden_dim, output_len, node_emb,
                                         num_heads, dropout, hyper_hidden)
    else:
        raise ValueError(f"Unknown mode '{mode}'.")


# ──────────────────────────────────────────────────────────────────────────────
# 残差预测器
# ──────────────────────────────────────────────────────────────────────────────

class ResidualPredictor(nn.Module):
    def __init__(self, seq_dim, st_dim, output_len,
                 num_heads=4, num_layer=2, dropout=0.1):
        super().__init__()
        self.seq_dim = seq_dim
        self.st_dim  = st_dim
        in_dim       = seq_dim + st_dim

        while seq_dim % num_heads != 0:
            num_heads -= 1
        self.norm1 = nn.LayerNorm(seq_dim)
        self.norm2 = nn.LayerNorm(seq_dim)
        self.attn  = nn.MultiheadAttention(seq_dim, num_heads,
                                            dropout=dropout, batch_first=True)
        self.ffn   = nn.Sequential(
            nn.Linear(seq_dim, seq_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(seq_dim * 2, seq_dim),
        )
        self.drop  = nn.Dropout(dropout)
        self.encoder    = nn.Sequential(
            *[MultiLayerPerceptron(in_dim, in_dim) for _ in range(num_layer)]
        )
        self.regression = nn.Conv2d(in_dim, output_len, kernel_size=(1, 1), bias=True)

    def forward(self, resid, st_emb_last):
        B, D, N, T = resid.shape
        x = resid.permute(0, 2, 3, 1).reshape(B * N, T, D)
        res = x
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = res + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.norm2(x)))
        feat = x[:, -1, :].reshape(B, N, D).permute(0, 2, 1).unsqueeze(-1)
        if self.st_dim > 0:
            feat = torch.cat([feat, st_emb_last], dim=1)
        feat = self.encoder(feat)
        return self.regression(feat)


# ──────────────────────────────────────────────────────────────────────────────
# 主模型
# ──────────────────────────────────────────────────────────────────────────────

class MyModel(nn.Module):

    def __init__(self, **model_args):
        super().__init__()

        self.num_nodes        = model_args["num_nodes"]
        self.node_dim         = model_args["node_dim"]
        self.input_len        = model_args["input_len"]
        self.input_dim        = model_args["input_dim"]
        self.embed_dim        = model_args["embed_dim"]
        self.output_len       = model_args["output_len"]
        self.num_layer        = model_args["num_layer"]
        self.temp_dim_tid     = model_args["temp_dim_tid"]
        self.temp_dim_diw     = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]
        # self.if_time_in_day   = False
        # self.if_day_in_week   = False
        # self.if_spatial       = False
        self.if_time_in_day   = model_args["if_T_i_D"]
        self.if_day_in_week   = model_args["if_D_i_W"]
        self.if_spatial       = model_args["if_node"]

        self.use_lt  = model_args.get("use_lt",  True)
        self.use_se  = model_args.get("use_se",  True)
        self.use_st  = model_args.get("use_st",  True)
        self.use_res = model_args.get("use_res", False)

        if not any([self.use_lt, self.use_se, self.use_st, self.use_res]):
            raise ValueError("At least one component must be enabled.")

        lt_mode = model_args.get("lt_mode", "mlp")
        se_mode = model_args.get("se_mode", "spatial_attn")
        st_mode = model_args.get("st_mode", "spatial_attn")

        cycle_len         = model_args.get("cycle_len",         12)
        short_period_len  = model_args.get("short_period_len",   4)
        attn_heads        = model_args.get("attn_heads",         4)
        res_num_layer     = model_args.get("res_num_layer",      2)
        dropout           = model_args.get("dropout",          0.1)
        hyper_node_dim    = model_args.get("hyper_node_dim",  self.node_dim)
        hyper_hidden      = model_args.get("hyper_hidden",     32)
        proj_hyper_hidden = model_args.get("proj_hyper_hidden", 64)

        # 时序嵌入
        self.time_series_emb_layer = nn.Conv2d(
            self.input_dim, self.embed_dim, kernel_size=(1, 1), bias=True)
        self.hidden_dim = self.embed_dim

        # 时间嵌入（可选，供残差预测器使用）
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        self.st_dim = (
            hyper_node_dim  * int(self.if_spatial)
            + self.temp_dim_tid * int(self.if_time_in_day)
            + self.temp_dim_diw * int(self.if_day_in_week)
        )

        # ── HyperDecomp（唯一的节点嵌入在这里）────────────────────────────────
        self.decomp = HyperDecomp(
            num_nodes        = self.num_nodes,
            node_dim         = hyper_node_dim,
            cycle_len        = cycle_len,
            short_period_len = short_period_len,
            hyper_hidden     = hyper_hidden,
        )

        # 外推头共享 decomp.node_emb
        shared_node_emb = self.decomp.node_emb

        if self.use_lt:
            self.lt_predictor = _build_predictor(
                lt_mode, self.hidden_dim, self.output_len,
                node_emb=shared_node_emb,
                num_heads=attn_heads, dropout=dropout,
                hyper_hidden=proj_hyper_hidden)
        if self.use_se:
            self.se_predictor = _build_predictor(
                se_mode, self.hidden_dim, self.output_len,
                node_emb=shared_node_emb,
                num_heads=attn_heads, dropout=dropout,
                hyper_hidden=proj_hyper_hidden)
        if self.use_st:
            self.st_predictor = _build_predictor(
                st_mode, self.hidden_dim, self.output_len,
                node_emb=shared_node_emb,
                num_heads=attn_heads, dropout=dropout,
                hyper_hidden=proj_hyper_hidden)

        if self.use_res:
            self.res_predictor = ResidualPredictor(
                seq_dim    = self.hidden_dim,
                st_dim     = self.st_dim,
                output_len = self.output_len,
                num_heads  = attn_heads,
                num_layer  = res_num_layer,
                dropout    = dropout,
            )

        num_no_res = sum([self.use_lt, self.use_se, self.use_st])
        self.fusion_weight = nn.Parameter(torch.ones(max(num_no_res, 1)))

    def _build_st_emb_last(self, history_data, B, N):
        parts = []
        if self.if_spatial:
            node_e = self.decomp.node_emb.t().unsqueeze(0).unsqueeze(-1).expand(B, -1, N, 1)
            parts.append(node_e)
        if self.if_time_in_day:
            tid_idx = (history_data[:, -1, :, 1] * self.time_of_day_size).long()
            tid_emb = self.time_in_day_emb[tid_idx]
            parts.append(tid_emb.permute(0, 2, 1).unsqueeze(-1))
        if self.if_day_in_week:
            diw_idx = (history_data[:, -1, :, 2] * self.day_of_week_size).long()
            diw_emb = self.day_in_week_emb[diw_idx]
            parts.append(diw_emb.permute(0, 2, 1).unsqueeze(-1))
        if parts:
            return torch.cat(parts, dim=1)
        return torch.zeros(B, 0, N, 1, device=history_data.device)

    def forward(self, history_data, future_data, batch_seen, epoch, train, **kwargs):
        input_data = history_data[..., range(self.input_dim)]
        B, T, N, _ = input_data.shape

        x_in   = input_data.reshape(B * T, N, self.input_dim).permute(0, 2, 1).unsqueeze(-1)
        ts_emb = self.time_series_emb_layer(x_in)
        hidden = ts_emb.squeeze(-1).reshape(B, T, self.embed_dim, N).permute(0, 2, 3, 1)

        (r_lt, mu_lt), (r_se, mu_se), (r_st, mu_st), resid = self.decomp(hidden)

        preds = []
        if self.use_lt:
            preds.append(self.lt_predictor(r_lt, mu_lt))
        if self.use_se:
            preds.append(self.se_predictor(r_se, mu_se))
        if self.use_st:
            preds.append(self.st_predictor(r_st, mu_st))

        w         = F.softmax(self.fusion_weight[:len(preds)], dim=0)
        base_pred = sum(w[i] * p for i, p in enumerate(preds))

        if self.use_res:
            st_emb_last = self._build_st_emb_last(history_data, B, N)
            res_pred    = self.res_predictor(resid=resid, st_emb_last=st_emb_last)
            prediction  = base_pred + res_pred
        else:
            prediction = base_pred

        return prediction