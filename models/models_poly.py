r"""PolyINR model."""
from typing import Optional
import torch.nn.functional as F
from typing import Optional
import torch
import torch.nn as nn
import math


class Clamp(nn.Module):
    """Crop values within a fixed range."""

    def __init__(self, threshold=256.):
        super(Clamp, self).__init__()
        if threshold <= 0.:
            raise ValueError(f"'threshold' ({threshold}) should be positive.")
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: apply clamp."""
        return torch.clamp(x, min=-self.threshold, max=self.threshold)


class UniformInitDense(nn.Module):
    # >>> import torch
    # >>> dense = UniformInitDense(10, 5, has_bias=True, scale=0.1)
    # >>> x = torch.randn(16, 10)
    # >>> y = dense(x)
    # >>> print(y.shape)
    def __init__(self, dim_in: int, dim_out: int, has_bias: bool = True, scale: float = None):
        super(UniformInitDense, self).__init__()

        # Initialize the linear layer
        self.fc = nn.Linear(dim_in, dim_out, bias=has_bias)

        # Initialize weights and bias
        if scale is None:
            scale = math.sqrt(1 / dim_in)  # Kaiming uniform initialization

        # Uniform initialization for weights
        nn.init.uniform_(self.fc.weight, -scale, scale)

        # Uniform initialization for bias, if applicable
        if has_bias:
            nn.init.uniform_(self.fc.bias, -scale, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 dim_hidden: int,
                 num_layers: int = 3,
                 compute_dtype=torch.float32) -> None:
        super(MLP, self).__init__()

        layers = []
        if num_layers > 1:
            layers.append(UniformInitDense(dim_in, dim_hidden))
            layers.append(nn.GELU())
            for _ in range(num_layers - 2):
                layers.append(UniformInitDense(dim_hidden, dim_hidden))
                layers.append(nn.GELU())
            layers.append(UniformInitDense(dim_hidden, dim_out))
        elif num_layers == 1:
            layers.append(UniformInitDense(dim_in, dim_out))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolyINR(nn.Module):

    # >>> import torch
    # >>> x = torch.randn(2, 10, 3)
    # >>> poly_inr = PolyINR(3, 64, 128, 2)
    # >>> out = poly_inr(x)
    # >>> print(out.shape)
    # torch.Size([2, 10, 64])

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 dim_hidden: int,
                 num_layers: int,
                 compute_dtype=torch.float32) -> None:
        super(PolyINR, self).__init__()
        self.num_layers = num_layers

        self.coord_pad = nn.ConstantPad1d((0, 1), 1.0)
        self.affines = nn.ModuleList([UniformInitDense(dim_in, dim_hidden) for _ in range(num_layers - 1)])
        #self.act = nn.Sequential(
        #    nn.LeakyReLU(0.2),
        #    nn.Hardtanh(min_val=0, max_val=256.)
        #)
        self.act = nn.GELU()
        self.dense_layers = nn.ModuleList([UniformInitDense(dim_hidden, dim_hidden) for _ in range(num_layers - 1)])
        self.last_layer = UniformInitDense(dim_hidden, dim_out)

    def forward(self,
                x: torch.Tensor,  # [bsz, num_points, dim_in]
                affine_modulations: Optional[torch.Tensor] = None,  # [num_layers-1, bsz, dim_in+1, dim_hidden]
                scale_modulations: Optional[torch.Tensor] = None,  # [num_layers-1, bsz, dim_hidden]
                shift_modulations: Optional[torch.Tensor] = None,  # [num_layers-1, bsz, dim_hidden]
                ) -> torch.Tensor:
        r"""forward"""
        hidden_state = 1.0

        x_pad = self.coord_pad(x)  # [bsz, n_pts, dim_in] -> [bsz, n_pts, dim_in+1]

        for layer_idx in range(self.num_layers - 1):
            if scale_modulations is None:
                scale = 1.0
            else:
                scale = 1.0 + scale_modulations[layer_idx].unsqueeze(1)  # [bsz, 1, dim_hidden]

            if shift_modulations is None:
                shift = 0.0
            else:
                shift = shift_modulations[layer_idx].unsqueeze(1)  # [bsz, 1, dim_hidden]

            tmp = self.affines[layer_idx](x)  # [bsz, n_pts, dim_hidden]
            if affine_modulations is not None:
                tmp2 = torch.matmul(x_pad, affine_modulations[layer_idx])  # [bsz, n_pts, dim_hidden]
                tmp = tmp + tmp2  # [bsz, n_pts, dim_hidden]
            hidden_state = hidden_state * tmp  # [bsz, n_pts, dim_hidden]
            hidden_state = self.dense_layers[layer_idx](hidden_state)  # [bsz, n_pts, dim_hidden]
            hidden_state = scale * hidden_state + shift
            hidden_state = self.act(hidden_state)  # [bsz, n_pts, dim_hidden]

        out = self.last_layer(hidden_state)  # [bsz, n_pts, dim_out]
        return out.reshape(out.shape[0], 1)


class PolyINRWithHypernet(nn.Module):
    def __init__(self,
                 inr_dim_in: int,
                 inr_dim_out: int,
                 inr_dim_hidden: int,
                 inr_num_layers: int,
                 hyper_dim_in: int,
                 hyper_dim_hidden: int,
                 hyper_num_layers: int,
                 share_hypernet: bool = False,
                 enable_affine: bool = False,
                 enable_shift: bool = True,
                 enable_scale: bool = True,
                 compute_dtype=torch.float32):
        super().__init__()
        self.inr_num_layers = inr_num_layers
        self.enable_affine = enable_affine
        self.enable_shift = enable_shift
        self.enable_scale = enable_scale
        if not (enable_affine or enable_shift or enable_scale):
            raise ValueError(
                "For 'PolyINRWithHypernet', at least one of ['enable_affine', "
                "'enable_shift', 'enable_scale'] should be True.")
        self.share_hypernet = share_hypernet
        self.affine_modulations_shape = (
            inr_num_layers - 1, -1, inr_dim_in + 1, inr_dim_hidden)

        self.inr = PolyINR(
            dim_in=inr_dim_in,
            dim_hidden=inr_dim_hidden,
            dim_out=inr_dim_out,
            num_layers=inr_num_layers,
            compute_dtype=compute_dtype
        )
        
        
        def new_hypernet_mlp(mode):
            if mode == 'affine':
                hyper_dim_out = (inr_dim_in + 1) * inr_dim_hidden
            elif mode in ['shift', 'scale']:
                hyper_dim_out = inr_dim_hidden
            return MLP(hyper_dim_in, hyper_dim_out, hyper_dim_hidden,
                       hyper_num_layers, compute_dtype)

        if self.share_hypernet:
            if enable_affine:
                self.affine_hypernet = new_hypernet_mlp('affine')
            if enable_shift:
                self.shift_hypernet = new_hypernet_mlp('shift')
            if enable_scale:
                self.scale_hypernet = new_hypernet_mlp('scale')
        else:
            num_hypernet = inr_num_layers - 1  # the number of hidden layers
            if self.enable_affine:
                self.affine_hypernets = nn.ModuleList([
                    new_hypernet_mlp('affine') for _ in range(num_hypernet)])
            if self.enable_shift:
                self.shift_hypernets = nn.ModuleList([
                    new_hypernet_mlp('shift') for _ in range(num_hypernet)])
            if self.enable_scale:
                self.scale_hypernets = nn.ModuleList([
                    new_hypernet_mlp('scale') for _ in range(num_hypernet)])

    def forward(self, hyper_in: torch.Tensor, coordinate) -> torch.Tensor:
        #coordinate = torch.cat((coordinate_x, coordinate_y), -1)
        if self.enable_affine:
            affine_modulations = []
            for idx in range(self.inr_num_layers - 1):
                encoder_in = hyper_in[idx]  # [n_graph, embed_dim]
                if self.share_hypernet:
                    encoder_out = self.affine_hypernet(encoder_in)
                else:
                    encoder_out = self.affine_hypernets[idx](encoder_in)
                affine_modulations.append(encoder_out)

            # tensor shape [inr_num_layers - 1, n_graph, (dim_in + 1) * inr_dim_hidden]
            # -> [inr_num_layers - 1, n_graph, dim_in + 1, inr_dim_hidden]
            affine_modulations = torch.stack(affine_modulations, dim=0).view(self.affine_modulations_shape)
        else:
            affine_modulations = None

        if self.enable_shift:
            shift_modulations = []
            for idx in range(self.inr_num_layers - 1):
                encoder_in = hyper_in # [n_graph, embed_dim]
                if self.share_hypernet:
                    encoder_out = self.shift_hypernet(encoder_in)
                else:
                    encoder_out = self.shift_hypernets[idx](encoder_in)
                shift_modulations.append(encoder_out)

            shift_modulations = torch.stack(shift_modulations, dim=0)  # [inr_num_layers - 1, n_graph, inr_dim_hidden]
        else:
            shift_modulations = None

        if self.enable_scale:
            scale_modulations = []
            for idx in range(self.inr_num_layers - 1):
                encoder_in = hyper_in # [n_graph, embed_dim]
                if self.share_hypernet:
                    encoder_out = self.scale_hypernet(encoder_in)
                else:
                    encoder_out = self.scale_hypernets[idx](encoder_in)
                scale_modulations.append(encoder_out)
            scale_modulations = torch.stack(scale_modulations, dim=0)

        coordinate = coordinate.reshape(coordinate.shape[0], -1, coordinate.shape[-1])
        out = self.inr(coordinate, affine_modulations, scale_modulations, shift_modulations)
        return out


