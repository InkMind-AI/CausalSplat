#
# CausalSplat: differentiable rasterization of per-Gaussian affinity features.
#

CAUSAL_FEATURE_CHANNELS = 32  # keep in sync with cuda_rasterizer/config_causal.h

from typing import NamedTuple

import torch
import torch.nn as nn
from . import _C

_CAUSAL_EMPTY_FLOAT = torch.tensor([], dtype=torch.float32)


def _cpu_deep_copy_tuple(input_tuple):
    return tuple(
        item.cpu().clone() if isinstance(item, torch.Tensor) else item
        for item in input_tuple
    )


def causal_rasterize_features(
    means3D,
    means2D,
    causal_features_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _CausalRasterizeFeatures.apply(
        means3D,
        means2D,
        causal_features_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )


class _CausalRasterizeFeatures(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        causal_features_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        args = (
            raster_settings.bg,
            means3D,
            causal_features_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            _CAUSAL_EMPTY_FLOAT,
            0,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
        )
        if raster_settings.debug:
            cpu_args = _cpu_deep_copy_tuple(args)
            try:
                num_rendered, features, radii, geomBuffer, binningBuffer, imgBuffer = _C.causal_rasterize_features(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, features, radii, geomBuffer, binningBuffer, imgBuffer = _C.causal_rasterize_features(*args)

        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            causal_features_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        )
        return features, radii

    @staticmethod
    def backward(ctx, grad_out_features, _):
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (
            causal_features_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        ) = ctx.saved_tensors

        args = (
            raster_settings.bg,
            means3D,
            radii,
            causal_features_precomp,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_out_features,
            _CAUSAL_EMPTY_FLOAT,
            0,
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.debug,
        )
        if raster_settings.debug:
            cpu_args = _cpu_deep_copy_tuple(args)
            try:
                (
                    grad_means2D,
                    grad_causal_features_precomp,
                    grad_opacities,
                    grad_means3D,
                    grad_cov3Ds_precomp,
                    _grad_sh,
                    grad_scales,
                    grad_rotations,
                ) = _C.causal_rasterize_features_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            (
                grad_means2D,
                grad_causal_features_precomp,
                grad_opacities,
                grad_means3D,
                grad_cov3Ds_precomp,
                _grad_sh,
                grad_scales,
                grad_rotations,
            ) = _C.causal_rasterize_features_backward(*args)

        return (
            grad_means3D,
            grad_means2D,
            grad_causal_features_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )


class CausalFeatureRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int  # unused; kept for caller compatibility
    campos: torch.Tensor
    prefiltered: bool
    debug: bool


class CausalFeatureRasterizer(nn.Module):
    """Differentiable rasterizer for precomputed CausalSplat affinity features."""

    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        causal_features_precomp,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
        colors_precomp=None,
        shs=None,
    ):
        if causal_features_precomp is None:
            causal_features_precomp = colors_precomp
        if causal_features_precomp is None:
            raise ValueError(
                f"causal_features_precomp is required ({CAUSAL_FEATURE_CHANNELS}-D affinity features)."
            )
        if shs is not None:
            raise ValueError("Spherical harmonics shading is not supported.")

        if (scales is None or rotations is None) and cov3D_precomp is None:
            raise ValueError("Provide either scales and rotations, or cov3D_precomp.")
        if (scales is not None or rotations is not None) and cov3D_precomp is not None:
            raise ValueError("Provide either scales/rotations or cov3D_precomp, not both.")

        if scales is None:
            scales = _CAUSAL_EMPTY_FLOAT
        if rotations is None:
            rotations = _CAUSAL_EMPTY_FLOAT
        if cov3D_precomp is None:
            cov3D_precomp = _CAUSAL_EMPTY_FLOAT

        return causal_rasterize_features(
            means3D,
            means2D,
            causal_features_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            self.raster_settings,
        )