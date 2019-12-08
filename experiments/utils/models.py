import logging
import torch
from torch import nn
from torch.nn import functional as F

from experiments.utils import ALGORITHMS
from manifold_flow import nn as nn_, transforms
from manifold_flow.flows import Flow, ManifoldFlow
from manifold_flow.nn import Conv2dSameSize
from manifold_flow.utils import various

logger = logging.getLogger(__name__)


class ConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.net = nn.Sequential(
            Conv2dSameSize(in_channels, hidden_channels, kernel_size=3),
            nn.ReLU(),
            Conv2dSameSize(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            Conv2dSameSize(hidden_channels, out_channels, kernel_size=3),
        )

    def forward(self, inputs, context=None):
        return self.net.forward(inputs)


def _create_image_transform_step(
    num_channels,
    hidden_channels=96,
    actnorm=True,
    coupling_layer_type="rational_quadratic_spline",
    spline_params={
        "apply_unconditional_transform": False,
        "min_bin_height": 0.001,
        "min_bin_width": 0.001,
        "min_derivative": 0.001,
        "num_bins": 4,
        "tail_bound": 3.0,
    },
    use_resnet=True,
    num_res_blocks=3,
    resnet_batchnorm=True,
    dropout_prob=0.0,
):
    if use_resnet:

        def create_convnet(in_channels, out_channels):
            net = nn_.ConvResidualNet(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                num_blocks=num_res_blocks,
                use_batch_norm=resnet_batchnorm,
                dropout_probability=dropout_prob,
            )
            return net

    else:
        if dropout_prob != 0.0:
            raise ValueError()

        def create_convnet(in_channels, out_channels):
            return ConvNet(in_channels, hidden_channels, out_channels)

    mask = various.create_mid_split_binary_mask(num_channels)

    if coupling_layer_type == "cubic_spline":
        coupling_layer = transforms.PiecewiseCubicCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            tails="linear",
            tail_bound=spline_params["tail_bound"],
            num_bins=spline_params["num_bins"],
            apply_unconditional_transform=spline_params["apply_unconditional_transform"],
            min_bin_width=spline_params["min_bin_width"],
            min_bin_height=spline_params["min_bin_height"],
        )
    elif coupling_layer_type == "quadratic_spline":
        coupling_layer = transforms.PiecewiseQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            tails="linear",
            tail_bound=spline_params["tail_bound"],
            num_bins=spline_params["num_bins"],
            apply_unconditional_transform=spline_params["apply_unconditional_transform"],
            min_bin_width=spline_params["min_bin_width"],
            min_bin_height=spline_params["min_bin_height"],
        )
    elif coupling_layer_type == "rational_quadratic_spline":
        coupling_layer = transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            tails="linear",
            tail_bound=spline_params["tail_bound"],
            num_bins=spline_params["num_bins"],
            apply_unconditional_transform=spline_params["apply_unconditional_transform"],
            min_bin_width=spline_params["min_bin_width"],
            min_bin_height=spline_params["min_bin_height"],
            min_derivative=spline_params["min_derivative"],
        )
    elif coupling_layer_type == "affine":
        coupling_layer = transforms.AffineCouplingTransform(mask=mask, transform_net_create_fn=create_convnet)
    elif coupling_layer_type == "additive":
        coupling_layer = transforms.AdditiveCouplingTransform(mask=mask, transform_net_create_fn=create_convnet)
    else:
        raise RuntimeError("Unknown coupling_layer_type")

    step_transforms = []

    if actnorm:
        step_transforms.append(transforms.ActNorm(num_channels))

    step_transforms.extend([transforms.OneByOneConvolution(num_channels), coupling_layer])

    logger.debug("  Flow based on %s", coupling_layer_type)

    return transforms.CompositeTransform(step_transforms)


def create_image_transform(c, h, w, levels=3, hidden_channels=96, steps_per_level=7, alpha=0.05, num_bits=8, preprocessing="glow", multi_scale=True):
    dim = c * h * w
    if not isinstance(hidden_channels, list):
        hidden_channels = [hidden_channels] * levels

    if multi_scale:
        mct = transforms.MultiscaleCompositeTransform(num_transforms=levels)
        for level, level_hidden_channels in zip(range(levels), hidden_channels):
            logger.debug("Level %s", level)
            squeeze_transform = transforms.SqueezeTransform()
            c, h, w = squeeze_transform.get_output_shape(c, h, w)
            logger.debug("  c, h, w = %s, %s, %s", c, h, w)

            logger.debug("  SqueezeTransform()")
            transform_level = transforms.CompositeTransform(
                [squeeze_transform]
                + [_create_image_transform_step(c, level_hidden_channels) for _ in range(steps_per_level)]
                + [transforms.OneByOneConvolution(c)]  # End each level with a linear transformation.
            )
            logger.debug("  OneByOneConvolution(%s)", c)

            new_shape = mct.add_transform(transform_level, (c, h, w))
            if new_shape:  # If not last layer
                c, h, w = new_shape
                logger.debug("  new_shape = %s, %s, %s", c, h, w)
    else:
        all_transforms = []

        for level, level_hidden_channels in zip(range(levels), hidden_channels):
            squeeze_transform = transforms.SqueezeTransform()
            c, h, w = squeeze_transform.get_output_shape(c, h, w)

            transform_level = transforms.CompositeTransform(
                [squeeze_transform]
                + [_create_image_transform_step(c, level_hidden_channels) for _ in range(steps_per_level)]
                + [transforms.OneByOneConvolution(c)]  # End each level with a linear transformation.
            )
            all_transforms.append(transform_level)

        all_transforms.append(transforms.ReshapeTransform(input_shape=(c, h, w), output_shape=(c * h * w,)))
        mct = transforms.CompositeTransform(all_transforms)

    # Inputs to the model in [0, 2 ** num_bits]

    if preprocessing == "glow":
        # Map to [-0.5,0.5]
        preprocess_transform = transforms.AffineScalarTransform(scale=(1.0 / 2 ** num_bits), shift=-0.5)
    elif preprocessing == "realnvp":
        preprocess_transform = transforms.CompositeTransform(
            [
                # Map to [0,1]
                transforms.AffineScalarTransform(scale=(1.0 / 2 ** num_bits)),
                # Map into unconstrained space as done in RealNVP
                transforms.AffineScalarTransform(shift=alpha, scale=(1 - alpha)),
                transforms.Logit(),
            ]
        )

    elif preprocessing == "realnvp_2alpha":
        preprocess_transform = transforms.CompositeTransform(
            [
                transforms.AffineScalarTransform(scale=(1.0 / 2 ** num_bits)),
                transforms.AffineScalarTransform(shift=alpha, scale=(1 - 2.0 * alpha)),
                transforms.Logit(),
            ]
        )
    else:
        raise RuntimeError("Unknown preprocessing type: {}".format(preprocessing))

    # Random permutation
    permutation = transforms.RandomPermutation(dim)
    logger.debug("RandomPermutation(%s)", dim)

    return transforms.CompositeTransform([preprocess_transform, mct, permutation])


def _create_vector_linear_transform(linear_transform_type, features):
    if linear_transform_type == "permutation":
        return transforms.RandomPermutation(features=features)
    elif linear_transform_type == "lu":
        return transforms.CompositeTransform([transforms.RandomPermutation(features=features), transforms.LULinear(features, identity_init=True)])
    elif linear_transform_type == "svd":
        return transforms.CompositeTransform(
            [transforms.RandomPermutation(features=features), transforms.SVDLinear(features, num_householder=10, identity_init=True)]
        )
    else:
        raise ValueError


def _create_vector_base_transform(
    i,
    base_transform_type,
    features,
    hidden_features,
    num_transform_blocks,
    dropout_probability,
    use_batch_norm,
    num_bins,
    tail_bound,
    apply_unconditional_transform,
    context_features,
    resnet_transform,
):
    if resnet_transform:
        transform_net_create_fn = lambda in_features, out_features: nn_.ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_transform_blocks,
            activation=F.relu,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
    else:
        transform_net_create_fn = lambda in_features, out_features: nn_.MLP(
            in_shape=(in_features,),
            out_shape=(out_features,),
            hidden_sizes=[hidden_features for _ in range(num_transform_blocks)],
            context_features=context_features,
            activation=F.relu,
        )

    if base_transform_type == "affine-coupling":
        return transforms.AffineCouplingTransform(
            mask=various.create_alternating_binary_mask(features, even=(i % 2 == 0)), transform_net_create_fn=transform_net_create_fn
        )
    elif base_transform_type == "quadratic-coupling":
        return transforms.PiecewiseQuadraticCouplingTransform(
            mask=various.create_alternating_binary_mask(features, even=(i % 2 == 0)),
            transform_net_create_fn=transform_net_create_fn,
            num_bins=num_bins,
            tails="linear",
            tail_bound=tail_bound,
            apply_unconditional_transform=apply_unconditional_transform,
        )
    elif base_transform_type == "rq-coupling":
        return transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=various.create_alternating_binary_mask(features, even=(i % 2 == 0)),
            transform_net_create_fn=transform_net_create_fn,
            num_bins=num_bins,
            tails="linear",
            tail_bound=tail_bound,
            apply_unconditional_transform=apply_unconditional_transform,
        )
    elif base_transform_type == "affine-autoregressive":
        return transforms.MaskedAffineAutoregressiveTransform(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=F.relu,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
    elif base_transform_type == "quadratic-autoregressive":
        return transforms.MaskedPiecewiseQuadraticAutoregressiveTransform(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_bins=num_bins,
            tails="linear",
            tail_bound=tail_bound,
            num_blocks=num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=F.relu,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
    elif base_transform_type == "rq-autoregressive":
        return transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_bins=num_bins,
            tails="linear",
            tail_bound=tail_bound,
            num_blocks=num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=F.relu,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
    else:
        raise ValueError


def create_vector_transform(
    dim,
    flow_steps,
    linear_transform_type="permutation",
    base_transform_type="rq-coupling",
    hidden_features=256,
    num_transform_blocks=3,
    dropout_probability=0.25,
    use_batch_norm=False,
    num_bins=8,
    tail_bound=3,
    apply_unconditional_transform=True,
    context_features=None,
    resnet_transform=True,
):
    transform = transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    _create_vector_linear_transform(linear_transform_type, dim),
                    _create_vector_base_transform(
                        i,
                        base_transform_type,
                        dim,
                        hidden_features,
                        num_transform_blocks,
                        dropout_probability,
                        use_batch_norm,
                        num_bins,
                        tail_bound,
                        apply_unconditional_transform,
                        context_features,
                        resnet_transform,
                    ),
                ]
            )
            for i in range(flow_steps)
        ]
        + [_create_vector_linear_transform(linear_transform_type, dim)]
    )
    return transform


def create_model(args, simulator):
    assert args.algorithm in ALGORITHMS

    if not simulator.is_image() and args.algorithm == "flow":
        logger.info(
            "Creating standard flow for vector data with %s layers, transform %s, %s context features",
            args.innerlayers + args.outerlayers,
            args.outertransform,
            context_features=simulator.parameter_dim(),
        )
        transform = create_vector_transform(
            args.datadim,
            args.innerlayers + args.outerlayers,
            linear_transform_type=args.lineartransform,
            base_transform_type=args.outertransform,
            context_features=simulator.parameter_dim(),
        )
        model = Flow(data_dim=args.datadim, transform=transform)

    elif simulator.is_image() and args.algorithm == "flow":
        logger.info(
            "Creating standard flow for image data with %s layers, transform %s, %s context features",
            args.innerlayers + args.outerlayers,
            args.outertransform,
            context_features=simulator.parameter_dim(),
        )
        raise NotImplementedError

    elif not simulator.is_image():
        logger.info(
            "Creating manifold flow for vector data with %s latent dimensions, %s + %s layers, transforms %s / %s, %s context features",
            args.modellatentdim,
            args.outerlayers,
            args.innerlayers,
            args.outertransform,
            args.innertransform,
            context_features=simulator.parameter_dim(),
        )

        outer_transform_kwargs = {}
        try:
            outer_transform_kwargs["hidden_features"] = args.outercouplinghidden
            outer_transform_kwargs["num_transform_blocks"] = args.outercouplinglayers
            outer_transform_kwargs["resnet_transform"] = not args.outercouplingmlp
            logger.info("Additional settings for outer transform: %s", outer_transform_kwargs)
        except:
            pass
        outer_transform = create_vector_transform(
            args.datadim,
            args.outerlayers,
            linear_transform_type=args.lineartransform,
            base_transform_type=args.outertransform,
            context_features=simulator.parameter_dim() if args.conditionalouter else None,
        )
        inner_transform = create_vector_transform(
            args.modellatentdim,
            args.innerlayers,
            linear_transform_type=args.lineartransform,
            base_transform_type=args.innertransform,
            context_features=simulator.parameter_dim(),
        )

        model = ManifoldFlow(
            data_dim=args.datadim,
            latent_dim=args.modellatentdim,
            outer_transform=outer_transform,
            inner_transform=inner_transform,
            apply_context_to_outer=args.conditionalouter,
        )

    else:
        logger.info(
            "Creating manifold flow for image data with %s latent dimensions, %s + %s layers, transforms %s / %s, %s context features",
            args.modellatentdim,
            args.outerlayers,
            args.innerlayers,
            args.outertransform,
            args.innertransform,
            context_features=simulator.parameter_dim(),
        )
        raise NotImplementedError

    return model
