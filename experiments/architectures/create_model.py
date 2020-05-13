import logging

from .image_transforms import create_image_transform, create_image_encoder
from .vector_transforms import create_vector_encoder, create_vector_transform
from manifold_flow import transforms
from manifold_flow.flows import Flow, EncoderManifoldFlow, VariableDimensionManifoldFlow, ManifoldFlow


logger = logging.getLogger(__name__)


ALGORITHMS = ["flow", "pie", "mf", "gamf", "hybrid", "emf"]  # , "dough"


def create_model(args, simulator):
    assert args.algorithm in ALGORITHMS

    if simulator.is_image():
        c, h, w = simulator.data_dim()
    else:
        c, h, w = None, None, None

    # Ambient flow for vector data
    if args.algorithm == "flow" and not simulator.is_image():
        model = create_vector_flow(args, simulator)

    # Ambient flow for image data
    elif args.algorithm == "flow" and simulator.is_image():
        model = create_image_flow(args, c, h, simulator, w)

    # FOM for vector data
    elif args.algorithm in ["mf", "gamf", "pie"] and args.specified and not simulator.is_image():
        model = create_vector_specified_flow(args, simulator)

    # FOM for image data
    elif args.algorithm in ["mf", "gamf", "pie"] and args.specified and simulator.is_image():
        raise NotImplementedError

    # M-flow or PIE for vector data
    elif args.algorithm in ["mf", "gamf", "pie"] and not args.specified and not simulator.is_image():
        model = create_vector_mf(args, simulator)

    # M-flow or PIE for image data, with structured (image) latent space
    elif args.algorithm in ["mf", "gamf", "pie"] and not args.specified and simulator.is_image() and args.structuredlatents:
        model = create_image_mf_structured(args, c, h, simulator, w)

    # M-flow or PIE for image data, with scalar latent space
    elif args.algorithm in ["mf", "gamf", "pie"] and not args.specified and simulator.is_image() and not args.structuredlatents:
        model = create_image_mf_unstructured(args, c, h, simulator, w)

    # M-flow with sep. encoder for vector data
    elif args.algorithm == "emf" and not simulator.is_image():
        model = create_vector_emf(args, simulator)

    # M-flow with sep. encoder for image data (unstructured latents)
    elif args.algorithm == "emf" and simulator.is_image() and not args.structuredlatents:
        model = create_image_emf_unstructured(args, c, h, simulator, w)

    # M-flow with sep. encoder for image data (structured latents)
    elif args.algorithm == "emf" and simulator.is_image() and args.structuredlatents:
        raise NotImplementedError

    # # PIE with variable epsilon for vector data
    # elif not simulator.is_image() and args.algorithm == "dough":
    #     model = create_scalar_dough(args, simulator)

    # # PIE with variable epsilon for image data
    # elif simulator.is_image() and args.algorithm == "dough":
    #     raise NotImplementedError

    else:
        raise ValueError(f"Don't know how to construct model for algorithm {args.algorithm} and image flag {simulator.is_image()}")

    return model


def create_image_mf_unstructured(args, c, h, simulator, w):
    if simulator.parameter_dim() is not None and args.conditionalouter:
        raise NotImplementedError
    steps_per_level = (args.outerlayers) // args.levels
    logger.info(
        "Creating manifold flow for image data with %s levels and %s steps per level in the outer transformation, %s layers in the inner transformation, transforms %s / %s, %s context features",
        args.levels,
        steps_per_level,
        args.innerlayers,
        args.outertransform,
        args.innertransform,
        simulator.parameter_dim(),
    )
    spline_params = {
        "apply_unconditional_transform": False,
        "min_bin_height": 0.001,
        "min_bin_width": 0.001,
        "min_derivative": 0.001,
        "num_bins": args.splinebins,
        "tail_bound": args.splinerange,
    }
    outer_transform = create_image_transform(
        c,
        h,
        w,
        levels=args.levels,
        hidden_channels=100,
        steps_per_level=steps_per_level,
        num_res_blocks=2,
        alpha=0.05,
        num_bits=8,
        preprocessing="glow",
        dropout_prob=args.dropout,
        multi_scale=True,
        spline_params=spline_params,
        postprocessing="partial_mlp",
        postprocessing_layers=args.linlayers,
        postprocessing_channel_factor=args.linchannelfactor,
        use_actnorm=args.actnorm,
        use_batchnorm=args.batchnorm,
    )
    inner_transform = create_vector_transform(
        args.modellatentdim,
        args.innerlayers,
        linear_transform_type=args.lineartransform,
        base_transform_type=args.innertransform,
        context_features=simulator.parameter_dim(),
        dropout_probability=args.dropout,
        tail_bound=args.splinerange,
        num_bins=args.splinebins,
        use_batch_norm=args.batchnorm,
    )
    model = ManifoldFlow(
        data_dim=args.datadim,
        latent_dim=args.modellatentdim,
        outer_transform=outer_transform,
        inner_transform=inner_transform,
        apply_context_to_outer=args.conditionalouter,
        pie_epsilon=args.pieepsilon,
    )
    return model


def create_image_mf_structured(args, c, h, simulator, w):
    if simulator.parameter_dim() is not None:
        raise NotImplementedError
    steps_per_level = args.outerlayers // args.levels
    inner_steps_per_level = args.innerlayers // args.innerlevels
    h_inner = h // 2 ** args.levels
    w_inner = w // 2 ** args.levels
    inner_channels = args.modellatentdim // (h_inner * w_inner)
    logger.debug("Channels after projection: %s x %s x %s", inner_channels, h_inner, w_inner)
    assert args.modellatentdim == inner_channels * (h_inner * w_inner)
    logger.info(
        "Creating manifold flow for image data with %s levels and %s steps per level in the outer transformation, %s levels and %s steps per level in the inner transformation, %s inner channels, transforms %s / %s, %s context features",
        args.levels,
        steps_per_level,
        args.innerlevels,
        inner_steps_per_level,
        inner_channels,
        args.outertransform,
        args.innertransform,
        simulator.parameter_dim(),
    )
    spline_params = {
        "apply_unconditional_transform": False,
        "min_bin_height": 0.001,
        "min_bin_width": 0.001,
        "min_derivative": 0.001,
        "num_bins": args.splinebins,
        "tail_bound": args.splinerange,
    }
    outer_transform = create_image_transform(
        c,
        h,
        w,
        levels=args.levels,
        hidden_channels=100,
        steps_per_level=steps_per_level,
        num_res_blocks=2,
        alpha=0.05,
        num_bits=8,
        preprocessing="glow",
        dropout_prob=args.dropout,
        multi_scale=True,
        spline_params=spline_params,
        postprocessing="none",
        use_actnorm=args.actnorm,
        use_batchnorm=args.batchnorm,
    )
    inner_transform = create_image_transform(
        inner_channels,
        h_inner,
        w_inner,
        levels=args.innerlevels,
        hidden_channels=100,
        steps_per_level=steps_per_level,
        num_res_blocks=2,
        alpha=0.05,
        num_bits=8,
        preprocessing="unflatten",
        dropout_prob=args.dropout,
        multi_scale=True,
        spline_params=spline_params,
        postprocessing="permutation",
        use_actnorm=args.actnorm,
        use_batchnorm=args.batchnorm,
    )
    model = ManifoldFlow(
        data_dim=args.datadim,
        latent_dim=args.modellatentdim,
        outer_transform=outer_transform,
        inner_transform=inner_transform,
        apply_context_to_outer=args.conditionalouter,
        pie_epsilon=args.pieepsilon,
    )
    return model


def create_vector_mf(args, simulator):
    logger.info(
        "Creating manifold flow for vector data with %s latent dimensions, %s + %s layers, transforms %s / %s, %s context features",
        args.modellatentdim,
        args.outerlayers,
        args.innerlayers,
        args.outertransform,
        args.innertransform,
        simulator.parameter_dim(),
    )
    outer_transform = create_vector_transform(
        args.datadim,
        args.outerlayers,
        linear_transform_type=args.lineartransform,
        base_transform_type=args.outertransform,
        context_features=simulator.parameter_dim() if args.conditionalouter else None,
        dropout_probability=args.dropout,
        tail_bound=args.splinerange,
        num_bins=args.splinebins,
        use_batch_norm=args.batchnorm,
    )
    inner_transform = create_vector_transform(
        args.modellatentdim,
        args.innerlayers,
        linear_transform_type=args.lineartransform,
        base_transform_type=args.innertransform,
        context_features=simulator.parameter_dim(),
        dropout_probability=args.dropout,
        tail_bound=args.splinerange,
        num_bins=args.splinebins,
        use_batch_norm=args.batchnorm,
    )
    model = ManifoldFlow(
        data_dim=args.datadim,
        latent_dim=args.modellatentdim,
        outer_transform=outer_transform,
        inner_transform=inner_transform,
        apply_context_to_outer=args.conditionalouter,
        pie_epsilon=args.pieepsilon,
    )
    return model


def create_vector_specified_flow(args, simulator):
    logger.info(
        "Creating manifold flow for vector data with %s latent dimensions, specified outer transformation + %s inner layers, transform %s, %s context features",
        args.modellatentdim,
        args.innerlayers,
        args.innertransform,
        simulator.parameter_dim(),
    )
    if args.dataset in ["spherical_gaussian", "conditional_spherical_gaussian"]:
        outer_transform = transforms.SphericalCoordinates(n=args.modellatentdim, r0=1.0)
    else:
        raise NotImplementedError("Specified outer transformation not yet implemented for dataset {}".format(args.dataset))
    inner_transform = create_vector_transform(
        args.modellatentdim,
        args.innerlayers,
        linear_transform_type=args.lineartransform,
        base_transform_type=args.innertransform,
        context_features=simulator.parameter_dim(),
        dropout_probability=args.dropout,
        tail_bound=args.splinerange,
        num_bins=args.splinebins,
        use_batch_norm=args.batchnorm,
    )
    model = ManifoldFlow(
        data_dim=args.datadim,
        latent_dim=args.modellatentdim,
        outer_transform=outer_transform,
        inner_transform=inner_transform,
        apply_context_to_outer=args.conditionalouter,
        pie_epsilon=args.pieepsilon,
    )
    return model


def create_scalar_dough(args, simulator):
    logger.info(
        "Creating variable-dimensional manifold flow for vector data with %s layers, transform %s, %s context features",
        args.innerlayers + args.outerlayers,
        args.outertransform,
        simulator.parameter_dim(),
    )
    transform = create_vector_transform(
        args.datadim,
        args.innerlayers + args.outerlayers,
        linear_transform_type=args.lineartransform,
        base_transform_type=args.outertransform,
        context_features=simulator.parameter_dim(),
        dropout_probability=args.dropout,
        tail_bound=args.splinerange,
        num_bins=args.splinebins,
        use_batch_norm=args.batchnorm,
    )
    model = VariableDimensionManifoldFlow(data_dim=args.datadim, transform=transform)
    return model


def create_vector_emf(args, simulator):
    logger.info(
        "Creating manifold flow + encoder for vector data with %s latent dimensions, %s + %s layers, encoder with %s blocks, transforms %s / %s, %s context features",
        args.modellatentdim,
        args.outerlayers,
        args.innerlayers,
        args.encoderblocks,
        args.outertransform,
        args.innertransform,
        simulator.parameter_dim(),
    )
    encoder = create_vector_encoder(
        args.datadim,
        args.modellatentdim,
        args.encoderhidden,
        args.encoderblocks,
        dropout_probability=args.dropout,
        context_features=simulator.parameter_dim() if args.conditionalouter else None,
        use_batch_norm=args.batchnorm,
    )
    outer_transform = create_vector_transform(
        args.datadim,
        args.outerlayers,
        linear_transform_type=args.lineartransform,
        base_transform_type=args.outertransform,
        context_features=simulator.parameter_dim() if args.conditionalouter else None,
        dropout_probability=args.dropout,
        tail_bound=args.splinerange,
        num_bins=args.splinebins,
        use_batch_norm=args.batchnorm,
    )
    inner_transform = create_vector_transform(
        args.modellatentdim,
        args.innerlayers,
        linear_transform_type=args.lineartransform,
        base_transform_type=args.innertransform,
        context_features=simulator.parameter_dim(),
        dropout_probability=args.dropout,
        tail_bound=args.splinerange,
        num_bins=args.splinebins,
        use_batch_norm=args.batchnorm,
    )
    model = EncoderManifoldFlow(
        data_dim=args.datadim,
        latent_dim=args.modellatentdim,
        encoder=encoder,
        outer_transform=outer_transform,
        inner_transform=inner_transform,
        apply_context_to_outer=args.conditionalouter,
        pie_epsilon=args.pieepsilon,
    )
    return model


def create_image_emf_unstructured(args, c, h, simulator, w):
    if simulator.parameter_dim() is not None and args.conditionalouter:
        raise NotImplementedError
    steps_per_level = (args.outerlayers) // args.levels
    logger.info(
        "Creating manifold flow + encoder for image data with %s levels and %s steps per level in the outer transformation, %s layers in the inner transformation, transforms %s / %s, %s context features",
        args.levels,
        steps_per_level,
        args.innerlayers,
        args.outertransform,
        args.innertransform,
        simulator.parameter_dim(),
    )
    spline_params = {
        "apply_unconditional_transform": False,
        "min_bin_height": 0.001,
        "min_bin_width": 0.001,
        "min_derivative": 0.001,
        "num_bins": args.splinebins,
        "tail_bound": args.splinerange,
    }
    encoder = create_image_encoder(
        c,
        h,
        w,
        args.modellatentdim,
        args.encoderhidden,
        args.encoderblocks,
        dropout_probability=args.dropout,
        context_features=simulator.parameter_dim() if args.conditionalouter else None,
        use_batch_norm=args.batchnorm,
    )
    outer_transform = create_image_transform(
        c,
        h,
        w,
        levels=args.levels,
        hidden_channels=100,
        steps_per_level=steps_per_level,
        num_res_blocks=2,
        alpha=0.05,
        num_bits=8,
        preprocessing="glow",
        dropout_prob=args.dropout,
        multi_scale=True,
        spline_params=spline_params,
        postprocessing="partial_mlp",
        postprocessing_layers=args.linlayers,
        postprocessing_channel_factor=args.linchannelfactor,
        use_actnorm=args.actnorm,
        use_batchnorm=args.batchnorm,
    )
    inner_transform = create_vector_transform(
        args.modellatentdim,
        args.innerlayers,
        linear_transform_type=args.lineartransform,
        base_transform_type=args.innertransform,
        context_features=simulator.parameter_dim(),
        dropout_probability=args.dropout,
        tail_bound=args.splinerange,
        num_bins=args.splinebins,
        use_batch_norm=args.batchnorm,
    )
    model = EncoderManifoldFlow(
        data_dim=args.datadim,
        latent_dim=args.modellatentdim,
        encoder=encoder,
        outer_transform=outer_transform,
        inner_transform=inner_transform,
        apply_context_to_outer=args.conditionalouter,
        pie_epsilon=args.pieepsilon,
    )
    return model


def create_image_flow(args, c, h, simulator, w):
    if simulator.parameter_dim() is not None:
        raise NotImplementedError
    steps_per_level = (args.innerlayers + args.outerlayers) // args.levels
    logger.info(
        "Creating standard flow for image data with %s levels and %s steps per level, transform %s, %s context features",
        args.levels,
        steps_per_level,
        args.outertransform,
        simulator.parameter_dim(),
    )
    spline_params = {
        "apply_unconditional_transform": False,
        "min_bin_height": 0.001,
        "min_bin_width": 0.001,
        "min_derivative": 0.001,
        "num_bins": args.splinebins,
        "tail_bound": args.splinerange,
    }
    transform = create_image_transform(
        c,
        h,
        w,
        levels=args.levels,
        hidden_channels=100,
        steps_per_level=steps_per_level,
        num_res_blocks=2,
        alpha=0.05,
        num_bits=8,
        preprocessing="glow",
        dropout_prob=args.dropout,
        multi_scale=True,
        spline_params=spline_params,
        use_batchnorm=args.batchnorm,
        use_actnorm=args.actnorm,
        postprocessing="permutation",
    )
    model = Flow(data_dim=args.datadim, transform=transform)
    return model


def create_vector_flow(args, simulator):
    logger.info(
        "Creating standard flow for vector data with %s layers, transform %s, %s context features",
        args.innerlayers + args.outerlayers,
        args.outertransform,
        simulator.parameter_dim(),
    )
    transform = create_vector_transform(
        args.datadim,
        args.innerlayers + args.outerlayers,
        linear_transform_type=args.lineartransform,
        base_transform_type=args.outertransform,
        context_features=simulator.parameter_dim(),
        dropout_probability=args.dropout,
        tail_bound=args.splinerange,
        num_bins=args.splinebins,
        use_batch_norm=args.batchnorm,
    )
    model = Flow(data_dim=args.datadim, transform=transform)
    return model
