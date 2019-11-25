from torch.nn import functional as F

from manifold_flow import transforms, utils
import manifold_flow.nn as nn_


def _create_linear_transform(linear_transform_type, features):
    if linear_transform_type == "permutation":
        return transforms.RandomPermutation(features=features)
    elif linear_transform_type == "lu":
        return transforms.CompositeTransform([transforms.RandomPermutation(features=features), transforms.LULinear(features, identity_init=True)])
    elif linear_transform_type == "svd":
        return transforms.CompositeTransform([transforms.RandomPermutation(features=features), transforms.SVDLinear(features, num_householder=10, identity_init=True)])
    else:
        raise ValueError


def _create_base_transform(
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
    context_features=None,
):
    if base_transform_type == "affine-coupling":
        return transforms.AffineCouplingTransform(
            mask=utils.create_alternating_binary_mask(features, even=(i % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                context_features=context_features,
                num_blocks=num_transform_blocks,
                activation=F.relu,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm,
            ),
        )
    elif base_transform_type == "quadratic-coupling":
        return transforms.PiecewiseQuadraticCouplingTransform(
            mask=utils.create_alternating_binary_mask(features, even=(i % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                context_features=context_features,
                num_blocks=num_transform_blocks,
                activation=F.relu,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm,
            ),
            num_bins=num_bins,
            tails="linear",
            tail_bound=tail_bound,
            apply_unconditional_transform=apply_unconditional_transform,
        )
    elif base_transform_type == "rq-coupling":
        return transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=utils.create_alternating_binary_mask(features, even=(i % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                context_features=context_features,
                num_blocks=num_transform_blocks,
                activation=F.relu,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm,
            ),
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


def create_transform(
    dim,
    flow_steps,
    linear_transform_type="permutation",
    base_transform_type="rq-coupling",
    hidden_features=256,
    num_transform_blocks=2,
    dropout_probability=0.25,
    use_batch_norm=False,
    num_bins=8,
    tail_bound=3,
    apply_unconditional_transform=True,
    context_features=None,
):
    transform = transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    _create_linear_transform(linear_transform_type, dim),
                    _create_base_transform(
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
                    ),
                ]
            )
            for i in range(flow_steps)
        ]
        + [_create_linear_transform(linear_transform_type, dim)]
    )
    return transform
