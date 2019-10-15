import sys
import torch
import logging

sys.path.append("../../")
import manifold_flow as mf

logging.basicConfig(
    format="%(asctime)-5.5s %(name)-30.30s %(levelname)-7.7s %(message)s",
    datefmt="%H:%M",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)
for key in logging.Logger.manager.loggerDict:
    if "manifold_flow" not in key and __name__ not in key:
        logging.getLogger(key).setLevel(logging.WARNING)

n = 100
x0 = torch.randn(n).view(-1,1)
x1 = 0.5*(1.5 + x0)*(1.5-x0)
x = torch.cat([x0,x1],1)

trf = mf.flows.vector_transforms.create_transform(
    dim=2,
    flow_steps=3,
    linear_transform_type="permutation",
    base_transform_type="affine-coupling",
    hidden_features=5,
    num_transform_blocks=2,
    dropout_probability=0.,
    use_batch_norm=False
)

flow = mf.flows.autoencoding_flow.TwoStepAutoencodingFlow(
    data_dim=2,
    latent_dim=1,
    inner_transform=None,
    outer_transform=trf,
)

x_reco_before, log_prob_before, u_before = flow(x)

logger.info("Estimated log prob: %s", log_prob_before.detach().numpy())

x_gen_before = flow.sample(n=n)

logger.info("Generated: %s", x_gen_before.detach().numpy())
