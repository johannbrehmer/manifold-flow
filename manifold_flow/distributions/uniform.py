from torch import distributions

from manifold_flow.utils import various


class TweakedUniform(distributions.Uniform):
    def log_prob(self, value, context):
        return various.sum_except_batch(super().log_prob(value))
        # result = super().log_prob(value)
        # if len(result.shape) == 2 and result.shape[1] == 1:
        #     return result.reshape(-1)
        # else:
        #     return result

    def sample(self, num_samples, context):
        return super().sample((num_samples,))
