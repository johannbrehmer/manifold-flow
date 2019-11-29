""" Strongly based on https://github.com/gpapamak/snl/blob/master/inference/mcmc.py """

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class MCMC_Sampler:
    """
    Superclass for MCMC samplers.
    """

    def __init__(self, x, lp_f, thin):
        """
        :param x: initial state
        :param lp_f: function that returns the log prob
        :param thin: amount of thinning; if None, no thinning
        """

        self.x = np.array(x, dtype=float)
        self.lp_f = lp_f
        self.L = lp_f(self.x)
        self.thin = 1 if thin is None else thin
        self.n_dims = self.x.size if self.x.ndim == 1 else self.x.shape[1]

    def set_state(self, x):
        """
        Sets the state of the chain to x.
        """

        self.x = np.array(x, dtype=float)
        self.L = self.lp_f(self.x)

    def gen(self, n_samples):
        """
        Generates MCMC samples. Should be implemented in a subclass.
        """
        raise NotImplementedError('Should be implemented as a subclass.')


class GaussianMetropolis(MCMC_Sampler):
    """
    Metropolis algorithm with isotropic gaussian proposal.
    """

    def __init__(self, x, lp_f, step, thin=None):
        """
        :param x: initial state
        :param lp_f: function that returns the log prob
        :param step: std of gaussian proposal
        :param thin: amount of thinning; if None, no thinning
        """

        MCMC_Sampler.__init__(self, x, lp_f, thin)
        self.step = step

    def gen(self, n_samples, show_info=False, rng=np.random):
        """
        :param n_samples: number of samples
        :param show_info: whether to plot info at the end of sampling
        :param rng: random number generator to use
        :return: numpy array of samples
        """

        assert n_samples >= 0, 'number of samples can''t be negative'

        n_acc = 0
        L_trace = []
        acc_rate_trace = []
        samples = np.empty([n_samples, self.n_dims])

        for n in range(n_samples):

            for _ in range(self.thin):

                # proposal
                x = self.x + self.step * rng.randn(*self.x.shape)

                # metropolis acceptance step
                L_new = self.lp_f(x)
                if rng.rand() < np.exp(L_new - self.L):
                    self.x = x
                    self.L = L_new
                    n_acc += 1

            samples[n] = self.x

            # acceptance rate
            acc_rate = n_acc / float(self.thin * (n+1))
            logger.debug('MCMC: sample = {0}, acc rate = {1:.2%}, log prob = {2:.2}\n'.format(n+1, acc_rate, self.L))

            # record traces
            if show_info:
                L_trace.append(self.L)
                acc_rate_trace.append(acc_rate)

        # show plot with the traces
        if show_info:
            fig, ax = plt.subplots(2, 1, sharex=True)
            ax[0].plot(L_trace)
            ax[0].set_ylabel('log probability')
            ax[1].plot(acc_rate_trace)
            ax[1].set_ylim([0, 1])
            ax[1].set_ylabel('acceptance rate')
            ax[1].set_xlabel('samples')
            plt.show(block=False)

        return samples


class SliceSampler(MCMC_Sampler):
    """
    Slice sampling for multivariate continuous probability distributions.
    It cycles sampling from each conditional using univariate slice sampling.
    """

    def __init__(self, x, lp_f, max_width=float('inf'), thin=None):
        """
        :param x: initial state
        :param lp_f: function that returns the log prob
        :param max_width: maximum bracket width
        :param thin: amount of thinning; if None, no thinning
        """

        MCMC_Sampler.__init__(self, x, lp_f, thin)
        self.max_width = max_width
        self.width = None

    def gen(self, n_samples, show_info=False, rng=np.random):
        """
        :param n_samples: number of samples
        :param show_info: whether to plot info at the end of sampling
        :param rng: random number generator to use
        :return: numpy array of samples
        """

        assert n_samples >= 0, 'number of samples can''t be negative'

        order = range(self.n_dims)
        L_trace = []
        samples = np.empty([n_samples, self.n_dims])

        if self.width is None:
            logger.debug('Tuning bracket width')
            self._tune_bracket_width(rng)

        for n in range(n_samples):

            for _ in range(self.thin):

                rng.shuffle(order)

                for i in order:
                    self.x[i], _ = self._sample_from_conditional(i, self.x[i], rng)

            samples[n] = self.x.copy()

            self.L = self.lp_f(self.x)
            logger.debug('MCMC: sample = {0}, log prob = {1:.2}\n'.format(n+1, self.L))

            if show_info:
                L_trace.append(self.L)

        # show trace plot
        if show_info:
            fig, ax = plt.subplots(1, 1)
            ax.plot(L_trace)
            ax.set_ylabel('log probability')
            ax.set_xlabel('samples')
            plt.show(block=False)

        return samples

    def _tune_bracket_width(self, rng):
        """
        Initial test run for tuning bracket width.
        Note that this is not correct sampling; samples are thrown away.
        :param rng: random number generator to use
        """

        n_samples = 50
        order = list(range(self.n_dims))
        x = self.x.copy()
        self.width = np.full(self.n_dims, 0.01)

        for n in range(n_samples):

            rng.shuffle(order)

            for i in range(self.n_dims):
                x[i], wi = self._sample_from_conditional(i, x[i], rng)
                self.width[i] += (wi - self.width[i]) / (n + 1)

    def _sample_from_conditional(self, i, cxi, rng):
        """
        Samples uniformly from conditional by constructing a bracket.
        :param i: conditional to sample from
        :param cxi: current state of variable to sample
        :param rng: random number generator to use
        :return: new state, final bracket width
        """

        # conditional log prob
        Li = lambda t: self.lp_f(np.concatenate([self.x[:i], [t], self.x[i+1:]]))
        wi = self.width[i]

        # sample a slice uniformly
        logu = Li(cxi) + np.log(1.0 - rng.rand())

        # position the bracket randomly around the current sample
        lx = cxi - wi * rng.rand()
        ux = lx + wi

        # find lower bracket end
        while Li(lx) >= logu and cxi - lx < self.max_width:
            lx -= wi

        # find upper bracket end
        while Li(ux) >= logu and ux - cxi < self.max_width:
            ux += wi

        # sample uniformly from bracket
        xi = (ux - lx) * rng.rand() + lx

        # if outside slice, reject sample and shrink bracket
        while Li(xi) < logu:
            if xi < cxi:
                lx = xi
            else:
                ux = xi
            xi = (ux - lx) * rng.rand() + lx

        return xi, ux - lx
