from __future__ import absolute_import, division, print_function, unicode_literals

import six
import logging
from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils import clip_grad_norm_

logger = logging.getLogger(__name__)


class EarlyStoppingException(Exception):
    pass


class NanException(Exception):
    pass


class Trainer(object):
    """ Trainer class. Any subclass has to implement the forward_pass() function. """

    def __init__(self, model, run_on_gpu=True, double_precision=False):
        self.model = model
        self.run_on_gpu = run_on_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.run_on_gpu else "cpu")
        self.dtype = torch.double if double_precision else torch.float

        self.model = self.model.to(self.device, self.dtype)

        logger.debug(
            "Training on %s with %s precision",
            "GPU" if self.run_on_gpu else "CPU",
            "double" if double_precision else "single",
        )

    def train(
        self,
        data,
        loss_functions,
        loss_weights=None,
        loss_labels=None,
        epochs=50,
        batch_size=100,
        optimizer=optim.Adam,
        optimizer_kwargs=None,
        initial_lr=0.001,
        final_lr=0.0001,
        validation_split=0.25,
        early_stopping=True,
        early_stopping_patience=None,
        clip_gradient=100.0,
        verbose="some",
    ):
        logger.debug("Initialising training data")
        self.check_data(data)
        self.report_data(data)
        data_labels, dataset = self.make_dataset(data)
        train_loader, val_loader = self.make_dataloaders(
            dataset, validation_split, batch_size
        )

        logger.debug("Setting up optimizer")
        optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        opt = optimizer(self.model.parameters(), lr=initial_lr, **optimizer_kwargs)

        early_stopping = (
            early_stopping and (validation_split is not None) and (epochs > 1)
        )
        best_loss, best_model, best_epoch = None, None, None
        if early_stopping and early_stopping_patience is None:
            logger.debug("Using early stopping with infinite patience")
        elif early_stopping:
            logger.debug(
                "Using early stopping with patience %s", early_stopping_patience
            )
        else:
            logger.debug("No early stopping")

        n_losses = len(loss_functions)
        loss_weights = [1.0] * n_losses if loss_weights is None else loss_weights

        # Verbosity
        if verbose == "all":  # Print output after every epoch
            n_epochs_verbose = 1
        elif verbose == "many":  # Print output after 2%, 4%, ..., 100% progress
            n_epochs_verbose = max(int(round(epochs / 50, 0)), 1)
        elif verbose == "some":  # Print output after 10%, 20%, ..., 100% progress
            n_epochs_verbose = max(int(round(epochs / 20, 0)), 1)
        elif verbose == "few":  # Print output after 20%, 40%, ..., 100% progress
            n_epochs_verbose = max(int(round(epochs / 5, 0)), 1)
        elif verbose == "none":  # Never print output
            n_epochs_verbose = epochs + 2
        else:
            raise ValueError("Unknown value %s for keyword verbose", verbose)
        logger.debug("Will print training progress every %s epochs", n_epochs_verbose)

        logger.debug("Beginning main training loop")
        losses_train, losses_val = [], []

        # Loop over epochs
        for i_epoch in range(epochs):
            logger.debug("Training epoch %s / %s", i_epoch + 1, epochs)

            lr = self.calculate_lr(i_epoch, epochs, initial_lr, final_lr)
            self.set_lr(opt, lr)
            logger.debug("Learning rate: %s", lr)

            try:
                loss_train, loss_val, loss_contributions_train, loss_contributions_val = self.epoch(
                    i_epoch,
                    data_labels,
                    train_loader,
                    val_loader,
                    opt,
                    loss_functions,
                    loss_weights,
                    clip_gradient,
                )
                losses_train.append(loss_train)
                losses_val.append(loss_val)
            except NanException:
                logger.info(
                    "Ending training during epoch %s because NaNs appeared", i_epoch + 1
                )
                break

            if early_stopping:
                try:
                    best_loss, best_model, best_epoch = self.check_early_stopping(
                        best_loss,
                        best_model,
                        best_epoch,
                        loss_val,
                        i_epoch,
                        early_stopping_patience,
                    )
                except EarlyStoppingException:
                    logger.info(
                        "Early stopping: ending training after %s epochs", i_epoch + 1
                    )
                    break

            verbose_epoch = (i_epoch + 1) % n_epochs_verbose == 0
            self.report_epoch(
                i_epoch,
                loss_labels,
                loss_train,
                loss_val,
                loss_contributions_train,
                loss_contributions_val,
                verbose=verbose_epoch,
            )

        if early_stopping and len(losses_val) > 0:
            self.wrap_up_early_stopping(
                best_model, losses_val[-1], best_loss, best_epoch
            )

        logger.debug("Training finished")

        return np.array(losses_train), np.array(losses_val)

    @staticmethod
    def report_data(data):
        logger.debug("Training data:")
        for key, value in six.iteritems(data):
            logger.debug(
                "  %s: shape %s, first %s, mean %s, min %s, max %s",
                key,
                value.shape,
                value[0],
                np.mean(value, axis=0),
                np.min(value, axis=0),
                np.max(value, axis=0),
            )

    @staticmethod
    def check_data(data):
        pass

    @staticmethod
    def make_dataset(data):
        tensor_data = []
        data_labels = []
        for key, value in six.iteritems(data):
            data_labels.append(key)
            tensor_data.append(torch.from_numpy(value))
        dataset = TensorDataset(*tensor_data)
        return data_labels, dataset

    def make_dataloaders(self, dataset, validation_split, batch_size):
        if validation_split is None or validation_split <= 0.0:
            train_loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, pin_memory=self.run_on_gpu
            )
            val_loader = None

        else:
            assert 0.0 < validation_split < 1.0, "Wrong validation split: {}".format(
                validation_split
            )

            n_samples = len(dataset)
            indices = list(range(n_samples))
            split = int(np.floor(validation_split * n_samples))
            np.random.shuffle(indices)
            train_idx, valid_idx = indices[split:], indices[:split]

            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(valid_idx)

            train_loader = DataLoader(
                dataset,
                sampler=train_sampler,
                batch_size=batch_size,
                pin_memory=self.run_on_gpu,
            )
            val_loader = DataLoader(
                dataset,
                sampler=val_sampler,
                batch_size=batch_size,
                pin_memory=self.run_on_gpu,
            )

        return train_loader, val_loader

    @staticmethod
    def calculate_lr(i_epoch, n_epochs, initial_lr, final_lr):
        if n_epochs == 1:
            return initial_lr
        return initial_lr * (final_lr / initial_lr) ** float(i_epoch / (n_epochs - 1.0))

    @staticmethod
    def set_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def epoch(
        self,
        i_epoch,
        data_labels,
        train_loader,
        val_loader,
        optimizer,
        loss_functions,
        loss_weights,
        clip_gradient=None,
    ):
        n_losses = len(loss_functions)

        self.model.train()
        loss_contributions_train = np.zeros(n_losses)
        loss_train = 0.0

        for i_batch, batch_data in enumerate(train_loader):
            batch_data = OrderedDict(list(zip(data_labels, batch_data)))
            batch_loss, batch_loss_contributions = self.batch_train(
                batch_data, loss_functions, loss_weights, optimizer, clip_gradient
            )
            loss_train += batch_loss
            for i, batch_loss_contribution in enumerate(batch_loss_contributions):
                loss_contributions_train[i] += batch_loss_contribution

        loss_contributions_train /= len(train_loader)
        loss_train /= len(train_loader)

        if val_loader is not None:
            self.model.eval()
            loss_contributions_val = np.zeros(n_losses)
            loss_val = 0.0

            for i_batch, batch_data in enumerate(val_loader):
                batch_data = OrderedDict(list(zip(data_labels, batch_data)))
                batch_loss, batch_loss_contributions = self.batch_val(
                    batch_data, loss_functions, loss_weights
                )
                loss_val += batch_loss
                for i, batch_loss_contribution in enumerate(batch_loss_contributions):
                    loss_contributions_val[i] += batch_loss_contribution

            loss_contributions_val /= len(val_loader)
            loss_val /= len(val_loader)

        else:
            loss_contributions_val = None
            loss_val = None

        return loss_train, loss_val, loss_contributions_train, loss_contributions_val

    def batch_train(
        self, batch_data, loss_functions, loss_weights, optimizer, clip_gradient=None
    ):
        loss_contributions = self.forward_pass(batch_data, loss_functions)
        loss = self.sum_losses(loss_contributions, loss_weights)

        self.optimizer_step(optimizer, loss, clip_gradient)

        loss = loss.item()
        loss_contributions = [contrib.item() for contrib in loss_contributions]
        return loss, loss_contributions

    def batch_val(self, batch_data, loss_functions, loss_weights):
        loss_contributions = self.forward_pass(batch_data, loss_functions)
        loss = self.sum_losses(loss_contributions, loss_weights)

        loss = loss.item()
        loss_contributions = [contrib.item() for contrib in loss_contributions]
        return loss, loss_contributions

    def forward_pass(self, batch_data, loss_functions):
        """
        Forward pass of the model. Needs to be implemented by any subclass.

        Parameters
        ----------
        batch_data : OrderedDict with str keys and Tensor values
            The data of the minibatch.

        loss_functions : list of function
            Loss functions.

        Returns
        -------
        losses : list of Tensor
            Losses as scalar pyTorch tensors.

        """
        raise NotImplementedError

    @staticmethod
    def sum_losses(contributions, weights):
        loss = weights[0] * contributions[0]
        for _w, _l in zip(weights[1:], contributions[1:]):
            loss = loss + _w * _l
        return loss

    def optimizer_step(self, optimizer, loss, clip_gradient):
        optimizer.zero_grad()
        loss.backward()
        if clip_gradient is not None:
            clip_grad_norm_(self.model.parameters(), clip_gradient)
        optimizer.step()

    def check_early_stopping(
        self,
        best_loss,
        best_model,
        best_epoch,
        loss,
        i_epoch,
        early_stopping_patience=None,
    ):
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_model = self.model.state_dict()
            best_epoch = i_epoch

        if (
            early_stopping_patience is not None
            and i_epoch - best_epoch > early_stopping_patience >= 0
        ):
            raise EarlyStoppingException

        return best_loss, best_model, best_epoch

    @staticmethod
    def report_epoch(
        i_epoch,
        loss_labels,
        loss_train,
        loss_val,
        loss_contributions_train,
        loss_contributions_val,
        verbose=False,
    ):
        logging_fn = logger.info if verbose else logger.debug

        def contribution_summary(labels, contributions):
            summary = ""
            for i, (label, value) in enumerate(zip(labels, contributions)):
                if i > 0:
                    summary += ", "
                summary += "{}: {:>6.3f}".format(label, value)
            return summary

        train_report = "Epoch {:>3d}: train loss {:>8.5f} ({})".format(
            i_epoch + 1,
            loss_train,
            contribution_summary(loss_labels, loss_contributions_train),
        )
        logging_fn(train_report)

        if loss_val is not None:
            val_report = "           val. loss  {:>8.5f} ({})".format(
                loss_val, contribution_summary(loss_labels, loss_contributions_val)
            )
            logging_fn(val_report)

    def wrap_up_early_stopping(self, best_model, currrent_loss, best_loss, best_epoch):
        if currrent_loss is None or best_loss is None:
            logger.warning("Loss is None, cannot wrap up early stopping")
        elif best_loss < currrent_loss:
            logger.info(
                "Early stopping after epoch %s, with loss %8.5f compared to final loss %8.5f",
                best_epoch + 1,
                best_loss,
                currrent_loss,
            )
            self.model.load_state_dict(best_model)
        else:
            logger.info("Early stopping did not improve performance")

    @staticmethod
    def _check_for_nans(label, *tensors):
        for tensor in tensors:
            if tensor is None:
                continue
            if torch.isnan(tensor).any():
                logger.warning(
                    "%s contains NaNs, aborting training! Data:\n%s", label, tensor
                )
                raise NanException


class SingleParameterizedRatioTrainer(Trainer):
    def __init__(self, model, run_on_gpu=True, double_precision=False):
        super(SingleParameterizedRatioTrainer, self).__init__(
            model, run_on_gpu, double_precision
        )
        self.calculate_model_score = True

    def check_data(self, data):
        data_keys = list(data.keys())
        if "x" not in data_keys or "theta" not in data_keys or "y" not in data_keys:
            raise ValueError(
                "Missing required information 'x', 'theta', or 'y' in training data!"
            )

        for key in data_keys:
            if key not in ["x", "theta", "y", "r_xz", "t_xz", "aux"]:
                logger.warning("Unknown key %s in training data! Ignoring it.", key)

        self.calculate_model_score = "t_xz" in data_keys
        if self.calculate_model_score:
            logger.debug("Model score will be calculated")
        else:
            logger.debug("Model score will not be calculated")

    def make_dataset(self, data):
        tensor_data = []
        data_labels = []
        for key, value in six.iteritems(data):
            data_labels.append(key)
            if key == "theta":
                tensor_data.append(torch.tensor(value, requires_grad=True))
            else:
                tensor_data.append(torch.from_numpy(value))
        dataset = TensorDataset(*tensor_data)
        return data_labels, dataset

    def forward_pass(self, batch_data, loss_functions):
        theta = batch_data["theta"].to(self.device, self.dtype)
        x = batch_data["x"].to(self.device, self.dtype)
        y = batch_data["y"].to(self.device, self.dtype)
        try:
            r_xz = batch_data["r_xz"].to(self.device, self.dtype)
        except KeyError:
            r_xz = None
        try:
            t_xz = batch_data["t_xz"].to(self.device, self.dtype)
        except KeyError:
            t_xz = None
        try:
            aux = batch_data["aux"].to(self.device, self.dtype)
        except KeyError:
            aux = None
        self._check_for_nans("Training data", theta, x, y, aux)
        self._check_for_nans("Augmented training data", r_xz, t_xz)

        s_hat, log_r_hat, t_hat, _ = self.model(
            theta,
            x,
            aux=aux,
            track_score=self.calculate_model_score,
            return_grad_x=False,
        )
        self._check_for_nans("Model output (log r)", log_r_hat)
        self._check_for_nans("Model output (s)", s_hat)
        self._check_for_nans("Model output (t)", t_hat)

        losses = [
            loss_function(s_hat, log_r_hat, t_hat, y, r_xz, t_xz)
            for loss_function in loss_functions
        ]
        self._check_for_nans("Loss", *losses)

        return losses
