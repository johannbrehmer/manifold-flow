import logging
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils import clip_grad_norm_

# from matplotlib import pyplot as plt
# from sklearn.manifold import TSNE
# from tqdm import tqdm

logger = logging.getLogger(__name__)


class EarlyStoppingException(Exception):
    pass


class NanException(Exception):
    pass


class Trainer(object):
    """ Trainer class. Any subclass has to implement the forward_pass() function. """

    def __init__(self, model, run_on_gpu=True, multi_gpu=True, double_precision=False):
        self.model = model

        self.run_on_gpu = run_on_gpu and torch.cuda.is_available()
        self.multi_gpu = self.run_on_gpu and multi_gpu and torch.cuda.device_count() > 1

        self.device = torch.device("cuda" if self.run_on_gpu else "cpu")
        self.dtype = torch.double if double_precision else torch.float
        if self.run_on_gpu and double_precision:
            torch.set_default_tensor_type("torch.cuda.DoubleTensor")
        elif self.run_on_gpu:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        elif double_precision:
            torch.set_default_tensor_type("torch.DoubleTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")

        self.model = self.model.to(self.device, self.dtype)

        logger.info(
            "Training on %s with %s precision",
            "{} GPUS".format(torch.cuda.device_count()) if self.multi_gpu else "GPU" if self.run_on_gpu else "CPU",
            "double" if double_precision else "single",
        )

    def train(
        self,
        dataset,
        loss_functions,
        loss_weights=None,
        loss_labels=None,
        epochs=50,
        batch_size=100,
        optimizer=optim.Adam,
        optimizer_kwargs=None,
        initial_lr=1.0e-3,
        scheduler=optim.lr_scheduler.CosineAnnealingLR,
        scheduler_kwargs=None,
        restart_scheduler=None,
        validation_split=0.25,
        early_stopping=True,
        early_stopping_patience=None,
        clip_gradient=1.0,
        verbose="some",
        parameters=None,
        callbacks=None,
        forward_kwargs=None,
        custom_kwargs=None,
    ):
        if loss_labels is None:
            loss_labels = [fn.__name__ for fn in loss_functions]

        logger.debug("Initialising training data")
        train_loader, val_loader = self.make_dataloader(dataset, validation_split, batch_size)

        logger.debug("Setting up optimizer")
        optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        if parameters is None:
            parameters = self.model.parameters()
        opt = optimizer(parameters, lr=initial_lr, **optimizer_kwargs)

        logger.debug("Setting up LR scheduler")
        if epochs < 2:
            scheduler = None
            logger.debug("Deactivating scheduler for only %s epoch", epochs)
        scheduler_kwargs = {} if scheduler_kwargs is None else scheduler_kwargs
        sched = None
        epochs_per_scheduler = restart_scheduler if restart_scheduler is not None else epochs
        if scheduler is not None:
            try:
                sched = scheduler(optimizer=opt, T_max=epochs_per_scheduler, **scheduler_kwargs)
            except:
                sched = scheduler(optimizer=opt, **scheduler_kwargs)

        early_stopping = early_stopping and (validation_split is not None) and (epochs > 1)
        best_loss, best_model, best_epoch = None, None, None
        if early_stopping and early_stopping_patience is None:
            logger.debug("Using early stopping with infinite patience")
        elif early_stopping:
            logger.debug("Using early stopping with patience %s", early_stopping_patience)
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
        # for i_epoch in tqdm(range(epochs)):
        for i_epoch in range(epochs):
            logger.debug("Training epoch %s / %s", i_epoch + 1, epochs)

            # LR schedule
            if sched is not None:
                logger.debug("Learning rate: %s", sched.get_lr()[0])

            try:
                loss_train, loss_val, loss_contributions_train, loss_contributions_val = self.epoch(
                    i_epoch,
                    train_loader,
                    val_loader,
                    opt,
                    loss_functions,
                    loss_weights,
                    clip_gradient,
                    forward_kwargs=forward_kwargs,
                    custom_kwargs=custom_kwargs,
                )
                losses_train.append(loss_train)
                losses_val.append(loss_val)
            except NanException:
                logger.info("Ending training during epoch %s because NaNs appeared", i_epoch + 1)
                break

            if early_stopping:
                try:
                    best_loss, best_model, best_epoch = self.check_early_stopping(best_loss, best_model, best_epoch, loss_val, i_epoch, early_stopping_patience)
                except EarlyStoppingException:
                    logger.info("Early stopping: ending training after %s epochs", i_epoch + 1)
                    break

            verbose_epoch = (i_epoch + 1) % n_epochs_verbose == 0
            self.report_epoch(i_epoch, loss_labels, loss_train, loss_val, loss_contributions_train, loss_contributions_val, verbose=verbose_epoch)

            # Callbacks
            if callbacks is not None:
                for callback in callbacks:
                    callback(i_epoch, self.model, loss_train, loss_val)

            # LR scheduler
            if sched is not None:
                sched.step(i_epoch)
                if restart_scheduler is not None and (i_epoch + 1) % restart_scheduler == 0:
                    try:
                        sched = scheduler(optimizer=opt, T_max=epochs_per_scheduler, **scheduler_kwargs)
                    except:
                        sched = scheduler(optimizer=opt, **scheduler_kwargs)

        if early_stopping and len(losses_val) > 0:
            self.wrap_up_early_stopping(best_model, losses_val[-1], best_loss, best_epoch)

        logger.debug("Training finished")

        return np.array(losses_train), np.array(losses_val)

    def make_dataloader(self, dataset, validation_split, batch_size):
        if validation_split is None or validation_split <= 0.0:
            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                # pin_memory=self.run_on_gpu,
                num_workers=10,
            )
            val_loader = None

        else:
            assert 0.0 < validation_split < 1.0, "Wrong validation split: {}".format(validation_split)

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
                # pin_memory=self.run_on_gpu,
                num_workers=10,
            )
            val_loader = DataLoader(
                dataset,
                sampler=val_sampler,
                batch_size=batch_size,
                # pin_memory=self.run_on_gpu,
                num_workers=10,
            )

        return train_loader, val_loader

    def epoch(self, i_epoch, train_loader, val_loader, optimizer, loss_functions, loss_weights, clip_gradient=None, forward_kwargs=None, custom_kwargs=None):
        n_losses = len(loss_functions)

        self.model.train()
        loss_contributions_train = np.zeros(n_losses)
        loss_train = 0.0

        for i_batch, batch_data in enumerate(train_loader):
            if i_batch == 0 and i_epoch == 0:
                self.first_batch(batch_data)
            batch_loss, batch_loss_contributions = self.batch_train(
                batch_data, loss_functions, loss_weights, optimizer, clip_gradient, forward_kwargs=forward_kwargs, custom_kwargs=custom_kwargs
            )
            loss_train += batch_loss
            for i, batch_loss_contribution in enumerate(batch_loss_contributions[:n_losses]):
                loss_contributions_train[i] += batch_loss_contribution

            self.report_batch(i_epoch, i_batch, True, batch_data)

        loss_contributions_train /= len(train_loader)
        loss_train /= len(train_loader)

        if val_loader is not None:
            self.model.eval()
            loss_contributions_val = np.zeros(n_losses)
            loss_val = 0.0

            for i_batch, batch_data in enumerate(val_loader):
                batch_loss, batch_loss_contributions = self.batch_val(
                    batch_data, loss_functions, loss_weights, forward_kwargs=forward_kwargs, custom_kwargs=custom_kwargs
                )
                loss_val += batch_loss
                for i, batch_loss_contribution in enumerate(batch_loss_contributions[:n_losses]):
                    loss_contributions_val[i] += batch_loss_contribution

                self.report_batch(i_epoch, i_batch, False, batch_data)

            loss_contributions_val /= len(val_loader)
            loss_val /= len(val_loader)

        else:
            loss_contributions_val = None
            loss_val = None

        return loss_train, loss_val, loss_contributions_train, loss_contributions_val

    def first_batch(self, batch_data):
        pass

    def batch_train(self, batch_data, loss_functions, loss_weights, optimizer, clip_gradient=None, forward_kwargs=None, custom_kwargs=None):
        loss_contributions = self.forward_pass(batch_data, loss_functions, forward_kwargs=forward_kwargs, custom_kwargs=custom_kwargs)
        loss = self.sum_losses(loss_contributions, loss_weights)

        self.optimizer_step(optimizer, loss, clip_gradient)

        loss = loss.item()
        loss_contributions = [contrib.item() for contrib in loss_contributions]
        return loss, loss_contributions

    def batch_val(self, batch_data, loss_functions, loss_weights, forward_kwargs=None, custom_kwargs=None):
        loss_contributions = self.forward_pass(batch_data, loss_functions, forward_kwargs=forward_kwargs, custom_kwargs=custom_kwargs)
        loss = self.sum_losses(loss_contributions, loss_weights)

        loss = loss.item()
        loss_contributions = [contrib.item() for contrib in loss_contributions]
        return loss, loss_contributions

    def forward_pass(self, batch_data, loss_functions, forward_kwargs=None, custom_kwargs=None):
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

    def check_early_stopping(self, best_loss, best_model, best_epoch, loss, i_epoch, early_stopping_patience=None):
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_model = self.model.state_dict()
            best_epoch = i_epoch

        if early_stopping_patience is not None and i_epoch - best_epoch > early_stopping_patience >= 0:
            raise EarlyStoppingException

        return best_loss, best_model, best_epoch

    def report_batch(self, i_epoch, i_batch, train, batch_data):
        pass

    @staticmethod
    def report_epoch(i_epoch, loss_labels, loss_train, loss_val, loss_contributions_train, loss_contributions_val, verbose=False):
        logging_fn = logger.info if verbose else logger.debug

        def contribution_summary(labels, contributions):
            summary = ""
            for i, (label, value) in enumerate(zip(labels, contributions)):
                if i > 0:
                    summary += ", "
                summary += "{}: {:>6.3f}".format(label, value)
            return summary

        train_report = "Epoch {:>3d}: train loss {:>8.5f} ({})".format(i_epoch + 1, loss_train, contribution_summary(loss_labels, loss_contributions_train))
        logging_fn(train_report)

        if loss_val is not None:
            val_report = "           val. loss  {:>8.5f} ({})".format(loss_val, contribution_summary(loss_labels, loss_contributions_val))
            logging_fn(val_report)

    def wrap_up_early_stopping(self, best_model, currrent_loss, best_loss, best_epoch):
        if currrent_loss is None or best_loss is None:
            logger.warning("Loss is None, cannot wrap up early stopping")
        elif best_loss < currrent_loss:
            logger.info("Early stopping after epoch %s, with loss %8.5f compared to final loss %8.5f", best_epoch + 1, best_loss, currrent_loss)
            self.model.load_state_dict(best_model)
        else:
            logger.info("Early stopping did not improve performance")

    @staticmethod
    def _check_for_nans(label, *tensors, fix_until=None, replace=0.0):
        for tensor in tensors:
            if tensor is None:
                continue

            if torch.isnan(tensor).any():
                if fix_until is not None:
                    n_nans = torch.sum(torch.isnan(tensor))
                    if n_nans <= fix_until:
                        logger.warning("%s contains %s NaNs, setting them to zero", label, n_nans)
                        tensor[torch.isnan(tensor)] = replace
                        return

                logger.warning("%s contains NaNs, aborting training! Data:\n%s", label, tensor)
                raise NanException


class ManifoldFlowTrainer(Trainer):
    def first_batch(self, batch_data):
        if self.multi_gpu:
            x, y = batch_data
            if len(x.size()) < 2:
                x = x.view(x.size(0), -1)
            x = x.to(self.device, self.dtype)
            self.model(x[: x.shape[0] // torch.cuda.device_count(), ...])

    def forward_pass(self, batch_data, loss_functions, forward_kwargs=None, custom_kwargs=None):
        if forward_kwargs is None:
            forward_kwargs = {}

        x, y = batch_data
        self._check_for_nans("Training data", x)

        if len(x.size()) < 2:
            x = x.view(x.size(0), -1)
        x = x.to(self.device, self.dtype)
        if self.multi_gpu:
            x_reco, log_prob, _ = nn.parallel.data_parallel(self.model, x, module_kwargs=forward_kwargs)
        else:
            x_reco, log_prob, _ = self.model(x, **forward_kwargs)
        self._check_for_nans("Reconstructed data", x_reco)
        if log_prob is not None:
            self._check_for_nans("Log likelihood", log_prob, fix_until=2)

        losses = [loss_fn(x_reco, x, log_prob) for loss_fn in loss_functions]
        self._check_for_nans("Loss", *losses)

        return losses


class ConditionalManifoldFlowTrainer(Trainer):
    def forward_pass(self, batch_data, loss_functions, forward_kwargs=None, custom_kwargs=None):
        if forward_kwargs is None:
            forward_kwargs = {}

        x, params = batch_data

        if len(x.size()) < 2:
            x = x.view(x.size(0), -1)
        if len(params.size()) < 2:
            params = params.view(params.size(0), -1)

        x = x.to(self.device, self.dtype)
        params = params.to(self.device, self.dtype)
        self._check_for_nans("Training data", x, params)

        if self.multi_gpu:
            x_reco, log_prob, _ = nn.parallel.data_parallel(self.model, x, module_kwargs={"context": params})
        else:
            x_reco, log_prob, _ = self.model(x, context=params, **forward_kwargs)
        self._check_for_nans("Reconstructed data", x_reco)
        if log_prob is not None:
            self._check_for_nans("Log likelihood", log_prob, fix_until=2)

        losses = [loss_fn(x_reco, x, log_prob) for loss_fn in loss_functions]
        self._check_for_nans("Loss", *losses)

        return losses


class VariableDimensionManifoldFlowTrainer(ManifoldFlowTrainer):
    def train(
        self,
        dataset,
        loss_functions,
        loss_weights=None,
        loss_labels=None,
        epochs=50,
        batch_size=100,
        optimizer=optim.Adam,
        optimizer_kwargs=None,
        initial_lr=1.0e-3,
        scheduler=optim.lr_scheduler.CosineAnnealingLR,
        scheduler_kwargs=None,
        restart_scheduler=None,
        validation_split=0.25,
        early_stopping=True,
        early_stopping_patience=None,
        clip_gradient=1.0,
        verbose="some",
        parameters=None,
        callbacks=None,
        forward_kwargs=None,
        custom_kwargs=None,
        l1=0.0,
        l2=0.0,
    ):
        # Prepare inputs
        if custom_kwargs is None:
            custom_kwargs = dict()
        if l1 is not None:
            custom_kwargs["l1"] = l1
        if l2 is not None:
            custom_kwargs["l2"] = l2

        n_losses = len(loss_functions) + 1
        if loss_labels is None:
            loss_labels = [fn.__name__ for fn in loss_functions]
        loss_labels.append("Regularizer")
        loss_weights = [1.0] * n_losses if loss_weights is None else loss_weights + [1.0]

        super().train(
            dataset,
            loss_functions,
            loss_weights,
            loss_labels,
            epochs,
            batch_size,
            optimizer,
            optimizer_kwargs,
            initial_lr,
            scheduler,
            scheduler_kwargs,
            restart_scheduler,
            validation_split,
            early_stopping,
            early_stopping_patience,
            clip_gradient,
            verbose,
            parameters,
            callbacks,
            forward_kwargs,
            custom_kwargs,
        )

    def forward_pass(self, batch_data, loss_functions, forward_kwargs=None, custom_kwargs=None):
        losses = super().forward_pass(batch_data, loss_functions, forward_kwargs)

        if custom_kwargs is not None:
            l1 = custom_kwargs.get("l1", 0.0)
            l2 = custom_kwargs.get("l2", 0.0)
            reg = self.model.latent_regularizer(l1, l2)
            losses.append(reg)

        return losses

    def report_epoch(self, i_epoch, loss_labels, loss_train, loss_val, loss_contributions_train, loss_contributions_val, verbose=False):
        logging_fn = logger.info if verbose else logger.debug
        super().report_epoch(i_epoch, loss_labels, loss_train, loss_val, loss_contributions_train, loss_contributions_val, verbose)

        logging_fn("           latent dim {:>8d}".format(self.model.calculate_latent_dim()))
        logger.debug("           stds        {}".format(self.model.latent_stds().detach().numpy()))


class ConditionalVariableDimensionManifoldFlowTrainer(ConditionalManifoldFlowTrainer):
    def train(
        self,
        dataset,
        loss_functions,
        loss_weights=None,
        loss_labels=None,
        epochs=50,
        batch_size=100,
        optimizer=optim.Adam,
        optimizer_kwargs=None,
        initial_lr=1.0e-3,
        scheduler=optim.lr_scheduler.CosineAnnealingLR,
        scheduler_kwargs=None,
        restart_scheduler=None,
        validation_split=0.25,
        early_stopping=True,
        early_stopping_patience=None,
        clip_gradient=1.0,
        verbose="some",
        parameters=None,
        callbacks=None,
        forward_kwargs=None,
        custom_kwargs=None,
        l1=0.0,
        l2=0.0,
    ):
        # Prepare inputs
        if custom_kwargs is None:
            custom_kwargs = dict()
        if l1 is not None:
            custom_kwargs["l1"] = l1
        if l2 is not None:
            custom_kwargs["l2"] = l2

        n_losses = len(loss_functions) + 1
        if loss_labels is None:
            loss_labels = [fn.__name__ for fn in loss_functions]
        loss_labels.append("Regularizer")
        loss_weights = [1.0] * n_losses if loss_weights is None else loss_weights + [1.0]

        super().train(
            dataset,
            loss_functions,
            loss_weights,
            loss_labels,
            epochs,
            batch_size,
            optimizer,
            optimizer_kwargs,
            initial_lr,
            scheduler,
            scheduler_kwargs,
            restart_scheduler,
            validation_split,
            early_stopping,
            early_stopping_patience,
            clip_gradient,
            verbose,
            parameters,
            callbacks,
            forward_kwargs,
            custom_kwargs,
        )

    def forward_pass(self, batch_data, loss_functions, forward_kwargs=None, custom_kwargs=None):
        losses = super().forward_pass(batch_data, loss_functions, forward_kwargs)

        if custom_kwargs is not None:
            l1 = custom_kwargs.get("l1", 0.0)
            l2 = custom_kwargs.get("l2", 0.0)
            reg = self.model.latent_regularizer(l1, l2)
            losses.append(reg)

        return losses

    def report_epoch(self, i_epoch, loss_labels, loss_train, loss_val, loss_contributions_train, loss_contributions_val, verbose=False):
        logging_fn = logger.info if verbose else logger.debug
        super().report_epoch(i_epoch, loss_labels, loss_train, loss_val, loss_contributions_train, loss_contributions_val, verbose)

        logging_fn("           latent dim {:>8d}".format(self.model.calculate_latent_dim()))
        logger.debug("           stds        {}".format(self.model.latent_stds().detach().numpy()))


class GenerativeTrainer(Trainer):
    # TODO: multi-GPU support
    def forward_pass(self, batch_data, loss_functions, forward_kwargs=None, custom_kwargs=None):
        if forward_kwargs is None:
            forward_kwargs = {}

        x, y = batch_data
        batch_size = x.size(0)
        if len(x.size()) < 2:
            x = x.view(batch_size, -1)
        x = x.to(self.device, self.dtype)
        self._check_for_nans("Training data", x)

        x_gen = self.model.sample(n=batch_size, **forward_kwargs)
        self._check_for_nans("Generated data", x_gen)

        losses = [loss_fn(x_gen, x, None) for loss_fn in loss_functions]
        self._check_for_nans("Loss", *losses)

        return losses


class ConditionalGenerativeTrainer(GenerativeTrainer):
    # TODO: multi-GPU support
    def forward_pass(self, batch_data, loss_functions, forward_kwargs=None, custom_kwargs=None):
        if forward_kwargs is None:
            forward_kwargs = {}

        x, params = batch_data
        batch_size = x.size(0)

        if len(x.size()) < 2:
            x = x.view(batch_size, -1)
        if len(params.size()) < 2:
            params = params.view(batch_size, -1)
        self._check_for_nans("Training data", x, params)

        x = x.to(self.device, self.dtype)
        params = params.to(self.device, self.dtype)

        x_gen = self.model.sample(n=batch_size, context=params, **forward_kwargs)
        self._check_for_nans("Generated data", x_gen)

        losses = [loss_fn(x_gen, x, None) for loss_fn in loss_functions]
        self._check_for_nans("Loss", *losses)

        return losses


# class ImageManifoldFlowTrainer(ManifoldFlowTrainer):
#     def __init__(
#         self, model, run_on_gpu=True, double_precision=False, output_filename=None
#     ):
#         super().__init__(model, run_on_gpu, double_precision)
#         self.output_filename = output_filename
#
#     def report_batch(self, i_epoch, i_batch, train, batch_data):
#         if i_batch > 0 or (not train) or self.output_filename is None:
#             return
#
#         x, y = batch_data
#         resolution = x.size()[1]
#         x = x.view(x.size(0), -1)
#         x = x.to(self.device, self.dtype)
#         x_out, _, u = self.model(x)
#
#         x = x.detach().numpy().reshape(-1, resolution, resolution)
#         x_out = x_out.detach().numpy().reshape(-1, resolution, resolution)
#         u = u.detach().numpy().reshape(x_out.shape[0], -1)
#         y = y.detach().numpy().astype(np.int).reshape(-1)
#         tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300).fit_transform(u)
#
#         plt.figure(figsize=(5, 5))
#         for i in range(10):
#             plt.scatter(
#                 u[y == i][:, 0],
#                 u[y == i][:, 1],
#                 s=15.0,
#                 alpha=1.0,
#                 label="{}".format(i + 1),
#             )
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig("{}_latent_epoch{}.pdf".format(self.output_filename, i_epoch))
#         plt.close()
#
#         plt.figure(figsize=(5, 5))
#         for i in range(10):
#             plt.scatter(
#                 tsne[y == i][:, 0],
#                 tsne[y == i][:, 1],
#                 s=15.0,
#                 alpha=1.0,
#                 label="{}".format(i + 1),
#             )
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig("{}_latent_tsne_epoch{}.pdf".format(self.output_filename, i_epoch))
#         plt.close()
#
#         plt.figure(figsize=(10, 10))
#         for i in range(8):
#             plt.subplot(4, 4, 2 * i + 1)
#             plt.imshow(x[i, :, :], vmin=-1.1, vmax=1.1)
#             plt.gca().get_xaxis().set_visible(False)
#             plt.gca().get_yaxis().set_visible(False)
#             plt.subplot(4, 4, 2 * i + 2)
#             plt.imshow(x_out[i, :, :], vmin=-1.1, vmax=1.1)
#             plt.gca().get_xaxis().set_visible(False)
#             plt.gca().get_yaxis().set_visible(False)
#         plt.tight_layout()
#         plt.savefig(
#             "{}_reconstruction_epoch{}.pdf".format(self.output_filename, i_epoch)
#         )
#         plt.close()
