import logging
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils import clip_grad_norm_
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


class EarlyStoppingException(Exception):
    pass


class NanException(Exception):
    pass


class NumpyDataset(Dataset):
    """ Dataset for numpy arrays with explicit memmap support """

    def __init__(self, *arrays, **kwargs):

        self.dtype = kwargs.get("dtype", torch.float)
        self.memmap = []
        self.data = []
        self.n = None

        for array in arrays:
            if self.n is None:
                self.n = array.shape[0]
            assert array.shape[0] == self.n

            if isinstance(array, np.memmap):
                self.memmap.append(True)
                self.data.append(array)
            else:
                self.memmap.append(False)
                tensor = torch.from_numpy(array).to(self.dtype)
                self.data.append(tensor)

    def __getitem__(self, index):
        items = []
        for memmap, array in zip(self.memmap, self.data):
            if memmap:
                tensor = np.array(array[index])
                items.append(torch.from_numpy(tensor).to(self.dtype))
            else:
                items.append(array[index])
        return tuple(items)

    def __len__(self):
        return self.n


class Trainer(object):
    """ Trainer class. Any subclass has to implement the forward_pass() function. """

    def __init__(self, model, run_on_gpu=True, double_precision=False):
        self.model = model

        self.run_on_gpu = run_on_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.run_on_gpu else "cpu")
        self.dtype = torch.double if double_precision else torch.float
        if self.run_on_gpu and double_precision:
            torch.set_default_tensor_type('torch.cuda.DoubleTensor')
        elif self.run_on_gpu:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        elif double_precision:
            torch.set_default_tensor_type('torch.DoubleTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        self.model = self.model.to(self.device, self.dtype)

        logger.debug(
            "Training on %s with %s precision",
            "GPU" if self.run_on_gpu else "CPU",
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
        initial_lr=0.001,
        final_lr=0.0001,
        validation_split=0.25,
        early_stopping=True,
        early_stopping_patience=None,
        clip_gradient=100.0,
        verbose="some",
        parameters=None,
        callbacks=None
    ):
        logger.debug("Initialising training data")
        train_loader, val_loader = self.make_dataloader(
            dataset, validation_split, batch_size
        )

        logger.debug("Setting up optimizer")
        optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        if parameters is None:
            parameters = self.model.parameters()
        opt = optimizer(parameters, lr=initial_lr, **optimizer_kwargs)

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

            if callbacks is not None:
                for callback in callbacks:
                    callback(i_epoch, self.model, loss_train, loss_val)

        if early_stopping and len(losses_val) > 0:
            self.wrap_up_early_stopping(
                best_model, losses_val[-1], best_loss, best_epoch
            )

        logger.debug("Training finished")

        return np.array(losses_train), np.array(losses_val)

    def make_dataloader(self, dataset, validation_split, batch_size):
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
            batch_loss, batch_loss_contributions = self.batch_train(
                batch_data, loss_functions, loss_weights, optimizer, clip_gradient
            )
            loss_train += batch_loss
            for i, batch_loss_contribution in enumerate(batch_loss_contributions):
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
                    batch_data, loss_functions, loss_weights
                )
                loss_val += batch_loss
                for i, batch_loss_contribution in enumerate(batch_loss_contributions):
                    loss_contributions_val[i] += batch_loss_contribution

                self.report_batch(i_epoch, i_batch, False, batch_data)

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

    def report_batch(self, i_epoch, i_batch, train, batch_data):
        pass

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


class AutoencodingFlowTrainer(Trainer):
    def __init__(
        self, model, run_on_gpu=True, double_precision=False, output_filename=None
    ):
        super().__init__(model, run_on_gpu, double_precision)
        self.output_filename = output_filename

    def forward_pass(self, batch_data, loss_functions):
        x, y = batch_data
        if len(x.size()) < 2:
            x = x.view(x.size(0), -1)
        x = x.to(self.device, self.dtype)
        x_reco, log_prob, _ = self.model(x)
        losses = [loss_fn(x_reco, x, log_prob) for loss_fn in loss_functions]
        return losses

    # def report_batch(self, i_epoch, i_batch, train, batch_data):
    #     if i_batch > 0 or (not train) or self.output_filename is None:
    #         return
    #
    #     x, y = batch_data
    #     x = x.view(x.size(0), -1)
    #     x = x.to(self.device, self.dtype)
    #     x_out, _, u = self.model(x)
    #
    #     x = x.detach().numpy().reshape(-1, 28, 28)
    #     x_out = x_out.detach().numpy().reshape(-1, 28, 28)
    #     u = u.detach().numpy().reshape(x_out.shape[0], -1)
    #     y = y.detach().numpy().astype(np.int).reshape(-1)
    #     tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300).fit_transform(
    #         u
    #     )
    #
    #     plt.figure(figsize=(5, 5))
    #     for i in range(10):
    #         plt.scatter(
    #             u[y == i][:, 0],
    #             u[y == i][:, 1],
    #             s=15.0,
    #             alpha=1.0,
    #             label="{}".format(i + 1),
    #         )
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig("{}_latent_epoch{}.pdf".format(self.output_filename, i_epoch))
    #     plt.close()
    #
    #     plt.figure(figsize=(5, 5))
    #     for i in range(10):
    #         plt.scatter(
    #             tsne[y == i][:, 0],
    #             tsne[y == i][:, 1],
    #             s=15.0,
    #             alpha=1.0,
    #             label="{}".format(i + 1),
    #         )
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig("{}_latent_tsne_epoch{}.pdf".format(self.output_filename, i_epoch))
    #     plt.close()
    #
    #     plt.figure(figsize=(10, 10))
    #     for i in range(8):
    #         plt.subplot(4, 4, 2 * i + 1)
    #         plt.imshow(x[i, :, :], vmin=-1.1, vmax=1.1)
    #         plt.gca().get_xaxis().set_visible(False)
    #         plt.gca().get_yaxis().set_visible(False)
    #         plt.subplot(4, 4, 2 * i + 2)
    #         plt.imshow(x_out[i, :, :], vmin=-1.1, vmax=1.1)
    #         plt.gca().get_xaxis().set_visible(False)
    #         plt.gca().get_yaxis().set_visible(False)
    #     plt.tight_layout()
    #     plt.savefig(
    #         "{}_reconstruction_epoch{}.pdf".format(self.output_filename, i_epoch)
    #     )
    #     plt.close()
