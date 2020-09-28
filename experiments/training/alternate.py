import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from .trainer import BaseTrainer, logger, NanException, EarlyStoppingException


class AlternatingTrainer(BaseTrainer):
    """ Alternating trainer: takes a number of trainers and alternates between them

     Many apologies for anyone reading this code -- this was an afterthought and not planned for in the original class design """

    def __init__(self, model, *trainers, run_on_gpu=True, multi_gpu=True, double_precision=False):
        super().__init__(model, run_on_gpu, multi_gpu, double_precision)

        assert len(trainers) > 0
        for trainer in trainers:
            assert trainer.model == self.model

        self.trainers = trainers

        logger.debug("Initiated alternating trainer based on %s individual trainers", len(trainers))

    def train(
        self,
        dataset,
        loss_functions,
        loss_function_trainers,
        loss_weights=None,
        loss_labels=None,
        epochs=50,
        subsets=1,
        batch_sizes=100,
        optimizer=optim.Adam,
        optimizer_kwargs=None,
        initial_lr=1.0e-3,
        scheduler=optim.lr_scheduler.CosineAnnealingLR,
        scheduler_kwargs=None,
        restart_scheduler=None,
        validation_split=0.25,
        early_stopping=True,
        early_stopping_patience=None,
        verbose="some",
        parameters=None,
        callbacks=None,
        trainer_kwargs=None,
        trainer_order=None,
        shuffle_trainer_order=False,
        subset_callbacks=None,
    ):
        """ Start training. """

        # Set up
        loss_function_trainers = np.array(loss_function_trainers, dtype=np.int)
        if trainer_order is None:
            trainer_order = list(range(len(self.trainers)))
        if loss_labels is None:
            loss_labels = [fn.__name__ for fn in loss_functions]
        if trainer_kwargs is None:
            trainer_kwargs = [{} for _ in self.trainers]
        if isinstance(batch_sizes, int):
            batch_sizes = [batch_sizes for _ in self.trainers]

        logger.debug("Initialising training data")
        train_loaders, val_loaders = self.make_dataloaders(dataset, validation_split, batch_sizes, subsets)

        epochs_per_scheduler, opts, parameters, scheds, scheduler, scheduler_kwargs = self._setup_opt_sched(epochs, initial_lr, optimizer, optimizer_kwargs, parameters,
                                                                                                            restart_scheduler, scheduler, scheduler_kwargs)

        best_epoch, best_loss, best_model, early_stopping = self._setup_early_stopping(early_stopping, early_stopping_patience, epochs, validation_split)

        n_epochs_verbose = self._set_verbosity(epochs, verbose)

        n_losses = len(loss_labels)
        loss_weights = [1.0] * n_losses if loss_weights is None else loss_weights
        losses_train, losses_val = [], []

        logger.debug("Beginning main training loop")

        # Loop over epochs
        for i_epoch in range(epochs):
            logger.debug("Training epoch %s / %s", i_epoch + 1, epochs)

            # LR schedule
            if scheds[0] is not None:
                logger.debug("Learning rate: %s", scheds[0].get_lr()[0])

            loss_train, loss_val = 0.0, 0.0
            loss_contributions_train, loss_contributions_val = np.zeros(n_losses), np.zeros(n_losses)
            batch_counters_train, batch_counters_val = [0] * len(trainer_order), [0] * len(trainer_order)

            try:

                # Loop over subsets of data
                for i_subset in range(subsets):
                    logger.debug("Epoch subset %s / %s", i_subset + 1, subsets)

                    # Loop over phases / trainers
                    for i_tr_unsrt, i_trainer in enumerate(trainer_order):
                        trainer = self.trainers[i_trainer]
                        opt = opts[i_trainer]
                        trainer_kwargs_ = trainer_kwargs[i_trainer]
                        train_loader = train_loaders[i_trainer][i_subset]
                        val_loader = val_loaders[i_trainer][i_subset]
                        loss_filter = np.argwhere(loss_function_trainers == i_trainer).flatten()
                        batch_counter_train = batch_counters_train[i_trainer]
                        batch_counter_val = batch_counters_val[i_trainer]
                        trainer_parameters = parameters[i_trainer]

                        assert len(loss_filter), "Didn't find losses matching trainer {}: input {} -> filter {}".format(i_trainer, loss_function_trainers, loss_filter)

                        # Number of batches for this subset
                        logger.debug(
                            "Trainer %s / %s: %s (%s) batches", i_tr_unsrt + 1, len(self.trainers), len(train_loader), "-" if val_loader is None else len(val_loader),
                        )

                        # Train
                        loss_train_trainer, loss_val_trainer, loss_contributions_train_trainer, loss_contributions_val_trainer = trainer.partial_epoch(
                            i_epoch=i_epoch,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            optimizer=opt,
                            loss_functions=[loss_functions[i] for i in loss_filter],
                            loss_weights=[loss_weights[i] for i in loss_filter],
                            parameters=trainer_parameters,
                            i_batch_start_train=batch_counter_train,
                            i_batch_start_val=batch_counter_val,
                            **trainer_kwargs_
                        )

                        # Keep track of losses
                        loss_train += loss_train_trainer / subsets
                        loss_contributions_train[loss_filter] += loss_contributions_train_trainer / subsets
                        if loss_val_trainer is not None:
                            loss_val += loss_val_trainer / subsets
                            loss_contributions_val[loss_filter] += loss_contributions_val_trainer / subsets

                        # Phase callbacks
                        if subset_callbacks is not None:
                            for callback in subset_callbacks:
                                callback(i_epoch, self.model, loss_train, loss_val, subset=i_subset, trainer=i_trainer)

                # Wrap up epoch
                losses_train.append(loss_train)
                losses_val.append(loss_val)

            except NanException:
                logger.info("Ending training during epoch %s because NaNs appeared", i_epoch + 1)
                break

            # Wrap up epoch
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
            for i_trainer, sched in enumerate(scheds):
                if sched is not None:
                    sched.step(i_epoch)
                    if restart_scheduler is not None and (i_epoch + 1) % restart_scheduler == 0:
                        try:
                            scheds[i_trainer] = scheduler(optimizer=opts[i_trainer], T_max=epochs_per_scheduler, **scheduler_kwargs)
                        except:
                            scheds[i_trainer] = scheduler(optimizer=opts[i_trainer], **scheduler_kwargs)

            # Shuffle trainer order
            if shuffle_trainer_order:
                np.random.shuffle(trainer_order)

        if early_stopping and len(losses_val) > 0:
            self.wrap_up_early_stopping(best_model, losses_val[-1], best_loss, best_epoch)

        logger.debug("Training finished")

        return np.array(losses_train), np.array(losses_val)

    def _setup_early_stopping(self, early_stopping, early_stopping_patience, epochs, validation_split):
        early_stopping = early_stopping and (validation_split is not None) and (epochs > 1)
        best_loss, best_model, best_epoch = None, None, None
        if early_stopping and early_stopping_patience is None:
            logger.debug("Using early stopping with infinite patience")
        elif early_stopping:
            logger.debug("Using early stopping with patience %s", early_stopping_patience)
        else:
            logger.debug("No early stopping")
        return best_epoch, best_loss, best_model, early_stopping

    def _setup_opt_sched(self, epochs, initial_lr, optimizer, optimizer_kwargs, parameters, restart_scheduler, scheduler, scheduler_kwargs):
        logger.debug("Setting up optimizer")
        optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        if parameters is None:
            parameters = [None for _ in self.trainers]
        opts = []

        for i, parameters_ in enumerate(parameters):
            if parameters_ is None:
                parameters[i] = self.model.parameters()
            opts.append(optimizer(parameters[i], lr=initial_lr, **optimizer_kwargs))

        logger.debug("Setting up LR scheduler")
        if epochs < 2:
            scheduler = None
            logger.debug("Deactivating scheduler for only %s epoch", epochs)
        scheduler_kwargs = {} if scheduler_kwargs is None else scheduler_kwargs
        epochs_per_scheduler = restart_scheduler if restart_scheduler is not None else epochs

        if scheduler is None:
            scheds = [None for _ in self.trainers]
        else:
            scheds = []
            for opt in opts:
                try:
                    scheds.append(scheduler(optimizer=opt, T_max=epochs_per_scheduler, **scheduler_kwargs))
                except:
                    scheds.append(scheduler(optimizer=opt, **scheduler_kwargs))

        return epochs_per_scheduler, opts, parameters, scheds, scheduler, scheduler_kwargs

    @staticmethod
    def make_dataloaders(dataset, validation_split, batch_sizes, subsets):
        if isinstance(batch_sizes, int):
            batch_sizes = [batch_sizes]

        all_train_loaders, all_val_loaders = [], []

        for batch_size in batch_sizes:

            # Prepare split
            n_samples = len(dataset)
            indices = list(range(n_samples))
            if validation_split is not None and 0.0 < validation_split < 1.0:
                split = int(np.floor(validation_split * n_samples))
            else:
                split = 0
            np.random.shuffle(indices)
            train_idx, valid_idx = indices[split:], indices[:split]

            train_loaders, val_loaders = [], []

            # Make train loaders
            last_split = 0
            for subset in range(subsets):
                this_split = n_samples if subset == subsets - 1 else last_split + int(np.round(len(train_idx) / subsets))
                idx = train_idx[last_split:this_split]
                last_split = this_split

                train_loaders.append(DataLoader(dataset, sampler=SubsetRandomSampler(idx), batch_size=batch_size, num_workers=4))

            all_train_loaders.append(train_loaders)

            # Make val loaders
            last_split = 0
            for subset in range(subsets):
                if split > 0:
                    this_split = n_samples if subset == subsets - 1 else last_split + int(np.round(len(valid_idx) / subsets))
                    idx = valid_idx[last_split:this_split]
                    last_split = this_split

                    val_loaders.append(DataLoader(dataset, sampler=SubsetRandomSampler(idx), batch_size=batch_size, num_workers=4))
                else:
                    val_loaders.append(None)

            all_val_loaders.append(val_loaders)

        return all_train_loaders, all_val_loaders
