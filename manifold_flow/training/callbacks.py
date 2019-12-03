import torch


def save_model_after_every_epoch(filename):
    def callback(i_epoch, model, loss_train, loss_val):
        torch.save(model.state_dict(), filename.format(i_epoch))

    return callback
