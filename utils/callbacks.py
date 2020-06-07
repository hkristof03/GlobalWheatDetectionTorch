import numpy as np

class EarlyStopping(object):

    def __init__(self, max_epochs_stop, path_save_model):
        # Early stopping initialization
        self.epochs_no_improve = 0
        self.valid_map_min = np.Inf
        self.best_epoch = 0

    def send(self, valid_loss, model, epoch):
        # Save the model if validation loss decreases
        if valid_loss < self.valid_map_min:
            # Save model
            torch.save(model.state_dict(), path_save_model)
            # Track improvement
            self.epochs_no_improve = 0
            self.valid_loss_min = valid_loss
            self.best_epoch = epoch

        # Otherwise increment count of epochs with no improvement
        else:
            self.epochs_no_improve += 1
            # Trigger early stopping
            if self.epochs_no_improve >= self.max_epochs_stop:
                print(
                    f'Early Stopping! Total epochs: {epoch}. Best epoch: '
                    f'{self.best_epoch} with loss: {valid_loss_min:.2f} and '
                    f'acc: {100 * valid_acc:.2f}%'
                )
                total_time = timer() - overall_start
                print(
                    f'{total_time:.2f} total seconds elapsed. '
                    f'{total_time / (epoch+1):.2f} seconds per epoch.'
                )
                # Load the best state dict
                model.load_state_dict(torch.load(save_file_name))
                # Attach the optimizer
                model.optimizer = optimizer
