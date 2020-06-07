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

            return True

        # Otherwise increment count of epochs with no improvement
        else:
            self.epochs_no_improve += 1
            # Trigger early stopping
            if self.epochs_no_improve >= self.max_epochs_stop:
                print(
                    f'Early Stopping! Total epochs: {epoch}. Best epoch: '
                    f'{self.best_epoch} with Mean Average Precision: '
                    f'{self.valid_map_min:.2f}.'
                )
            return False
