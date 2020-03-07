import sys
import time
import torch
import numpy as np

__all__ = ['TorchHelper']


class TorchHelper:
    checkpoint_history = []
    early_stop_monitor_vals = []
    best_score = 0
    best_epoch = 0

    def __init__(self):
        self.USE_GPU = torch.cuda.is_available()

    def show_progress(self, current_iter, total_iter, start_time, training_loss, additional_msg=''):
        bar_length = 50
        ratio = current_iter / total_iter
        progress_length = int(ratio * bar_length)
        percents = int(ratio * 100)
        bar = '[' + '=' * (progress_length - 1) + '>' + '-' * (bar_length - progress_length) + ']'

        current_time = time.time()
        # elapsed_time = time.gmtime(current_time - start_time).tm_sec
        elapsed_time = round(current_time - start_time, 0)
        estimated_time_needed = round((elapsed_time / current_iter) * (total_iter - current_iter), 0)

        sys.stdout.write(
            'Iter {}/{}: {} {}%  Loss: {} ETA: {}s, Elapsed: {}s, TLI: {} {} \r\r'.format(current_iter, total_iter, bar,
                                                                                       percents,
                                                                                       round(training_loss, 4),
                                                                                       estimated_time_needed,
                                                                                       elapsed_time,
                                                                                       np.round(
                                                                                           elapsed_time / current_iter,
                                                                                           3), additional_msg))

        if current_iter < total_iter:
            sys.stdout.flush()
        else:
            sys.stdout.write('\n')

    def checkpoint_model(self, model_to_save, optimizer_to_save, path_to_save, current_score, epoch, mode='max'):
        """
        Checkpoints models state after each epoch.

        :param model_to_save:
        :param optimizer_to_save:
        :param path_to_save:
        :param current_score:
        :param epoch:
        :param n_epoch:
        :param mode:
        :return:
        """
        model_state = {'epoch'      : epoch + 1,
                       'model_state': model_to_save.state_dict(),
                       'score'      : current_score,
                       'optimizer'  : optimizer_to_save.state_dict()}

        # Save the model as a regular checkpoint
        torch.save(model_state, path_to_save + 'last.pth'.format(epoch))

        self.checkpoint_history.append(current_score)
        is_best = False

        # If the model is best so far according to the score, save as the best model state
        if ((np.max(self.checkpoint_history) == current_score and mode == 'max') or
                (np.min(self.checkpoint_history) == current_score and mode == 'min')):
            is_best = True
            self.best_score = current_score
            self.best_epoch = epoch
            # print('inside checkpoint', current_score, np.max(self.checkpoint_history))
            # torch.save(model_state, path_to_save + '{}_best.pth'.format(n_epoch))
            torch.save(model_state, path_to_save + 'best.pth')
            print('BEST saved')

        print('Current best', round(max(self.checkpoint_history), 4), 'after epoch {}'.format(self.best_epoch))

        return is_best


    def load_saved_model(self, model, path):
        """
        Load a saved model from dump
        :return:
        """
        # self.active_model.load_state_dict(self.best_model_path)['model_state']
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state'])