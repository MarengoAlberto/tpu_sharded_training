from torch.utils.tensorboard import SummaryWriter

class TensorBoardVisualizer():
    def __init__(self, logs_dir):
        self._writer = SummaryWriter(logs_dir)

    def update_charts(
            self,
            classes,
            train_metric,
            test_metric,
            train_total_loss,
            test_total_loss,
            train_cls_loss,
            train_loc_loss,
            test_cls_loss,
            test_loc_loss,
            learning_rate,
            epoch):


        # Add Train metrics to tensorboard.
        if train_metric is not None:
            for train_metric_key, train_metric_value in train_metric.items():

                if "map_per_class" in train_metric_key:
                    if train_metric["map_per_class"].ndim == 0: # single class
                        self._writer.add_scalar(f"Per Class AP/train-{classes[0]}", train_metric["map_per_class"].numpy(), epoch)

                    else: # multi-class
                        for i, class_ap in enumerate(test_metric["map_per_class"]):
                            self._writer.add_scalar(f"Per Class AP/train-{classes[i]}", class_ap.numpy(), epoch)

                if train_metric_key in ["map", "map_50", "mar_100"]:
                    self._writer.add_scalar(f"eval metrics/train-{train_metric_key}", train_metric_value.numpy(), epoch)


        # Add Val metrics to tensorboard.
        if test_metric is not None:
            for test_metric_key, test_metric_value in test_metric.items():

                if "map_per_class" in test_metric_key:
                    if test_metric["map_per_class"].ndim == 0: # single class
                        self._writer.add_scalar(f"Per Class AP/{classes[0]}", test_metric["map_per_class"].numpy(), epoch)

                    else: # multi-class
                        for i, class_ap in enumerate(test_metric["map_per_class"]):
                            self._writer.add_scalar(f"Per Class AP/{classes[i]}", class_ap.numpy(), epoch)

                if test_metric_key in ["map", "map_50", "mar_100"]:
                    self._writer.add_scalar(f"eval metrics/{test_metric_key}", test_metric_value.numpy(), epoch)


        # Add train and/or val loss to tensorbord.
        if (train_total_loss is not None) and (test_total_loss is not None):
            self._writer.add_scalars("losses/total-loss", {'train': train_total_loss, 'test': test_total_loss}, epoch)
        elif train_total_loss is not None:
            self._writer.add_scalar("losses/train-total-loss", train_total_loss, epoch)
        elif test_total_loss is not None:
            self._writer.add_scalar("losses/test-total-loss", test_total_loss, epoch)


        # Add train and/or val cls_loss to tensorbord.
        if (train_cls_loss is not None) and (test_cls_loss is not None):
            self._writer.add_scalars("losses/classification loss", {'train_cls': train_cls_loss, 'test_cls': test_cls_loss}, epoch)
        elif train_cls_loss is not None:
            self._writer.add_scalar("losses/train-classification loss", train_cls_loss, epoch)
        elif test_cls_loss is not None:
            self._writer.add_scalar("losses/test-classification loss", test_cls_loss, epoch)


        # Add train and/or val loc_loss to tensorbord.
        if (train_loc_loss is not None) and (test_loc_loss is not None):
            self._writer.add_scalars("losses/regression loss", {'train_loc': train_loc_loss, 'test_loc': test_loc_loss}, epoch)
        elif train_loc_loss is not None:
            self._writer.add_scalar("losses/train-regression loss", train_loc_loss, epoch)
        elif test_loc_loss is not None:
            self._writer.add_scalar("losses/test-regression loss", test_loc_loss, epoch)


        # Add LR to Tensorboard.
        self._writer.add_scalar("learning_rate", learning_rate, epoch)

    def close_tensorboard(self):
        self._writer.close()
