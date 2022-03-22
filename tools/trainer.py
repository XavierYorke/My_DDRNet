import torch
import pytorch_lightning as pl
from models import AttaNet
from loss import Tverskyloss
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from medpy import metric
import numpy as np


class TrainingModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.learning_rate = self.config['learning_rate']
        self.model = AttaNet(config['n_classes'])
        self.lossfn = Tverskyloss()

    def training_step(self, batch, batch_idx):
        image, label = batch
        result = self.model(image)
        loss = self.lossfn(result, label)
        self.log('loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        image, label = batch
        result = self.model(image)
        loss = self.lossfn(result, label)
        self.log('val_loss', loss)

        plot_2d_or_3d_image(tag="result", data=result[0], step=self.current_epoch, writer=self.logger.experiment,
                            frame_dim=-1)
        plot_2d_or_3d_image(tag="image", data=image, step=self.current_epoch, writer=self.logger.experiment,
                            frame_dim=-1)
        plot_2d_or_3d_image(tag="label", data=label, step=self.current_epoch, writer=self.logger.experiment,
                            frame_dim=-1)

        image = result[0].cpu().detach().numpy()
        label = label.cpu().detach().numpy()

        image[image >= 0.5] = 1
        image[image < 0.5] = 0
        label[label >= 0.5] = 1
        label[label < 0.5] = 0

        # Dice系数是一种集合相似度度量函数，通常用于计算两个样本的相似度，取值范围在[0,1]
        dice = metric.dc(image.astype(np.uint8), label.astype(np.uint8))
        # 预测正确的个数占总的正类预测个数的比例（从预测结果角度看，有多少预测是准确的）
        precision = metric.precision(image.astype(np.uint8), label.astype(np.uint8))
        # 确定了正类被预测为正类占所有标注的个数（从标注角度看，有多少被召回）
        recall = metric.recall(image.astype(np.uint8), label.astype(np.uint8))
        # 真负类率(True Negative Rate),所有真实负类中，模型预测正确负类的比例
        tnr = metric.true_negative_rate(image.astype(np.uint8), label.astype(np.uint8))

        return {'val_loss': loss, 'val_number': len(result),
                'dice': dice, 'precision': precision,
                'recall': recall, 'tnr': tnr
                }

    def training_epoch_end(self, step_outputs):
        loss = 0.
        for result in step_outputs:
            loss += result['loss']
        loss = loss / len(step_outputs)

        self.print("\ntrain: loss:{}\n".format(loss))
        self.logger.experiment.add_scalar('train/loss', loss, global_step=self.current_epoch)

    def validation_epoch_end(self, step_outputs):
        num_items, loss, dice, precision, recall, tnr = 0, 0., 0., 0., 0., 0.
        for result in step_outputs:
            loss += result['val_loss'].sum().item()
            dice += result['dice']
            precision += result['precision']
            recall += result['recall']
            tnr += result['tnr']
            num_items += result['val_number']

        loss = loss / len(step_outputs)
        dice = dice / num_items
        precision = precision / num_items
        recall = recall / num_items
        tnr = tnr / num_items

        self.print('\ndice: {}\tprecision: {}\trecall: {}\ttnr: {}\n'.format(dice, precision, recall, tnr))
        self.logger.experiment.add_scalar('val/loss', loss, global_step=self.current_epoch)
        self.logger.experiment.add_scalar('dice', dice, global_step=self.current_epoch)
        self.logger.experiment.add_scalar('precision', precision, global_step=self.current_epoch)
        self.logger.experiment.add_scalar('recall', recall, global_step=self.current_epoch)
        self.logger.experiment.add_scalar('tnr', tnr, global_step=self.current_epoch)
        self.log('mean_val_loss', loss)
        self.log('dice', dice)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        return optimizer
