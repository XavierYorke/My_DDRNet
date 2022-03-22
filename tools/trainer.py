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
        images, labels = batch
        result = self.model(images)
        loss = self.lossfn(result, labels)
        self.log('val_loss', loss)

        plot_2d_or_3d_image(tag="result", data=result[0], step=self.current_epoch, writer=self.logger.experiment,
                            frame_dim=-1)
        plot_2d_or_3d_image(tag="image", data=images, step=self.current_epoch, writer=self.logger.experiment,
                            frame_dim=-1)
        plot_2d_or_3d_image(tag="label", data=labels, step=self.current_epoch, writer=self.logger.experiment,
                            frame_dim=-1)

        result = result[0]
        dice, precision, recall, tnr = [], [], [], []
        val_number = len(result)
        for index in range(val_number):
            image = result[index].cpu().detach().numpy()
            label = labels[index].cpu().detach().numpy()

            image[image >= 0.5] = 1
            image[image < 0.5] = 0
            label[label >= 0.5] = 1
            label[label < 0.5] = 0

            # Dice系数是一种集合相似度度量函数，通常用于计算两个样本的相似度，取值范围在[0,1]
            dice.append(metric.dc(image.astype(np.uint8), label.astype(np.uint8)))
            # 预测正确的个数占总的正类预测个数的比例（从预测结果角度看，有多少预测是准确的）
            precision.append(metric.precision(image.astype(np.uint8), label.astype(np.uint8)))
            # 确定了正类被预测为正类占所有标注的个数（从标注角度看，有多少被召回）
            recall.append(metric.recall(image.astype(np.uint8), label.astype(np.uint8)))
            # 真负类率(True Negative Rate),所有真实负类中，模型预测正确负类的比例
            tnr.append(metric.true_negative_rate(image.astype(np.uint8), label.astype(np.uint8)))

        return {'val_loss': loss,
                'dice': np.mean(dice), 'precision': np.mean(precision),
                'recall': np.mean(recall), 'tnr': np.mean(tnr)
                }

    def training_epoch_end(self, step_outputs):
        loss = 0.
        for result in step_outputs:
            loss += result['loss']
        loss = loss / len(step_outputs)


    def validation_epoch_end(self, step_outputs):
        loss = 0.
        dices, precisions, recalls, tnrs = [], [], [], []
        # val_number = 0
        size = len(step_outputs)
        for result in step_outputs:
            loss += result['val_loss'].sum().item()
            dices.append(result['dice'])
            precisions.append(result['precision'])
            recalls.append(result['recall'])
            tnrs.append(result['tnr'])
            # val_number += result['val_number']

        loss = loss / size
        dice = np.mean(dices)
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        tnr = np.mean(tnrs)

        self.print('\ndice: {}\tprecision: {}\trecall: {}\ttnr: {}\n'.format(dice, precision, recall, tnr))
        self.logger.experiment.add_scalar('dice', dice, global_step=self.current_epoch)
        self.logger.experiment.add_scalar('precision', precision, global_step=self.current_epoch)
        self.logger.experiment.add_scalar('recall', recall, global_step=self.current_epoch)
        self.logger.experiment.add_scalar('tnr', tnr, global_step=self.current_epoch)
        self.log('mean_val_loss', loss)
        self.log('dice', float(dice))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        return optimizer
