import argparse
import yaml
from dataloaders import split_2ds, D2Dataset, train_transforms, val_transforms
from torch.utils.data import DataLoader
from models import AttaNet
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary, ModelCheckpoint
import os.path as osp


def main(epochs, batch_size, learning_rate, output_dir, iterations, seed):
    log_dir = osp.join(output_dir, 'logs')
    tb_logger = TensorBoardLogger(save_dir=log_dir)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # checkpoint_callback = ModelCheckpoint(
    #     monitor="mean_val_dice",
    #     filename="vessel_seg-{epoch:02d}-{mean_val_dice:.2f}",
    #     save_last=True,
    #     save_top_k=3,
    #     mode="max",
    #     save_on_train_epoch_end=True
    # )

    train_dict, val_dict = split_2ds(data_config['dataset_dir'], 0.8)
    train_ds = D2Dataset(train_dict, train_transforms)
    val_ds = D2Dataset(val_dict, val_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=16)
    model = AttaNet(**net_config)
    trainer = pl.Trainer(
        gpus=[0],
        max_epochs=epochs,
        logger=tb_logger,
        # checkpoint_callback=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        # callbacks=[]
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SSNet')
    parser.add_argument('--config', default='configs\Atta_config.yaml')
    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    data_config = config['data_config']
    train_config = config['train_config']
    net_config = config['net_config']
