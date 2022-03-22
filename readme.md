## 基于pytorch-lightning的语义分割模型复现

### 项目结构
````
├─configs
│  └─Atta_config.yaml
├─dataloaders
│  └─D2Dataset.py
│  └─split_ds.py
├─loss
│  └─loss.py
├─models
│  ├─AttaNet
│  │  └─AttaNet.py
│  │  └─backbone.py
│  │  └─resnet18-5c106cde.pth
├─outputs
│  └─logs
└─tools
│  └─trainer
└─main.py
````
### 数据目录格式
```
├─Datasets
│  └─Data
│      ├─images
│      │  ├─1
│      │  └─2
│      └─labels
│          ├─1
│          └─2
```
### 训练
```bash
python main.py
```
### 可视化
```bash
tensorboard --port 8080 --logdir outputs/logs/default/version_0
```
