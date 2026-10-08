import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
import numpy as np
import yaml
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, R2Score
from torchmetrics.classification import Accuracy, F1Score, AUROC
from torchmetrics.regression import MeanAbsoluteError, ExplainedVariance, MeanSquaredError, R2Score


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_config(args):
    cfg = edict()
    
    # copy args
    for k,v in args.items():
        cfg[k] = v

    # copy model config
    config_root = cfg.get('config_root', 'configs')
    model_config = load_config(f'{config_root}/{cfg.dataset}/{cfg.model}.yaml')
    for k,v in model_config.items():
        cfg[k] = v
    
    # copy data config
    data_config = load_config(f'{config_root}/{cfg.dataset}/dataset.yaml')
    for k,v in data_config.items():
        cfg[k] = v
        
    cfg.data.seperate_ft_types = False

    
    return cfg
    
def load_dataset(config):
    # Check if dataset has H5 format specified in config
    if hasattr(config.data, 'format') and config.data.format == 'h5':
        from data.dataset import create_h5_dataset_from_config
        # Create datasets for train/val/test splits
        train_set = create_h5_dataset_from_config(config, 'train')
        test_set = create_h5_dataset_from_config(config, 'test')
        val_set = create_h5_dataset_from_config(config, 'val')
        
        # Set feature dimensions and number of features
        if hasattr(train_set, 'field_dims') and train_set.field_dims is not None:
            config.data.feature_dims = {f'{i}': int(train_set.field_dims[i]) for i in range(len(train_set.field_dims))}
            config.data.num_features = len(train_set.field_dims)
        
        return {'train': train_set, 'val': val_set, 'test': test_set}
    raise ValueError('MTT runtime requires pre-exported HDF5 data')

def create_data_loaders(config, return_splits=False):
    dataset = load_dataset(config)
    if return_splits:
        return dataset
    train_dataset, val_dataset, test_dataset = dataset['train'], dataset['val'], dataset['test']

    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, num_workers=4, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, num_workers=4, pin_memory=True, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, num_workers=4, pin_memory=True, shuffle=False)
    return train_loader, val_loader, test_loader

def create_model(config):
    mt_tasks = config.data['tasks'].copy()
    mt_task_out_dim = config.data['task_out_dim'].copy()

    if config.model.name == 'mtt':
        from models.mtt import MTT
        return MTT(tasks=mt_tasks,
                   task_out_dim=mt_task_out_dim,
                   feature_dims=config.data['feature_dims'],
                   tower_hid_dims=config.model['tower_hid_dims'],
                   n_blocks=config.model['n_blocks'],
                   embed_dim=config.model['embed_dim'],
                   ff_hid_dim=config.model['ff_hid_dim'],
                   n_heads=config.model['n_heads'],
                   ff_dropout=config.training['ff_dropout'],
                   att_dropout=config.training['att_dropout'],
                   mask_mode_if=config.model['mask_mode_if'],
                   att_type=config.model['att_type'],
                   multi_token=config.model['multi_token'],
                   rope=config.model['rope'])
    else:
        raise NotImplementedError('This runtime supports only MTT')
    
def get_optimizer(config, model):
    if config.model.type == 'mt':
        if config.training.optimizer == 'adam':
            optimizer = Adam(model.parameters(), config.training.lr, weight_decay=config.training.weight_decay)
        elif config.training.optimizer == 'adamw':
            optimizer = AdamW(model.parameters(), config.training.lr, weight_decay=config.training.weight_decay)
        elif config.training.optimizer == 'sgd':
            optimizer = SGD(model.parameters(), config.training.lr)
        else:
            raise NotImplementedError
        return optimizer  
    elif config.model.type == 'st':
        if config.training.optimizer == 'adam':
            optimizer = [Adam(model[task].parameters(), config.training.lr, weight_decay=config.training.weight_decay) for task in config.data.tasks]
        elif config.training.optimizer == 'adamw':
            optimizer = [AdamW(model[task].parameters(), config.training.lr, weight_decay=config.training.weight_decay) for task in config.data.tasks]
        elif config.training.optimizer == 'sgd':
            optimizer = [SGD(model[task].parameters(), config.training.lr) for task in config.data.tasks]
        else:
            raise NotImplementedError
        return optimizer
    else:
        raise NotImplementedError
    
def get_metrics(config, prefix):
    metrics = {}
    for task in config.data.tasks:
        if config.data.task_type[task] == 'binary':
            task_metrics = MetricCollection([
                Accuracy(task='binary'),
                F1Score(task='binary'),
                AUROC(task='binary')
            ], prefix=f'{prefix}_{task}_')
        elif config.data.task_type[task] == 'classification':
            task_metrics = MetricCollection([
                Accuracy(task='multiclass', num_classes=config.data.task_out_dim[task]),
                F1Score(task='multiclass', num_classes=config.data.task_out_dim[task]),
                AUROC(task='multiclass', num_classes=config.data.task_out_dim[task])
            ], prefix=f'{prefix}_{task}_')
        elif config.data.task_type[task] == 'regression':
            task_metrics = MetricCollection([
                MeanAbsoluteError(),
                ExplainedVariance(),
                R2Score(),
                MeanSquaredError()
            ], prefix=f'{prefix}_{task}_')
        elif config.data.task_type[task] == 'resampled_denoising' or config.data.task_type[task] == 'masked_denoising':
            task_metrics = MetricCollection([
                MeanSquaredError(),
                MeanAbsoluteError(),
            ], prefix=f'{prefix}_{task}_')
        else:
            raise NotImplementedError
        metrics[task] = task_metrics
    metrics = nn.ModuleDict(metrics)
    return metrics      

def model_fit(pred, gt, task_type):
    if task_type == 'classification':
        loss = F.cross_entropy(pred, gt.long())
    elif task_type == 'binary':
        loss = F.binary_cross_entropy(F.sigmoid(pred).squeeze(-1), gt)
    elif task_type == 'regression':
        loss = F.mse_loss(pred.squeeze(-1), gt)
    elif task_type == 'resampled_denoising' or task_type == 'masked_denoising':
        loss = F.mse_loss(pred, gt)
    else:
        raise NotImplementedError
    return loss

def create_run_name(config):
    data = config.data.short_name
    lr = config.training.lr
    
    if config.model.name == 'mtt':
        b = config.model.n_blocks
        h = config.model.n_heads
        e = config.model.embed_dim
        hd = config.model.ff_hid_dim
        thd = config.model.tower_hid_dims
        fdrop = config.training.ff_dropout
        adrop = config.training.att_dropout
        mask_if = config.model.mask_mode_if
        mtk = 'mtk' if config.model.multi_token else 'stk'
        att = config.model.att_type
        rope = 'rope' if config.model.rope else ''
        name = f'{data}_mtt_b{b}_h{h}_e{e}_hd{hd}_thd{thd}_lr{lr}_fdrop{fdrop}_adrop{adrop}_att_{att}_maskif_{mask_if}_{mtk}_{rope}'
        return name
    else:
        raise NotImplementedError('This runtime supports only MTT')
