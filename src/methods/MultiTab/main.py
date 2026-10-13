import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping
from config import create_config, create_data_loaders, create_run_name
import argparse
import os
from models.wrappers import MultitaskModel, SingletaskModel
from mtt_refit import run_mtt_selection_and_refit, run_standard_training

torch.set_float32_matmul_precision('high')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser()

# environment configurations
parser.add_argument('--data_root', required=True, type=str, help='root directory of the data')
parser.add_argument('--model', default='mtt', choices=['mtt'], type=str, help='name of the model')
parser.add_argument('--dataset', required=True, type=str, help='external dataset/fold config name')
parser.add_argument('--seed', default=42, type=int, help='random seed for pytorch')
parser.add_argument('--patience', default=5, type=int, help='patience for early stopping')
parser.add_argument('--run_name', default='auto', type=str, help='override the name of the run')
parser.add_argument('--cuda', default=0, type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7], help='gpu index')
parser.add_argument('--log-root', default='logs', type=str, help='isolated Lightning CSV log root')
parser.add_argument('--config-root', default='configs', type=str, help='dataset/model config root')
parser.add_argument('--mtt-protocol', default='selection-refit', choices=['selection-refit', 'standalone'], help='MTT training protocol')


def run_training(
    config, train_loader, val_loader, test_loader, logger,
    mtt_model_factory=MultitaskModel,
    singletask_model_factory=SingletaskModel,
    selection_refit_runner=run_mtt_selection_and_refit,
    standard_runner=run_standard_training,
    trainer_factory=Trainer,
    early_stopping_factory=EarlyStopping,
    seed_fn=seed_everything,
):
    if config.model.name == 'mtt':
        if config.mtt_protocol == 'standalone':
            model = mtt_model_factory(config)
            return standard_runner(
                config, model, train_loader, val_loader, test_loader, logger,
                trainer_factory=trainer_factory,
                early_stopping_factory=early_stopping_factory,
            )
        return selection_refit_runner(
            config, train_loader, val_loader, test_loader, logger,
            model_factory=mtt_model_factory,
            trainer_factory=trainer_factory,
            early_stopping_factory=early_stopping_factory,
            seed_fn=seed_fn,
        )
    model = mtt_model_factory(config) if config.model.type == 'mt' else singletask_model_factory(config)
    return standard_runner(
        config, model, train_loader, val_loader, test_loader, logger,
        trainer_factory=trainer_factory,
        early_stopping_factory=early_stopping_factory,
    )

if __name__ == "__main__":
    # Create the configuration
    args = parser.parse_args()
    config = create_config(vars(args))

    # Set seeds
    seed_everything(config.seed, workers=True)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config)
    print(f'Loaded dataset: {config.data.name} | # train samples: {len(train_loader)*config.training.batch_size}')

    # Initialize CSV logger
    run_name = create_run_name(config) if config.run_name == 'auto' else config.run_name
    
    from pathlib import Path

    path = Path(config.dataset)
    fold = path.name

    logger = CSVLogger(
        save_dir=config.log_root,
        name=run_name,
#        version=f'seed_{config.seed}'
        version=fold

    )
    
    # Log hyperparameters
    logger.log_hyperparams(config)

    run_training(config, train_loader, val_loader, test_loader, logger)
