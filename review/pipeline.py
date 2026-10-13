#!/usr/bin/env python3
import argparse
import importlib
import importlib.util
import json
import os

import subprocess
import sys

from pathlib import Path

MODEL_ALIASES = {'hmtrf_uniform': 'hu', 'hmtrf_uniform_norm': 'hun',
                 'hmtrf_random_norm': 'hrn', 'local': 'loc', 'ecc': 'ecc', 'rfstd': 'rfstd', 'mlp': 'mlp'}


def registry_module():
    path = Path(__file__).resolve().parent
    registry = path / 'dataset_registry.py'
    spec = importlib.util.spec_from_file_location('_article_registry', registry)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load dataset registry: {registry}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_registry = registry_module()
checked_path = _registry.checked_path
regular_tree = _registry.regular_tree
separate_output = _registry.separate_output
write_json = _registry.write_json


def inventory(root):
    return _registry.inventory(root)


def entry(root, label):
    try:
        return inventory(root)[label]
    except KeyError as error:
        raise ValueError(f'Unknown official label: {label}') from error


def command(root, relative, *arguments):
    return [sys.executable, str(Path(root) / relative), *map(str, arguments)]



class Backend:
    def __init__(self, root):
        self.root = Path(root)
        _registry.configure_imports(self.root)

    def module(self, name):
        loaded = importlib.import_module(name)
        if loaded.__file__ is None or not any(Path(loaded.__file__).resolve().is_relative_to(parent)
                                             for parent in (self.root / 'src', self.root / 'review', self.root / 'datasets/higgs')):
            raise RuntimeError(f'{name} belongs to another checkout, run the release in an isolated interpreter')
        return loaded

    def adapter(self, item):
        path = checked_path(self.root, item['preprocess_module'])
        regular_tree(path)
        module = self.module('dataset_registry')._load_dataset_module('_prepared_' + item['label'], path)
        return module.create_adapter()

    def prepare_source(self, item, source, workers):
        parameters = item.get('preparation')
        if not parameters or parameters['n_folds'] != 5 or parameters['seed'] != 42:
            raise ValueError('Preparation parameters are missing or invalid')
        adapter = self.adapter(item)
        self.module('fold_preparation').prepare_dataset_folds(adapter, source.parent, workers=workers, **parameters)


    def specs(self, item, source):
        return self.module('mtt_shared_registry').build_specs(
            self.root, labels=[item['label']], sources={item['label']: source})

    def export_mtt(self, item, source, work):
        data, configs = work / 'mtt/data', work / 'mtt/configs'
        self.module('mtt_shared_export').export_labels(
            self.root, labels=[item['label']], specs=self.specs(item, source), data_path=data, config_path=configs)
        if item['family'] not in {'assistments', 'nedbox'}:
            self.materialize_native(item, source, work)

    def materialize_native(self, item, source, work):
        return self.module('prepared_inputs').materialize_native(self, item, source, work)

    def verify_inputs(self, item, source, work, models):
        data, configs = work / 'mtt/data', work / 'mtt/configs'
        label = self.specs(item, source)[item['label']].label
        for parent in (data / label, configs / label):
            regular_tree(checked_path(self.root, parent))
        self.verify_persisted_export(item, source, data, configs, label)

    def verify_persisted_export(self, item, source, data, configs, label):
        return self.module('prepared_inputs').verify_persisted_export(self, item, source, data, configs, label)

    def run_native(self, item, source, output, models):
        args = [source, output / 'native', *[MODEL_ALIASES[model] for model in models]]
        environment = os.environ.copy()
        environment['CUDA_VISIBLE_DEVICES'] = '-1'
        return subprocess.run(command(self.root, 'src/execution/native_run.py', *args),
                              cwd=self.root, env=environment, check=False).returncode

    def training_ready(self, models):
        if 'mtt' in models:
            runtime = checked_path(self.root, 'src/methods/MultiTab/main.py')
            regular_tree(runtime)

    def run_mtt(self, item, source, work, output):
        data, configs = work / 'mtt/data', work / 'mtt/configs'
        return self.module('mtt_shared_run').run(
            self.root, labels=[item['label']], specs=self.specs(item, source),
            data_path=data, config_path=configs, log_root_path=output / 'mtt_logs', exit_root=output / 'exitcodes')

    def native_truth(self, item, source):
        metrics = self.module('experiment_metrics')
        folds = self.module('fold_preparation').read_prepared_folds(source)
        targets = folds[0]['manifest']['schema']['target_columns']
        return {fold['fold']: (fold['validation_frame'][targets], metrics.np.arange(len(fold['validation_frame']))) for fold in folds}

    def native_aliases(self, item):
        if item['family'] == 'assistments':
            return item['target_aliases']
        return {target: target for target in item['target_types']}

    def finish_results(self, item, source, work, output, models):
        return self.module('experiment_metrics').finish_results(self, item, source, work, output, models)

    def verify_results(self, item, source, work, output, models):
        return self.module('experiment_metrics').verify_results(self, item, source, work, output, models)

    def preprocess(self, item, input_path=None, output_path=None, device=None, batch_size=None):
        if item['family'] in {'higgs', 'spr'}:
            family = item['family']
            defaults = {
                'higgs': ('datasets/higgs/train_val_test.h5', 'datasets/higgs/higgs_50k.indices.npz', 'datasets/higgs/preprocess.py'),
                'spr': ('datasets/spr xray', 'datasets/spr xray/cohort_selection.csv', 'datasets/spr xray/preprocess.py'),
            }
            default_input, default_selection, script_name = defaults[family]
            source = checked_path(self.root, input_path if input_path is not None else default_input)
            selection = checked_path(self.root, default_selection)
            output = checked_path(self.root, output_path if output_path is not None else item['input_location'])
            if output in (source, selection) or source.is_relative_to(output) or selection.is_relative_to(output):
                raise ValueError('Cohort output must not replace its source or selection map')
            if output.exists():
                raise FileExistsError(f'Preprocessing must not overwrite {output}')
            raw_spr = family == 'spr' and (source / 'train_age.csv').is_file() and (source / 'train_gender.csv').is_file() and (source / 'kaggle/kaggle/train').is_dir()
            if raw_spr:
                if source.is_symlink() or not source.is_dir():
                    raise ValueError('Raw SPR input must be a regular Kaggle dataset directory')
            else:
                regular_tree(source)
            regular_tree(selection)
            script = checked_path(self.root, script_name)
            regular_tree(script)
            arguments = [sys.executable, str(script), '--input', str(source), '--output', str(output), '--selection', str(selection)]
            if raw_spr:
                arguments += ['--raw']
                if device is not None:
                    arguments += ['--device', str(device)]
                if batch_size is not None:
                    arguments += ['--batch-size', str(batch_size)]
            code = subprocess.run(arguments,
                                   cwd=self.root, check=False).returncode
            if code == 0:
                regular_tree(output)
            return code
        
        if input_path is not None or output_path is not None:
            raise ValueError('--input/--output overrides are supported only for HIGGS/SPR cohort reconstruction')
        
        if item['family'] in {'assistments', 'nedbox'}:
            adapter = self.adapter(item)
            output = checked_path(self.root, item['canonical_output'])
            if output.exists():
                raise FileExistsError(f'Refusing to overwrite canonical input: {output}')
            if item['family'] == 'nedbox':
                frame, _ = adapter.module.preprocess_dataframes(*adapter.module.load_sources())
                frame.to_csv(output, index=False)
            else:
                adapter.load()['artifact'].to_csv(output, index=False)
            return 0
        
        script = item.get('preprocess_script')

        if not script:
            raise ValueError(f"{item['label']}: no frozen raw reconstruction route, install the official input")
        outputs = item.get('preprocess_outputs')

        if not outputs or item['input_location'] not in outputs:
            raise ValueError('Raw route has no complete declared output contract')
        
        for relative in outputs:
            if checked_path(self.root, relative).exists():
                raise FileExistsError(f'Preprocessing must not overwrite {relative}')
            
        path = checked_path(self.root, script)
        arguments = item.get('preprocess_args', [])
        code = subprocess.run([sys.executable, str(path), *arguments], cwd=path.parent, check=False).returncode
        
        if code == 0:
            for relative in outputs:
                regular_tree(checked_path(self.root, relative))
        return code


def preprocess(root, label, backend=None, input_path=None, output_path=None,
               work_root=None, workers=1, device=None, batch_size=None):
    root = checked_path(Path.cwd(), root)
    handler = backend or Backend(root)
    item = entry(root, label)
    if work_root is not None:
        if input_path is not None or output_path is not None:
            raise ValueError('Fold reconstruction uses registered raw sources, input/output overrides are not supported')
        return prepare(root, label, work_root, workers=workers, backend=handler, reconstruct=True,
                       device=device, batch_size=batch_size)
    if input_path is not None or output_path is not None:
        return handler.preprocess(entry(root, label), input_path=input_path, output_path=output_path, device=device, batch_size=batch_size)
    return handler.preprocess(entry(root, label), device=device, batch_size=batch_size)


def prepare(root, label, work_root, workers=1, backend=None, reconstruct=False, device=None, batch_size=None):
    root = checked_path(Path.cwd(), root)
    item = entry(root, label)
    work = checked_path(root, work_root)
    source = checked_path(root, item['input_location'])
    separate_output(root, work, [source])
    if work.exists():
        raise FileExistsError(f'Refusing existing preparation output: {work}')
    backend = backend or Backend(root)
    work.mkdir(parents=True)
    if reconstruct and item['family'] not in {'assistments', 'nedbox'}:
        code = backend.preprocess(item, device=device, batch_size=batch_size) if item['family'] == 'spr' else backend.preprocess(item)
        if code != 0:
            raise RuntimeError(f"Source preprocessing failed with exit code {code}")
    elif reconstruct or not source.exists():
        if item['family'] not in {'assistments', 'nedbox'}:
            raise FileNotFoundError(f'Install declared input before prepare: {source}')
        source = work / 'native' / label
        backend.prepare_source(item, source, workers)
    backend.export_mtt(item, source, work)
    backend.verify_inputs(item, source, work, ['mtt'])
    stored_source = source.relative_to(root) if source.is_relative_to(root) else source
    write_json(work / 'prepared.json', {'label': label, 'source': str(stored_source), 'preparation': item.get('preparation')})
    return 0


def train(root, label, models, output_root, work_root=None, backend=None):
    run_panel = Backend(root).module('experiments').run_panel
    return run_panel(root, label, models, output_root, work_root, backend)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Experiment workflow: preprocess, prepare, and train stages.')
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    stages = parser.add_subparsers(dest='stage', required=True)
    for name in ('preprocess', 'prepare', 'train'):
        stages.add_parser(name).add_argument('--label', required=True)
    stages.choices['preprocess'].add_argument('--input', type=Path, help='HIGGS MultiTab HDF5, full SPR embeddings, or SPR Kaggle dataset root')
    stages.choices['preprocess'].add_argument('--output', type=Path, help='Reconstructed HIGGS CSV or SPR cohort directory')
    stages.choices['preprocess'].add_argument('--work-root', type=Path, help='Preprocess the registered source and prepare the native folds and MTT inputs')
    stages.choices['preprocess'].add_argument('--workers', type=int, default=1, help='Number of worker processes to use for preprocessing')
    stages.choices['preprocess'].add_argument('--device', help='Explicit device for SPR raw image extraction')
    stages.choices['preprocess'].add_argument('--batch-size', type=int, help='Batch size for SPR raw image extraction')
    stages.choices['prepare'].add_argument('--work-root', type=Path, required=True)
    stages.choices['prepare'].add_argument('--workers', type=int, default=1)
    stages.choices['train'].add_argument('--models', nargs='+', required=True)
    stages.choices['train'].add_argument('--work-root', type=Path, required=True)
    stages.choices['train'].add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.stage == 'preprocess': return preprocess(args.project_root, args.label, input_path=args.input, output_path=args.output, work_root=args.work_root, workers=args.workers, device=args.device, batch_size=args.batch_size)
    if args.stage == 'prepare': return prepare(args.project_root, args.label, args.work_root, workers=args.workers)
    return train(args.project_root, args.label, args.models, args.output_root, args.work_root)


if __name__ == '__main__': raise SystemExit(main())
