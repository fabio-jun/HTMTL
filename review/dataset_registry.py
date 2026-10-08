import importlib.util
import json
import sys
from pathlib import Path

def _load_dataset_module(unique_name: str, path: Path):
    if unique_name in sys.modules:
        cached = sys.modules[unique_name]
        if Path(cached.__file__).resolve() != Path(path).resolve():
            raise RuntimeError("Dataset module belongs to another checkout")
        return cached
    spec = importlib.util.spec_from_file_location(unique_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(unique_name, None)
        raise
    return module

def configure_imports(root):
    root = Path(root)
    for relative in ('review', 'src/preparation', 'src/execution', 'src/results', 'src/methods', 'datasets/higgs'):
        path = str(root / relative)
        if path not in sys.path:
            sys.path.insert(0, path)


def inventory(root):
    records = json.loads((Path(root) / 'review/datasets.json').read_text())['datasets']
    labels = [record['label'] for record in records]
    if len(labels) != len(set(labels)):
        raise ValueError('Duplicate labels in official inventory')
    return {record['label']: record for record in records}


def entry(root, label):
    try:
        return inventory(root)[label]
    except KeyError as error:
        raise ValueError(f'Unknown official label: {label}') from error


def checked_path(root, value):
    path = Path(value)
    if not path.is_absolute():
        path = Path(root) / path
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f'Symlinked path is forbidden: {component}')
    return path.resolve(strict=False)


def regular_tree(path):
    if not path.exists():
        raise FileNotFoundError(f'Missing input: {path}; run preprocess or prepare first')
    for child in path.rglob('*') if path.is_dir() else (path,):
        if child.is_symlink():
            raise ValueError(f'Symlinked input is forbidden: {child}')


def separate_output(root, destination, sources):
    root = Path(root).resolve()
    if destination == root or root.is_relative_to(destination):
        raise ValueError('Output must not replace the release root')
    for source in [root / 'experiment_outputs', root / 'reports', *sources]:
        if destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
            raise ValueError(f'Output overlaps input or protected results: {source}')


def write_json(path, record):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)
