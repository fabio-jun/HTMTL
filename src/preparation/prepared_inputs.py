import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import yaml

from dataset_registry import checked_path


def native_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_native_matrices(target, label, features, targets, folds):
    target = Path(target)
    if target.exists():
        raise FileExistsError('Native materialized output exists')
    target.mkdir(parents=True)

    manifest = {'protocol': 'native_materialized', 'label': label, 'n_folds': 5, 'seed': 42,
                'schema': {'feature_columns': list(features), 'target_columns': list(targets)}, 'files': {}}

    for record in folds:
        frames = []
        for role, prefix in (('train', 'train'), ('validation', 'validation')):
            frame = pd.DataFrame(record['X_' + prefix], columns=features)
            for index, name in enumerate(targets): frame[name] = record['y_' + prefix][:, index]
            frame.insert(0, 'source_row_index', record[prefix + '_indices'])
            frame.insert(0, 'source_role', role)
            frames.append(frame)
        name = f"fold_{record['fold']}.csv"
        pd.concat(frames, ignore_index=True).to_csv(target / name, index=False)
        manifest['files'][name] = native_digest(target / name)
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    read_native_matrices(target)


def read_native_matrices(target):
    target = Path(target)
    if target.is_symlink() or not target.is_dir() or any(path.is_symlink() for path in target.rglob('*')):
        raise ValueError('Native matrix bundle must contain only regular files')

    manifest = json.loads((target / 'manifest.json').read_text())
    if manifest.get('protocol') != 'native_materialized' or manifest.get('n_folds') != 5 or manifest.get('seed') != 42:
        raise ValueError('Native matrix protocol mismatch')

    expected = {f'fold_{fold}.csv' for fold in range(1, 6)}
    if set(manifest['files']) != expected or {path.name for path in target.iterdir()} != expected | {'manifest.json'}:
        raise ValueError('Native matrix bundle file coverage mismatch')
    features = manifest['schema']['feature_columns']
    targets = manifest['schema']['target_columns']
    columns = ['source_role', 'source_row_index', *features, *targets]
    if len(columns) != len(set(columns)) or not features or not targets:
        raise ValueError('Native matrix schema mismatch')
    result, validation_once, population = [], [], None

    for fold in range(1, 6):
        path = target / f'fold_{fold}.csv'
        if native_digest(path) != manifest['files'][path.name]:
            raise ValueError('Native matrix integrity mismatch')

        frame = pd.read_csv(path, float_precision='round_trip')
        if list(frame.columns) != columns or set(frame['source_role']) != {'train', 'validation'}:
            raise ValueError('Native matrix columns/roles mismatch')

        numeric = frame[['source_row_index', *features, *targets]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all(): raise ValueError('Nonfinite native matrix')

        index = frame['source_row_index'].to_numpy(dtype=float)
        if (index < 0).any() or (index != np.floor(index)).any() or len(np.unique(index)) != len(index):
            raise ValueError('Native row identity mismatch')

        current = np.sort(index.astype(int))
        if population is None: population = current
        if not np.array_equal(population, current) or not np.array_equal(current, np.arange(len(current))):
            raise ValueError('Native fold cohort mismatch')

        train = frame.loc[frame['source_role'] == 'train'].reset_index(drop=True)
        validation = frame.loc[frame['source_role'] == 'validation'].reset_index(drop=True)

        validation_once.extend(validation['source_row_index'].tolist())
        result.append({'fold': fold, 'manifest': manifest, 'train_frame': train, 'validation_frame': validation})

    if not np.array_equal(np.sort(validation_once), population):
        raise ValueError('Native validation-once coverage mismatch')
    return result


def materialize_native(backend, item, source, work):
    records = []

    spec = backend.specs(item, source)[item['label']]
    exporter = backend.module('mtt_shared_export')

    features, targets = list(spec.feature_columns), list(spec.aliases)

    for fold in range(1, 6):
        bundle = exporter.build_fold(spec, backend.root, fold)
        indices = np.concatenate([bundle.role_indices['train'], bundle.role_indices['val']])
        order = np.argsort(indices)
        records.append({'fold': fold,
                        'X_train': np.concatenate([bundle.role_features['train'], bundle.role_features['val']])[order],
                        'X_validation': bundle.role_features['test'],
                        'y_train': np.column_stack([np.concatenate([bundle.role_targets['train'][target], bundle.role_targets['val'][target]])[order] for target in targets]),
                        'y_validation': np.column_stack([bundle.role_targets['test'][target] for target in targets]),
                        'train_indices': indices[order], 'validation_indices': bundle.role_indices['test']})
    output_targets = [spec.aliases[target] for target in targets]
    write_native_matrices(work / 'native' / item['label'], item['label'], features, output_targets, records)


def verify_persisted_export(backend, item, source, data, configs, label):
    bundle_dir = data / label
    record = json.loads((bundle_dir / 'bundle_manifest.json').read_text())
    if record.get('label') != label or record.get('folds') != list(range(1, 6)):
        raise ValueError('MTT bundle identity/coverage mismatch')
    prepared = item['family'] in {'assistments', 'nedbox'}
    if prepared:
        selected_sources = {source / 'manifest.json'}
    elif item['family'] == 'spr':
        selected_sources = {source / name for name in ('spr_xray_embeddings.npy', 'spr_xray_manifest.csv', 'cohort_selection.csv')}
    else:
        selected_sources = {source}
    selected_sources = {checked_path(backend.root, path) for path in selected_sources}

    spec = backend.specs(item, source)[item['label']]

    native_folds = None if prepared else read_native_matrices(data.parent.parent / 'native' / item['label'])
    if native_folds is not None:
        manifest = native_folds[0]['manifest']
        if manifest['label'] != item['label'] or set(manifest['schema']['target_columns']) != set(item['target_types']):
            raise ValueError('Native prepared label/targets differ from the official inventory')

    for fold in range(1, 6):
        directory, config = bundle_dir / f'fold_{fold}', configs / label / f'fold_{fold}'
        manifest_path = directory / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if native_digest(manifest_path) != record['fold_manifest_sha256'][str(fold)]:
            raise ValueError('MTT fold manifest integrity mismatch')
        if manifest.get('label') != label or manifest.get('fold') != fold or manifest.get('seed') != 42:
            raise ValueError('MTT fold identity mismatch')

        source_hashes = manifest.get('source_hashes', {str(source): manifest.get('source_csv_sha256')})
        recorded_sources = {checked_path(backend.root, filename) for filename in source_hashes}
        if recorded_sources != selected_sources:
            raise ValueError('Prepared artifact source differs from the selected source')
        if item['family'] == 'higgs' and checked_path(backend.root, manifest['source_csv']) != source:
            raise ValueError('HIGGS artifact source differs from the selected source')

        if spec.profile == 'higgs50k':
            _verify_fold_against_source(spec, backend.root, fold, bundle_dir, configs / label)
        else:
            for filename, expected in source_hashes.items():
                path = checked_path(backend.root, filename)
                if not expected or native_digest(path) != expected:
                    raise ValueError('MTT input is stale')

            if native_digest(directory / 'train_val_test.h5') != manifest['hdf5_sha256']:
                raise ValueError('MTT HDF5 integrity mismatch')

            for role in ('train', 'val', 'test'):
                if native_digest(directory / f'{role}_audit.csv') != manifest['audit_sha256'][role]:
                    raise ValueError('MTT audit integrity mismatch')

            for filename, key in (('dataset.yaml', 'dataset_config_sha256'), ('mtt.yaml', 'mtt_config_sha256')):
                if native_digest(config / filename) != manifest[key]:
                    raise ValueError('MTT config integrity mismatch')

            model = yaml.safe_load((config / 'mtt.yaml').read_text())
            expected_model = manifest.get('mtt_config')
            if expected_model is None:
                expected_model = backend.module('mtt_shared_registry').mtt_config(spec)
            if model != expected_model:
                raise ValueError('MTT model configuration mismatch')

        with h5py.File(directory / 'train_val_test.h5', 'r') as handle:
            if spec.profile != 'higgs50k':
                if set(handle) != {'train', 'val', 'test'}:
                    raise ValueError('MTT HDF5 role mismatch')
                for role in handle:
                    if len(handle[role]['features_numerical']) != manifest['row_counts'][role]:
                        raise ValueError('MTT HDF5 row count mismatch')
            if prepared:
                expected = backend.module('mtt_shared_export').build_fold(spec, backend.root, fold)
                for role in ('train', 'val', 'test'):
                    if not np.array_equal(handle[role]['features_numerical'][:], expected.role_features[role].astype(np.float32)):
                        raise ValueError('MTT input differs from the prepared native features')
                    for target in spec.aliases:
                        if not np.array_equal(handle[role][target][:], expected.role_targets[role][target].astype(np.float32)):
                            raise ValueError('MTT target differs from the prepared native target')
            else:
                native = native_folds[fold - 1]
                for role in ('train', 'val', 'test'):
                    frame = native['validation_frame'] if role == 'test' else native['train_frame']
                    frame = frame.set_index('source_row_index')
                    indices = pd.read_csv(directory / f'{role}_audit.csv')['row_index'].to_numpy()
                    features = native['manifest']['schema']['feature_columns']
                    expected = frame.loc[indices, features].to_numpy().astype(np.float32)
                    if not np.array_equal(expected, handle[role]['features_numerical'][:]):
                        raise ValueError('MTT input differs from the materialized native matrix')
                    for h5_target, output_target in spec.aliases.items():
                        if not np.array_equal(frame.loc[indices, output_target].to_numpy().astype(np.float32), handle[role][h5_target][:]):
                            raise ValueError('MTT/native target parity mismatch')


def _audit_native_index(audit):
    column = "legacy_row_index" if "legacy_row_index" in audit.columns else "row_index"
    return audit[column].to_numpy()


def _verify_fold_against_source(spec, project_root, fold, data_path, config_path):
    from mtt_shared_registry import SEED
    from mtt_shared_export import ROLES, build_fold, sha256_file, sha256_text

    fold_data = data_path / f"fold_{fold}"

    manifest = json.loads((fold_data / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("label") != spec.label or int(manifest.get("fold", -1)) != fold:
        raise ValueError(f"{spec.label} fold {fold}: manifest identity mismatch")
    if (spec.profile != "higgs50k" and manifest.get("family") != spec.family) or int(manifest.get("seed", -1)) != SEED:
        raise ValueError(f"{spec.label} fold {fold}: manifest family/seed mismatch")
    if list(manifest.get("feature_names", [])) != list(spec.feature_columns):
        raise ValueError(f"{spec.label} fold {fold}: manifest feature contract mismatch")
    task_map = [[spec.aliases[s], s, spec.target_types[s]] for s in spec.aliases] if spec.profile == "higgs50k" else [[s, spec.aliases[s], spec.target_types[s]] for s in spec.aliases]
    if manifest.get("task_map") != task_map:
        raise ValueError(f"{spec.label} fold {fold}: manifest task map mismatch")

    source_hashes = manifest.get("source_hashes", {manifest.get("source_csv", ""): manifest.get("source_csv_sha256")})
    if spec.profile == "higgs50k" and (Path(manifest.get("source_csv", "")).resolve() != Path(spec.source).resolve() or manifest.get("source_csv_sha256") != "bc149ac7c5dcaf4e3d2d0b61bd4cc5f393412e1bd2b595aec249613dc4439a6a"):
        raise ValueError("HIGGS source binding mismatch")
    for path, digest in source_hashes.items():
        if not Path(path).is_file() or sha256_file(path) != digest:
            raise ValueError(f"{spec.label} fold {fold}: stale source {path}")
    if sha256_file(fold_data / "train_val_test.h5") != manifest["hdf5_sha256"]:
        raise ValueError(f"{spec.label} fold {fold}: HDF5 hash mismatch")

    dataset_text = (config_path / f"fold_{fold}" / "dataset.yaml").read_text(encoding="utf-8")
    mtt_text = (config_path / f"fold_{fold}" / "mtt.yaml").read_text(encoding="utf-8")
    if sha256_text(dataset_text) != manifest["dataset_config_sha256"] or sha256_text(mtt_text) != manifest["mtt_config_sha256"]:
        raise ValueError(f"{spec.label} fold {fold}: config hash mismatch")

    expected = build_fold(spec, project_root, fold)
    if spec.profile == "higgs50k":
        import yaml
        from mtt_shared_export import _dataset_config
        from mtt_shared_registry import mtt_config
        if yaml.safe_load(dataset_text) != _dataset_config(expected, spec.num_features) or yaml.safe_load(mtt_text) != mtt_config(spec):
            raise ValueError("HIGGS persisted configs differ from the official recipe")
        if manifest.get("selection") != expected.selection or manifest.get("transform") != expected.transform or manifest.get("row_indices") != {role: expected.role_indices[role].tolist() for role in ROLES}:
            raise ValueError("HIGGS selection/transform/order provenance mismatch")
        if manifest.get("mtt_config") != mtt_config(spec) or manifest.get("preprocessing_fit_population") != "fold_outer_train_40000_rows_before_selection_split":
            raise ValueError("HIGGS preprocessing contract mismatch")

    for role in ROLES:
        if manifest["row_counts"][role] != len(expected.role_indices[role]):
            raise ValueError(f"{spec.label} fold {fold}: row count mismatch for {role}")

    with h5py.File(fold_data / "train_val_test.h5", "r") as handle:
        if set(handle) != set(ROLES):
            raise ValueError(f"{spec.label} fold {fold}: malformed HDF5 roles")
        for role in ROLES:
            audit = pd.read_csv(fold_data / f"{role}_audit.csv")
            if sha256_file(fold_data / f"{role}_audit.csv") != manifest["audit_sha256"][role]:
                raise ValueError(f"{spec.label} fold {fold}: audit hash mismatch for {role}")
            if not np.array_equal(_audit_native_index(audit), _audit_native_index(expected.role_identities[role])):
                raise ValueError(f"{spec.label} fold {fold}: {role} identity mismatch")
            if spec.profile == "higgs50k" and (list(audit.columns) != list(expected.role_identities[role].columns) or not np.array_equal(audit.to_numpy(dtype=float), expected.role_identities[role].to_numpy(dtype=float))):
                # Exact serialized audit comparison avoids pandas' non-round-trip float parsing.
                if (fold_data / f"{role}_audit.csv").read_text() != expected.role_identities[role].to_csv(index=False):
                    raise ValueError("HIGGS audit target values mismatch")

            group = handle[role]
            if group["features_numerical"].dtype != np.dtype("float32"):
                raise ValueError(f"{spec.label} fold {fold}: {role} feature dtype mismatch")
            if not np.array_equal(group["features_numerical"][:], np.asarray(expected.role_features[role], dtype=np.float32)):
                raise ValueError(f"{spec.label} fold {fold}: {role} features do not reproduce the native matrix")
            if sorted(group.keys()) != sorted(["features_numerical", *spec.aliases]):
                raise ValueError(f"{spec.label} fold {fold}: {role} dataset names mismatch")

            for source in spec.aliases:
                expected_target = np.asarray(expected.role_targets[role][source], dtype=np.float32)
                if not np.array_equal(group[source][:], expected_target):
                    raise ValueError(f"{spec.label} fold {fold}: {role} target {source} mismatch")

    return manifest, expected
