import argparse
from pathlib import Path


def main(argv=None):
    from experiments import all_models, run_native_folds, short_to_long

    parser = argparse.ArgumentParser(description='Internal CPU worker for native fold training.')
    parser.add_argument('source', type=Path)
    parser.add_argument('output_root', type=Path)
    parser.add_argument('models', nargs='+')
    args = parser.parse_args(argv)
    selected = all_models if args.models == ['all'] else [short_to_long.get(model, model) for model in args.models]
    invalid = sorted(set(selected).difference(all_models))
    if invalid:
        parser.error(f"invalid native model name(s): {', '.join(invalid)}")
    run_native_folds(args.source, args.output_root, selected)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
