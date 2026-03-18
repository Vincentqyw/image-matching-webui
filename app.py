import argparse
from pathlib import Path
from imcui.ui.app_class import ImageMatchingApp


def get_default_config_path():
    """Get default config path, same logic as CLI."""
    # First check if config/app.yaml exists in current working directory
    current_dir_config = Path.cwd() / "config" / "app.yaml"
    if current_dir_config.exists():
        return current_dir_config
    # Fall back to package's default config
    default_config_path = Path(__file__).parent / "imcui" / "config" / "app.yaml"
    return default_config_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--server_name",
        type=str,
        default="0.0.0.0",
        help="server name",
    )
    parser.add_argument(
        "-p",
        "--server_port",
        type=int,
        default=7860,
        help="server port",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="config file (default: check current dir, then use package default)",
    )
    args = parser.parse_args()

    # Resolve config path: use provided path, or find in current dir, or use package default
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = get_default_config_path()

    # Resolve example data path
    example_data_root = Path(__file__).parent / "imcui" / "datasets"

    ImageMatchingApp(
        args.server_name,
        args.server_port,
        config=config_path,
        example_data_root=example_data_root,
    ).run()
