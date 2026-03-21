"""HuggingFace Spaces entry point.

This module provides backward compatibility for HuggingFace Spaces deployment
and users who run 'python app.py' directly.
"""

import argparse
from pathlib import Path

from imcui import ImageMatchingApp, get_default_config_path, get_example_data_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--server-name",
        type=str,
        default="0.0.0.0",
        help="server name",
    )
    parser.add_argument(
        "-p",
        "--server-port",
        type=int,
        default=7860,
        help="server port",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="config file (default: auto-detect from current dir or use package default)",
    )
    parser.add_argument(
        "-d",
        "--example-data-root",
        type=str,
        default=None,
        help="root directory containing example datasets (default: auto-download to user cache)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable verbose output",
    )
    args = parser.parse_args()

    # Use provided config path, or get default using shared utility
    config_path = Path(args.config) if args.config else get_default_config_path()

    # Resolve example data path (with auto-download support)
    data_path = (
        Path(args.example_data_root)
        if args.example_data_root
        else get_example_data_path()
    )

    ImageMatchingApp(
        args.server_name,
        args.server_port,
        config=config_path,
        example_data_root=data_path,
    ).run()
