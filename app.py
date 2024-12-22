import argparse
from pathlib import Path

from ui.app_class import ImageMatchingApp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server_name",
        type=str,
        default="0.0.0.0",
        help="server name",
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=7860,
        help="server port",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=Path(__file__).parent / "ui/config.yaml",
        help="config file",
    )
    args = parser.parse_args()
    ImageMatchingApp(args.server_name, args.server_port, config=args.config).run()
