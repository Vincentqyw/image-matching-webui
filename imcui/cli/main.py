"""
Command Line Interface (CLI) for Image Matching WebUI application.
This module provides a Click-based CLI to launch the ImageMatchingApp with configurable options.
"""

import click
from loguru import logger
from pathlib import Path
from imcui.ui.app_class import ImageMatchingApp


def get_default_config_path():
    """
    Get the default configuration file path.

    Returns:
        Path: Path to the default configuration file, either from current directory
              or from the package's internal config directory.
    """
    # First check if config.yaml exists in current working directory
    current_dir_config = Path.cwd() / "app.yaml"
    if current_dir_config.exists():
        logger.info(f"Using config file from current directory: {current_dir_config}")
        return current_dir_config

    # Then check if config/config.yaml exists in current working directory
    current_config_dir = Path.cwd() / "config" / "app.yaml"
    if current_config_dir.exists():
        logger.info(f"Using config file from current directory: {current_config_dir}")
        return current_config_dir

    # Fall back to the package's default config
    default_config_path = Path(__file__).parent.parent / "config" / "app.yaml"
    logger.info(
        f"No config file found in current directory. Using default: {default_config_path}"
    )
    return default_config_path


def get_example_data_default_path():
    """
    Get the default example data root path.

    Returns:
        Path: Path to the default example data root directory.
    """
    get_example_data_default_path = Path(__file__).parent.parent / "datasets"
    logger.info(f"Using example data root: {get_example_data_default_path}")
    return get_example_data_default_path


@click.command()
@click.option(
    "--server-name",
    "-s",
    default="0.0.0.0",
    show_default=True,
    help="Hostname or IP address to bind the server to. "
    'Use "0.0.0.0" to make the server accessible from other devices on the network.',
)
@click.option(
    "--server-port",
    "-p",
    type=int,
    default=7860,
    show_default=True,
    help="Port number to run the server on. "
    "Ensure the port is available and not blocked by firewall rules.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default=lambda: str(get_default_config_path()),
    show_default=True,
    help="Path to the configuration YAML file. "
    "Contains application settings, model configurations, and feature extraction parameters. "
    "If not specified, searches for config.yaml in current directory, then config/config.yaml, "
    "and finally falls back to the package default.",
)
@click.option(
    "--example-data-root",
    "-d",
    type=click.Path(exists=True, file_okay=False, readable=True),
    default=lambda: str(get_example_data_default_path()),
    show_default=True,
    help="Root directory containing example datasets for demonstration purposes. "
    "Should contain subdirectories with image pairs for matching.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output for debugging purposes. "
    "Shows detailed logs about the application startup and operation.",
)
@click.version_option(
    version="1.0.0", message="Image Matching WebUI Version %(version)s"
)
def main(server_name, server_port, config, example_data_root, verbose):
    """
    Launch the Image Matching WebUI application.

    This command starts a web-based interface for image matching and feature extraction.
    The application provides a user-friendly interface to compare images using various
    computer vision algorithms and deep learning models.

    Examples:

    \b
    # Start with default settings
    python -m imcui.cli

    \b
    # Start on a specific port
    python -m imcui.cli --server-port 8080

    \b
    # Use a custom configuration file
    python -m imcui.cli --config /path/to/custom/config.yaml

    \b
    # Start with verbose logging
    python -m imcui.cli --verbose
    """
    if verbose:
        click.echo("Starting Image Matching WebUI...")
        click.echo(f"Server: {server_name}:{server_port}")
        click.echo(f"Config file: {config}")
        click.echo(f"Example data root: {example_data_root}")

    try:
        # Initialize and run the ImageMatchingApp
        ImageMatchingApp(
            server_name,
            server_port,
            config=Path(config),
            example_data_root=Path(example_data_root),
        ).run()
    except Exception as e:
        click.echo(f"Error starting application: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    # Entry point when executed as a script
    main()
