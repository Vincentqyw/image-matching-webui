"""
Command Line Interface (CLI) for Image Matching WebUI application.
This module provides a Click-based CLI to launch the ImageMatchingApp with configurable options.
"""

import click
from pathlib import Path

from imcui import (
    ImageMatchingApp,
    get_default_config_path,
    get_example_data_path,
    get_version,
)


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
    type=click.Path(file_okay=False),
    default=None,
    show_default=True,
    help="Root directory containing example datasets for demonstration purposes. "
    "If not specified, auto-downloads to user cache directory on first run. "
    "Developers can also set IMCUI_DATA_DIR environment variable.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output for debugging purposes. "
    "Shows detailed logs about the application startup and operation.",
)
@click.version_option(
    version=get_version(), message="Image Matching WebUI Version %(version)s"
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

    # Resolve example data path (with auto-download support)
    data_path = (
        Path(example_data_root) if example_data_root else get_example_data_path()
    )

    if verbose:
        click.echo(f"Example data root: {data_path}")

    try:
        # Initialize and run the ImageMatchingApp
        ImageMatchingApp(
            server_name,
            server_port,
            config=Path(config),
            example_data_root=data_path,
        ).run()
    except Exception as e:
        click.echo(f"Error starting application: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    # Entry point when executed as a script
    main()
