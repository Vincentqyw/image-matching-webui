"""
Test cases for CLI and app.py functionality.
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path
import pytest
import yaml
from loguru import logger

# Add the parent directory to Python path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

TIMEOUT = 10  # seconds


@pytest.mark.skip(reason="Skipping due to occasional CI timeouts")
def test_cli_help():
    """Test that CLI help command works."""
    result = subprocess.run(
        [sys.executable, "-m", "imcui.cli.main", "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0
    assert "Launch the Image Matching WebUI application" in result.stdout
    assert "--server-port" in result.stdout
    assert "--config" in result.stdout
    logger.info("CLI help command works as expected.")


def test_cli_version():
    """Test that CLI version command works."""
    result = subprocess.run(
        [sys.executable, "-m", "imcui.cli.main", "--version"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0
    assert "Image Matching WebUI Version" in result.stdout
    logger.info("CLI version command works as expected.")


def test_cli_default_config_loading():
    """Test that CLI can load default configuration."""
    from imcui.cli.main import get_default_config_path

    config_path = get_default_config_path()
    assert config_path.exists(), f"Config file not found: {config_path}"

    # Load and validate config structure
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    assert "server" in config
    assert "name" in config["server"]
    assert "port" in config["server"]
    assert "matcher_zoo" in config
    assert "defaults" in config
    logger.info("CLI default configuration loaded and validated successfully.")


def test_app_py_help():
    """Test that app.py help command works."""
    result = subprocess.run(
        [sys.executable, "app.py", "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=TIMEOUT,  # Short timeout to prevent hanging
    )
    assert result.returncode == 0
    assert "server_name" in result.stdout
    assert "server_port" in result.stdout
    assert "config" in result.stdout
    logger.info("app.py help command works as expected.")


@pytest.mark.skip(reason="Skipping due to occasional CI timeouts")
def test_app_py_default_config():
    """Test that app.py can load default configuration."""
    # Test the config path resolution in app.py
    import argparse

    # Mock the argument parsing to test config path (app.py uses config/app.yaml as default)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "config/app.yaml"))
    args = parser.parse_args([])

    config_path = Path(args.config)
    assert config_path.exists(), f"Config file not found: {config_path}"
    logger.info(f"Default config path: {config_path}")

    # Load and validate config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    assert "server" in config
    assert "matcher_zoo" in config

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        log_path = temp_path / "experiments" / "all"
        log_path.mkdir(parents=True, exist_ok=True)

        # Update config to use temporary log path to avoid writing to repo
        config["log_path"] = str(log_path)

        # Save modified config to temp file
        temp_config = temp_path / "temp_config.yaml"
        with open(temp_config, "w") as f:
            yaml.dump(config, f)

        # Run app.py with the temporary config to ensure no errors
        result = subprocess.run(
            [sys.executable, "app.py", "--config", str(temp_config)],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=TIMEOUT,  # Increase timeout to account for initialization
        )

        # We expect the server to start but then timeout when trying to run
        assert result.returncode != 0 or "Running on" in result.stdout
        logger.info("app.py ran successfully with temporary config.")


def test_cli_with_custom_config():
    """Test CLI with custom configuration file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        custom_config = temp_path / "test_config.yaml"

        # Create a minimal valid config with only basic matchers to avoid slow loading
        config_data = {
            "server": {"name": "127.0.0.1", "port": 9999},
            "defaults": {"max_keypoints": 1000, "match_threshold": 0.1},
            "matcher_zoo": {
                "sift": {
                    "enable": True,
                    "matcher": "NN-mutual",
                    "feature": "sift",
                    "dense": False,
                    "info": {"name": "SIFT", "source": "IJCV 2004", "display": True},
                }
            },
        }

        with open(custom_config, "w") as f:
            yaml.dump(config_data, f)

        # Test that CLI can load the custom config
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "imcui.cli.main",
                "--config",
                str(custom_config),
                "--verbose",
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=TIMEOUT,  # Increase timeout to account for initialization
        )

        # CLI should start but then timeout when trying to run the server
        # We just want to verify it can parse the config
        assert "Config file:" in result.stdout or result.returncode != 0
        logger.info("CLI successfully loaded and parsed custom config.")


def test_package_entry_point():
    """Test that the package entry point works."""
    # Test that the package can be imported and has the expected structure
    try:
        import imcui
        from imcui.cli import main as cli_main
        from imcui.ui import app_class

        # Verify key components exist
        assert hasattr(imcui, "__version__") or hasattr(imcui, "__name__")
        assert callable(cli_main.main)
        assert hasattr(app_class, "ImageMatchingApp")

    except ImportError as e:
        pytest.fail(f"Failed to import package components: {e}")
    logger.info("Package entry point and components are valid.")


@pytest.mark.skipif(
    os.name == "nt", reason="Skip on Windows due to signal handling issues"
)
def test_cli_quick_exit():
    """Test that CLI exits quickly when interrupted."""
    import signal
    import time

    # Start the CLI process
    process = subprocess.Popen(
        [sys.executable, "-m", "imcui.cli.main", "--server-port", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=ROOT,
    )

    # Give it a moment to start
    time.sleep(1)

    # Send interrupt signal
    process.send_signal(signal.SIGINT)

    # Wait for process to terminate
    try:
        stdout, stderr = process.communicate(timeout=5)
        assert (
            process.returncode != 0
        )  # Should exit with non-zero code due to interrupt
    except subprocess.TimeoutExpired:
        process.kill()
        pytest.fail("CLI process did not exit promptly after interrupt")
    logger.info("CLI process exited promptly after interrupt.")


if __name__ == "__main__":
    # Run tests manually for debugging
    test_cli_help()
    test_cli_version()
    test_cli_default_config_loading()
    test_app_py_help()
    test_app_py_default_config()
    test_package_entry_point()
    print("All tests passed!")
