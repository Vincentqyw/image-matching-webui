Command Line Interface
=======================

The ``imcui`` package provides a powerful command-line interface with various options.

Usage
-----

.. code-block:: bash

   imcui [OPTIONS]

Options
-------

All options have both long and short forms:

.. table:: CLI Options

   +----------------------+----------+----------------+------------------------------------------+
   | Option               | Short    | Default        | Description                              |
   +======================+==========+================+==========================================+
   | --server-name        | -s       | 0.0.0.0        | Hostname or IP to bind the server to     |
   +----------------------+----------+----------------+------------------------------------------+
   | --server-port        | -p       | 7860           | Port number to run the server on         |
   +----------------------+----------+----------------+------------------------------------------+
   | --config             | -c       | Auto-detected  | Path to custom configuration YAML file   |
   +----------------------+----------+----------------+------------------------------------------+
   | --example-data-root  | -d       | Auto-download  | Root directory for example datasets      |
   +----------------------+----------+----------------+------------------------------------------+
   | --verbose            | -v       | False          | Enable verbose output for debugging      |
   +----------------------+----------+----------------+------------------------------------------+
   | --version            |          |                | Show version information and exit         |
   +----------------------+----------+----------------+------------------------------------------+
   | --help               | -h       |                | Show help message and exit               |
   +----------------------+----------+----------------+------------------------------------------+

Configuration Resolution
-------------------------

Configuration files are loaded in this order (first found):

1. Custom path via ``-c`` flag
2. ``cwd/app.yaml`` (current working directory)
3. ``cwd/config/app.yaml`` (current working directory config subdirectory)
4. Package default: ``imcui/config/app.yaml``

Example Usage
-------------

Basic Usage
~~~~~~~~~~~

.. code-block:: bash

   # Run with default settings
   imcui

   # Run with verbose output
   imcui --verbose

   # Display version
   imcui --version

Custom Host and Port
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run on custom port
   imcui -p 8080

   # Run on specific host
   imcui -s 127.0.0.1

   # Combine options
   imcui -s 127.0.0.1 -p 8080

Custom Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Use custom configuration file
   imcui -c /path/to/config.yaml

Custom Data Directory
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Specify custom data directory
   imcui -d /path/to/datasets

   # Or use environment variable
   export IMCUI_DATA_DIR=/path/to/datasets
   imcui

All Options Combined
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   imcui -s 127.0.0.1 -p 8080 -c my_config.yaml -d /my/data --verbose

Configuration File
------------------

The configuration file uses YAML format. Here's an example:

.. code-block:: yaml

   # Device configuration
   device: cuda  # or cpu

   # Matching configuration
   default_matcher: superpoint-lightglue

   # Visualization options
   show_keypoints: true
   show_lines: false

   # RANSAC configuration
   ransac_method: MAGSAC
   ransac_reproj_threshold: 1.0
   ransac_confidence: 0.99

Save this as ``app.yaml`` and use it:

.. code-block:: bash

   imcui -c app.yaml
