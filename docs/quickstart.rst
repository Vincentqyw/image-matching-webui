Quick Start Guide
=================

Running the Web Interface
--------------------------

Start the application:

.. code-block:: bash

   imcui

Then open http://localhost:7860 in your browser.

Basic Usage
-----------

1. **Select Images**: Upload two images you want to match
2. **Choose Algorithm**: Select from the available matching algorithms
3. **Run Matching**: Click the "Match" button to start the matching process
4. **View Results**: The matching results will be displayed with visualization

Command Line Options
--------------------

The ``imcui`` package provides several command-line options:

.. code-block:: bash

   # Run with verbose output
   imcui --verbose

   # Run on a specific port
   imcui -p 7860

   # Run on a specific host
   imcui -s 127.0.0.1

   # Use custom configuration
   imcui -c /path/to/config.yaml

   # Specify custom data directory
   imcui -d /path/to/datasets

For all available options, use:

.. code-block:: bash

   imcui --help

Example Datasets
----------------

On first run, example datasets are automatically downloaded. They include sample image pairs for testing different matching algorithms.

The datasets are downloaded to:

* **Linux/macOS**: ``~/.cache/imcui/datasets/``
* **Windows**: ``%LOCALAPPDATA%\imcui\datasets\``

Common Use Cases
---------------

Matching Local Images
~~~~~~~~~~~~~~~~~~~~~

Simply upload two local images through the web interface and select your preferred matching algorithm.

Real-time Webcam Matching
~~~~~~~~~~~~~~~~~~~~~~~~~

Use the webcam input option to capture images in real-time and perform matching.

Testing Different Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the example datasets to compare performance across different matching algorithms.

Custom Configuration
~~~~~~~~~~~~~~~~~~~~

Create a custom ``app.yaml`` configuration file to set default parameters:

.. code-block:: yaml

   device: cuda

   defaults:
     setting_threshold: 0.1
     max_keypoints: 2000
     match_threshold: 0.2
     setting_geometry: Homography
     enable_ransac: true
     ransac_method: CV2_USAC_MAGSAC
     ransac_reproj_threshold: 8.0

Then use it:

.. code-block:: bash

   imcui -c /path/to/custom_config.yaml
