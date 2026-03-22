Configuration
=============

Image Matching WebUI uses YAML configuration files to customize behavior and default parameters.

Configuration Resolution
-------------------------

Configuration files are loaded in this order (first found):

1. Custom path via ``-c`` flag
2. ``cwd/app.yaml`` (current working directory)
3. ``cwd/config/app.yaml`` (current working directory config subdirectory)
4. Package default: ``imcui/config/app.yaml``

Configuration Options
---------------------

Device Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   device: cuda  # Options: cuda, cpu, mps (for macOS)

**CUDA Device Selection**: To use a specific GPU, set the ``CUDA_VISIBLE_DEVICES`` environment variable:

.. code-block:: bash

   export CUDA_VISIBLE_DEVICES=0
   imcui

Matching Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Default matcher to use
   default_matcher: superpoint-lightglue

   # Preprocessing options
   preprocessing:
     resize: 1024  # Resize images to this size (0 for no resize)
     grayscale: false  # Convert to grayscale

Visualization Options
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Keypoint display
   show_keypoints: true
   keypoint_size: 2

   # Match display
   show_lines: true
   line_thickness: 1

   # Match confidence threshold
   match_threshold: 0.5

   # Maximum matches to display
   max_matches: -1  # -1 for unlimited

RANSAC Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # RANSAC method
   ransac_method: MAGSAC  # Options: RANSAC, MAGSAC, LMEDS

   # RANSAC parameters
   ransac_reproj_threshold: 1.0  # Reprojection error threshold in pixels
   ransac_confidence: 0.99  # Confidence level
   ransac_max_iters: 5000  # Maximum iterations

   # Output options
   ransac_enable: true  # Enable RANSAC filtering

Geometry Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Geometry estimation type
   geometry_type: homography  # Options: homography, fundamental, essential

   # Essential matrix parameters (for calibrated cameras)
   essential:
     focal1: 1000.0  # Focal length for image 1
     focal2: 1000.0  # Focal length for image 2
     pp1: [640, 480]  # Principal point for image 1 [x, y]
     pp2: [640, 480]  # Principal point for image 2 [x, y]

Example Configuration Files
---------------------------

Minimal Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Minimal configuration file
   device: cuda
   default_matcher: superpoint-lightglue

Full Configuration
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Full configuration file
   device: cuda

   # Matching
   default_matcher: superpoint-lightglue
   preprocessing:
     resize: 1024
     grayscale: false

   # Visualization
   show_keypoints: true
   show_lines: true
   keypoint_size: 2
   line_thickness: 1
   match_threshold: 0.5
   max_matches: -1

   # RANSAC
   ransac_method: MAGSAC
   ransac_reproj_threshold: 1.0
   ransac_confidence: 0.99
   ransac_max_iters: 5000
   ransac_enable: true

   # Geometry
   geometry_type: homography

Data Configuration
------------------

Example Datasets Path
~~~~~~~~~~~~~~~~~~~~~

Example datasets can be configured via:

1. **Environment variable**:
   .. code-block:: bash

      export IMCUI_DATA_DIR=/path/to/datasets

2. **CLI flag**:
   .. code-block:: bash

      imcui -d /path/to/datasets

3. **Configuration file**:
   .. code-block:: yaml

      example_data_root: /path/to/datasets

If not specified, datasets are automatically downloaded to:

* **Linux/macOS**: ``~/.cache/imcui/datasets/``
* **Windows**: ``%LOCALAPPDATA%\imcui\datasets\``

Using Multiple Configurations
-----------------------------

You can have multiple configuration files for different use cases:

.. code-block:: bash

   # Development configuration
   imcui -c config_dev.yaml

   # Production configuration
   imcui -c config_prod.yaml

   # Testing configuration
   imcui -c config_test.yaml -d /test/data
