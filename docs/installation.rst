Installation
============

Requirements
------------

* `Python 3.10+ <https://www.python.org/downloads/>`_

Install from PyPI (Recommended)
--------------------------------

The recommended way to install Image Matching WebUI is via PyPI:

.. code-block:: bash

   pip install imcui

Optional: Install with auto-download support for example datasets:

.. code-block:: bash

   pip install imcui[datasets]

**Note**: Example datasets (82MB) are **automatically downloaded** from HuggingFace on first run to your user cache directory:

* **Linux/macOS**: ``~/.cache/imcui/datasets/``
* **Windows**: ``%LOCALAPPDATA%\imcui\datasets\``

The download happens transparently on first launch. For offline use or custom data paths, use:

* Environment variable: ``export IMCUI_DATA_DIR=/path/to/datasets``
* CLI flag: ``imcui -d /path/to/datasets``

Install from Source
-------------------

To install from source:

.. code-block:: bash

   git clone https://github.com/Vincentqyw/image-matching-webui.git
   cd image-matching-webui
   pip install -e .

Datasets will be auto-downloaded on first run.

Docker Installation
-------------------

Using Docker Compose:

.. code-block:: bash

   docker pull vincentqin/image-matching-webui:latest

   # Start the WebUI service
   docker-compose up webui

   # Or run in the background
   docker-compose up -d webui

Deprecated: Conda Installation
-------------------------------

..警告::

   ``environment.yaml`` is deprecated. Please use pip instead.

   .. code-block:: bash

      git clone https://github.com/Vincentqyw/image-matching-webui.git
      cd image-matching-webui
      conda env create -f environment.yaml
      conda activate imcui
      pip install -e .

Verification
------------

To verify your installation:

.. code-block:: bash

   # Check version
   imcui --version

   # Display help
   imcui --help

If the installation was successful, you should see the version information and help text.
