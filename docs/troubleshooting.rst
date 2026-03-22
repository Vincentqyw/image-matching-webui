Troubleshooting
===============

This section provides solutions to common issues you may encounter when using Image Matching WebUI.

Common Problems
----------------

Installation Issues
~~~~~~~~~~~~~~~~~~~

Package Not Found Error
^^^^^^^^^^^^^^^^^^^^^^^

If you get a "Package not found" error when installing from PyPI:

.. code-block:: bash

   pip install imcui

Try upgrading pip:

.. code-block:: bash

   pip install --upgrade pip
   pip install imcui

Or install directly from the source:

.. code-block:: bash

   git clone https://github.com/Vincentqyw/image-matching-webui.git
   cd image-matching-webui
   pip install -e .

Missing Dependencies Error
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you encounter missing dependencies:

.. code-block:: bash

   pip install -e .[dev]  # For development dependencies

Runtime Issues
~~~~~~~~~~~~~~

CUDA Out of Memory
^^^^^^^^^^^^^^^^^^

If you encounter CUDA out of memory errors:

1. **Use CPU mode**: Set ``device: cpu`` in your configuration file
2. **Reduce image size**: Lower the ``max_keypoints`` parameter
3. **Close other GPU applications**: Ensure no other processes are using GPU memory
4. **Use batch processing**: Process images one at a time

Example configuration for CUDA memory issues:

.. code-block:: yaml

   device: cpu

   defaults:
     max_keypoints: 1024  # Reduce from default 2000
     setting_threshold: 0.15  # Increase threshold to reduce detections

Dataset Download Issues
~~~~~~~~~~~~~~~~~~~~~~~

Download Fails or Hangs
^^^^^^^^^^^^^^^^^^^^^^^^

If the automatic dataset download fails:

1. **Manual download**: Download datasets from HuggingFace manually
2. **Set custom path**: Specify a local directory with existing data
3. **Check internet connection**: Ensure you have stable internet connection

Use a custom data directory:

.. code-block:: bash

   export IMCUI_DATA_DIR=/path/to/your/datasets
   imcui

Or use the CLI flag:

.. code-block:: bash

   imcui -d /path/to/your/datasets

Import Errors
~~~~~~~~~~~~~

Module Import Error
^^^^^^^^^^^^^^^^^^^

If you encounter import errors:

1. **Verify installation**: Ensure the package is installed correctly

   .. code-block:: bash

      pip show imcui

2. **Reinstall**: If showing issues, reinstall the package

   .. code-block:: bash

      pip uninstall imcui
      pip install -e .

3. **Check Python version**: Ensure you're using Python 3.10+

   .. code-block:: bash

      python --version

4. **Verify dependencies**: Check if all dependencies are installed

   .. code-block:: bash

      pip check

Vismatch Import Error
^^^^^^^^^^^^^^^^^^^^

If you get an error related to `vismatch`:

.. code-block:: bash

   pip install --upgrade vismatch

The `vismatch` package is essential as it contains all the matching algorithms used by this WebUI.

Algorithm-Specific Issues
-------------------------

Model Loading Errors
~~~~~~~~~~~~~~~~~~~~

GPU Memory Insufficient for Large Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some large models may not fit in your GPU memory:

1. Use smaller models (e.g., `superpoint-lightglue` instead of larger ones)
2. Reduce image resolution
3. Use CPU mode if necessary

Model Download Fails
^^^^^^^^^^^^^^^^^^^^

If model download fails:

1. Check your internet connection
2. Check available disk space
3. Try manual download from the model repository
4. Clear HuggingFace cache and retry

Performance Issues
------------------

Slow Processing Speed
~~~~~~~~~~~~~~~~~~~~

If matching is slower than expected:

1. **Enable GPU**: Ensure you're using CUDA device

   .. code-block:: yaml

      device: cuda

2. **Reduce image resolution**: Smaller images process faster

3. **Adjust detection parameters**: Increase threshold to reduce keypoints

   .. code-block:: yaml

      defaults:
        setting_threshold: 0.15  # Higher = fewer but stronger keypoints

4. **Disable RANSAC**: If results are acceptable without filtering

   .. code-block:: yaml

      defaults:
        enable_ransac: false

High Memory Usage
~~~~~~~~~~~~~~~~~

If the application uses excessive memory:

1. **Reduce max_keypoints**: Limit the number of detected keypoints

   .. code-block:: yaml

      defaults:
        max_keypoints: 1024

2. **Clear model cache**: Restart the application to free cached models

3. **Use CPU mode**: If memory is an issue, CPU mode may be more predictable

Configuration Issues
--------------------

Configuration Not Applied
~~~~~~~~~~~~~~~~~~~~~~~~~

If your configuration file changes are not taking effect:

1. **Check file location**: Ensure the configuration file is in a valid location

2. **Verify syntax**: Check YAML syntax is correct

3. **Use absolute path**: Specify the configuration file path explicitly

   .. code-block:: bash

      imcui -c /full/path/to/your/config.yaml

4. **Check precedence**: Remember that CLI flags override configuration files

Web Interface Issues
--------------------

Gradio Interface Not Loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the web interface doesn't load:

1. **Check port**: Ensure port 7860 is not already in use

   .. code-block:: bash

      # Kill existing process on port 7860
      lsof -ti:7860 | xargs kill -9

      # Or use a different port
      imcui -p 8080

2. **Check network**: Ensure the specified host is accessible

   .. code-block:: bash

      # Use localhost only
      imcui -s 127.0.0.1

3. **Check logs**: Look at the console output for error messages

   .. code-block:: bash

      imcui --verbose

API Issues
----------

Server Start Fails
~~~~~~~~~~~~~~~~~~

If the API server fails to start:

1. **Check dependencies**: Ensure FastAPI and Uvicorn are installed

   .. code-block:: bash

      pip install fastapi uvicorn>=0.27.0

2. **Check port**: Ensure the API port is not already in use

3. **Verify configuration**: Check your API configuration file

Getting Help
------------

If you cannot resolve your issue:

1. **Check existing issues**: Look at the `GitHub issue tracker <https://github.com/Vincentqyw/image-matching-webui/issues>`_

2. **Create a new issue**: Include the following information:

   - Your operating system
   - Python version
   - Error message (full stack trace if available)
   - Steps to reproduce the issue
   - Configuration file (if relevant)

3. **Consult the vismatch repository**: Many algorithm-specific issues are covered in the `vismatch documentation <https://github.com/gmberton/vismatch>`_

4. **Check the community**: Look for similar discussions in related repositories and forums
