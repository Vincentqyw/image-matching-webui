Development
===========

This section covers development setup, testing, and contributing to Image Matching WebUI.

Development Installation
-------------------------

Install the package in development mode:

.. code-block:: bash

   git clone https://github.com/Vincentqyw/image-matching-webui.git
   cd image-matching-webui
   pip install -e .

Install development dependencies:

.. code-block:: bash

   pip install -e .[dev]

Code Quality
------------

Pre-commit Hooks
~~~~~~~~~~~~~~~~

Setup pre-commit hooks for automatic code formatting and linting:

.. code-block:: bash

   pip install pre-commit
   pre-commit install

Run pre-commit checks:

.. code-block:: bash

   pre-commit run -a

This automatically runs:

* **ruff**: Fast Python linter
* **ruff-format**: Code formatter
* **mypy**: Type checker
* **clang-format**: C++ code formatter

Manual Formatting
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Check code with ruff
   ruff check .

   # Format code with ruff
   ruff format .

   # Type check with mypy
   mypy imcui/

Testing
-------

Run Tests
~~~~~~~~~

.. code-block:: bash

   # Run all tests
   pytest tests/ -v

   # Run specific test
   pytest tests/test_image_matching.py::test_one -v

   # Run with coverage
   pytest tests/ -v --cov=imcui

Test Structure
~~~~~~~~~~~~~~

The test suite includes:

* Unit tests for core functions
* Integration tests for the API
* End-to-end tests for the UI

Contributing
------------
Contributions are welcome! Please follow these guidelines:

Pull Request Process
~~~~~~~~~~~~~~~~~~~~

1. Fork the repository
2. Create a feature branch (``git checkout -b feature/amazing-feature``)
3. Make your changes
4. Run tests and pre-commit checks
5. Commit your changes
6. Push to the branch (``git push origin feature/amazing-feature``)
7. Open a Pull Request

Code Style
~~~~~~~~~~

* Follow `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ style guidelines
* Use meaningful variable and function names
* Add docstrings for new functions and classes
* Keep functions focused and modular

Adding New Features
~~~~~~~~~~~~~~~~~~~

UI Features
^^^^^^^^^^^

New UI features should be added to the ``imcui/ui/`` module:

1. Add the feature to the appropriate submodule
2. Update ``imcui/ui/__init__.py`` to export the feature
3. Add tests in ``tests/`` directory
4. Update documentation

API Features
^^^^^^^^^^^^

API features should be added to ``imcui/api/`` module:

1. Implement the feature in ``imcui/api/image_matching_api.py`` or create a new module
2. Add tests
3. Update API documentation

Supported Algorithms
~~~~~~~~~~~~~~~~~~~~

.. note::

   This WebUI no longer maintains matching algorithms. All matchers are maintained in the `vismatch <https://github.com/gmberton/vismatch>`_ repository by `@gmberton <https://github.com/gmberton>`_. To add new matchers or algorithms, please contribute to the vismatch repository.

Documentation
-------------

Building Documentation
~~~~~~~~~~~~~~~~~~~~~

Build the documentation locally:

.. code-block:: bash

   cd docs
   pip install -r requirements.txt
   make html

The documentation will be built in ``docs/_build/html/``.

Open the documentation:

.. code-block:: bash

   open _build/html/index.html  # macOS
   xdg-open _build/html/index.html  # Linux
   start _build/html/index.html  # Windows

Adding Documentation
~~~~~~~~~~~~~~~~~~~~

1. Create or update RST files in ``docs/`` directory
2. Update ``index.rst`` to include new pages
3. Build documentation to verify changes

Project Structure
-----------------

.. list-table:: Project Structure

   * - ``imcui/``
     - Main package
   * - ``imcui/cli/``
     - CLI entry point
   * - ``imcui/ui/``
     - Gradio-based web interface
   * - ``imcui/api/``
     - Core matching API
   * - ``imcui/config/``
     - Configuration files
   * - ``cpp/``
     - C++ code (independent build system)
   * - ``tests/``
     - Python test suite
   * - ``docker/``
     - Docker deployment configuration
   * - ``docs/``
     - Documentation

Module Organization
~~~~~~~~~~~~~~~~~~~

UI Modules (``imcui/ui/``):

* ``image_matching_app.py`` - Main Gradio application class
* ``config.py`` - Configuration constants and matching utilities
* ``config_utils.py`` - Path resolution, version management
* ``matching.py`` - Matching logic and RANSAC filtering
* ``geometry.py`` - Geometry estimation
* ``image_utils.py`` - Image processing utilities
* ``examples.py`` - Example dataset generation
* ``visualization.py`` - Visualization utilities
* ``model_cache.py`` - Model caching implementation

Debugging
---------

Verbose Mode
~~~~~~~~~~~~

Enable verbose output for debugging:

.. code-block:: bash

   imcui --verbose

Error Logs
~~~~~~~~~~

Logs are written to:

* Run-time logs: ``log.txt`` (if created)
* Gradio logs: Console output
* Model cache: ``~/.cache/imcui/`` (Linux/macOS)

Common Issues
-------------

CUDA Out of Memory
~~~~~~~~~~~~~~~~~~

If you encounter CUDA out of memory errors:

1. Use CPU mode: ``device: cpu`` in config
2. Reduce image size in preprocessing
3. Close other GPU applications

Dataset Download Issues
~~~~~~~~~~~~~~~~~~~~~~~

If dataset download fails:

1. Manually download from HuggingFace
2. Set ``IMCUI_DATA_DIR`` to local directory
3. Use ``-d`` flag to specify local path

Import Errors
~~~~~~~~~~~~~

If you encounter import errors:

1. Ensure the package is installed: ``pip install -e .``
2. Check Python version compatibility (requires 3.10+)
3. Verify all dependencies are installed

Resources
---------

* `GitHub Repository <https://github.com/Vincentqyw/image-matching-webui>`_
* `Issue Tracker <https://github.com/Vincentqyw/image-matching-webui/issues>`_
* `vismatch Repository <https://github.com/gmberton/vismatch>`_
