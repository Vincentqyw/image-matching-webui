Image Matching WebUI Documentation
===================================

Welcome to Image Matching WebUI (IMCUI) documentation. IMCUI is a Gradio-based web interface for matching image pairs using various state-of-the-art computer vision algorithms.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   cli
   api
   configuration
   development

Introduction
------------

Image Matching WebUI efficiently matches image pairs using multiple famous image matching algorithms. The tool features a Graphical User Interface (GUI) designed using `Gradio <https://gradio.app/>`_. You can effortlessly select two images and a matching algorithm and obtain a precise matching result.

**Note**: the images source can be either local images or webcam images.

Key Features
------------

* **Multiple Algorithms**: Support for sparse (keypoint-based) and dense (learned) matching methods
* **Easy to Use**: User-friendly Gradio-based web interface
* **Flexible Input**: Support for local images and webcam input
* **Configurable**: Custom configuration via YAML files
* **Package Installation**: Easy installation via PyPI

Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install imcui

Run the web interface:

.. code-block:: bash

   imcui

Then open http://localhost:7860 in your browser.

Online Demo
-----------

Try the online demo on `HuggingFace Spaces <https://huggingface.co/spaces/Realcat/image-matching-webui>`_.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
