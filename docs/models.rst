Available Matching Models
==========================

Image Matching WebUI supports a wide range of state-of-the-art matching algorithms through the `vismatch <https://github.com/gmberton/vismatch>`_ library. All models are dynamically loaded and configurable.

.. note::

   The actual matching algorithms are maintained in the `vismatch <https://github.com/gmberton/vismatch>`_ repository. For detailed model information and the most up-to-date list of supported models, please visit `vismatch documentation <https://vismatch.readthedocs.io/en/latest/>`_.

Getting Available Models
-------------------------

To list all available models:

.. code-block:: python

   from imcui.ui import get_available_model_names
   print(get_available_model_names())

Or via configuration:

.. code-block:: python

   from imcui.ui import get_matcher_zoo
   matcher_zoo = get_matcher_zoo()
   print(matcher_zoo.keys())

Model Categories
----------------

**Dense Matching**

Dense methods work on full image correlations and can match every pixel in the image.

* **roma** and **tiny-roma** - High-quality dense matching
* **duster** - Efficient dense matching
* **master** - Master matching algorithm
* **ufm** - Ultra-fast matching

**Semi-Dense Matching**

Semi-dense methods provide a balance between sparse and dense matching.

* **loftr** and variants (eloftr, se2loftr) - LoFTR-based matching
* **xoftr** - Cross-window optical flow transformer
* **aspanformer** - Adaptive span transformer
* **matchformer** - Transformer-based matching
* **xfeat-star** - XFeat with star topology

**Sparse Matching**

Sparse methods extract and match discrete keypoints.

* **superpoint-lightglue** - SuperPoint keypoints with LightGlue matcher
* **sift-lightglue** - SIFT keypoints with LightGlue matcher
* **disk-lightglue** - DISK keypoints with LightGlue matcher
* **dedode** - DeDoDe detector and descriptor
* **xfeat** - XFeat feature extractor
* **omniglue** - OmniGlue matcher
* **disk** - DISK corner detector
* **aliked** - ALIKED detector
* **darkfeat** - DarkFeat features


Model Details and Configurations
---------------------------------

Each model has its own unique configuration parameters. The matcher zoo provides default configurations for all models.

Accessing Model Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from imcui.ui import get_matcher_zoo

   matcher_zoo = get_matcher_zoo()
   conf = matcher_zoo["superpoint-lightglue"]

   # Model configuration structure
   print(conf.keys())
   # May include: name, model, preprocessing, device, etc.


Common Configuration Parameters
------------------------------

While each model has specific parameters, most models support these common options:

**Preprocessing Options**

* ``resize`` - Target image size (integer or tuple)
* ``grayscale`` - Convert images to grayscale (boolean)
* ``max_keypoints`` - Maximum number of keypoints to extract

**Device Options**

* ``device`` - Device to run the model on ('cpu' or 'cuda')

**Model-specific Options**

Each model may have additional parameters specific to its architecture. Refer to the individual model documentation for details.


Performance Considerations
--------------------------

When selecting a model, consider:

* **Speed vs. Accuracy**: Sparse methods are generally faster than dense methods
* **GPU Memory**: Dense methods require more GPU memory
* **Use Case**: Sparse methods are better for feature matching, dense methods are better for geometric tasks


Adding New Models
-----------------

.. note::

   This WebUI no longer maintains matching algorithms. To add new matchers, please contribute to the `vismatch <https://github.com/gmberton/vismatch>`_ repository.

If you've added a new matcher to vismatch, it will automatically be available in Image Matching WebUI after updating vismatch:

.. code-block:: bash

   pip install --upgrade vismatch


References
----------

For detailed information about each matching algorithm, its performance, and research papers, please refer to:

* `vismatch Model Documentation <https://vismatch.readthedocs.io/en/latest/model_details.html>`_
* `vismatch Repository <https://github.com/gmberton/vismatch>`_
* `Image Matching Workshop <https://image-matching-workshop.github.io>`_
