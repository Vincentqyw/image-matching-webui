API Reference
=============

ImageMatchingAPI
----------------

.. autoclass:: imcui.api.ImageMatchingAPI
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

Matching Functions
~~~~~~~~~~~~~~~~~~

.. autofunction:: imcui.ui.run_matching
.. autofunction:: imcui.ui.run_ransac

Geometry Functions
~~~~~~~~~~~~~~~~~~

.. autofunction:: imcui.ui.geometry.filter_matches
.. autofunction:: imcui.ui.geometry.compute_geometry
.. autofunction:: imcui.ui.geometry.process_ransac_matches

Visualization Functions
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: imcui.ui.visualization.display_keypoints
.. autofunction:: imcui.ui.visualization.display_matches
.. autofunction:: imcui.ui.visualization.plot_images

Image Processing Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: imcui.ui.image_utils.wrap_images
.. autofunction:: imcui.ui.image_utils.generate_warp_images

Matcher Management
------------------

.. autofunction:: imcui.ui.get_matcher_zoo
.. autofunction:: imcui.ui.get_available_model_names

Model Caching
-------------

.. autoclass:: imcui.ui.model_cache.ARCSizeAwareModelCache
   :members:
   :undoc-members:

.. autoclass:: imcui.ui.model_cache.LRUModelCache
   :members:
   :undoc-members:

Configuration Utilities
------------------------

.. autofunction:: imcui.ui.config_utils.get_default_config_path
.. autofunction:: imcui.ui.config_utils.get_example_data_path
.. autofunction:: imcui.ui.config_utils.get_version

Example Usage
-------------

Basic API Usage
~~~~~~~~~~~~~~~

.. code-block:: python

   from imcui.api import ImageMatchingAPI
   import cv2

   # Load images (RGB format expected)
   image0 = cv2.cvtColor(cv2.imread('image0.jpg'), cv2.COLOR_BGR2RGB)
   image1 = cv2.cvtColor(cv2.imread('image1.jpg'), cv2.COLOR_BGR2RGB)

   # Get matcher configuration
   from imcui.ui import get_matcher_zoo, DEVICE
   matcher_zoo = get_matcher_zoo()
   conf = matcher_zoo["superpoint-lightglue"]

   # Create API instance
   api = ImageMatchingAPI(conf=conf, device=DEVICE)

   # Run matching
   result = api(image0, image1)

   # Access results
   keypoints0 = result['keypoints0']
   keypoints1 = result['keypoints1']
   matches = result['matches']

With RANSAC Filtering
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from imcui.api import ImageMatchingAPI
   from imcui.ui import run_ransac, get_matcher_zoo, DEVICE
   import cv2

   # Load and process images
   image0 = cv2.cvtColor(cv2.imread('image0.jpg'), cv2.COLOR_BGR2RGB)
   image1 = cv2.cvtColor(cv2.imread('image1.jpg'), cv2.COLOR_BGR2RGB)

   # Create API and run matching
   matcher_zoo = get_matcher_zoo()
   conf = matcher_zoo["superpoint-lightglue"]
   api = ImageMatchingAPI(conf=conf, device=DEVICE)
   result = api(image0, image1)

   # Apply RANSAC filtering
   ransac_result = run_ransac(
       image0, image1,
       result['keypoints0'],
       result['keypoints1'],
       result['matches'],
       method='MAGSAC',
       reproj_threshold=1.0
   )

   # Filtered matches
   inliers = ransac_result['matches']

Custom Matcher Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from imcui.api import ImageMatchingAPI

   # Create custom configuration
   conf = {
       'name': 'MyCustomMatcher',
       'model': {
           'detector': 'superpoint',
           'matcher': 'lightglue'
       },
       'preprocessing': {
           'resize': 1024
       }
   }

   # Create API with custom config
   api = ImageMatchingAPI(conf=conf, device='cuda')

   # Use the API
   result = api(image0, image1)
