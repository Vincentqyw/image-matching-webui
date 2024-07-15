python -m flake8 ui/*.py hloc/*.py hloc/matchers/*.py hloc/extractors/*.py 
python -m isort ui/*.py hloc/*.py hloc/matchers/*.py hloc/extractors/*.py
python -m black ui/*.py hloc/*.py hloc/matchers/*.py hloc/extractors/*.py