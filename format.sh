python -m flake8 ui/*.py api/*.py hloc/*.py hloc/matchers/*.py hloc/extractors/*.py 
python -m isort ui/*.py api/*.py hloc/*.py hloc/matchers/*.py hloc/extractors/*.py
python -m black ui/*.py api/*.py hloc/*.py hloc/matchers/*.py hloc/extractors/*.py