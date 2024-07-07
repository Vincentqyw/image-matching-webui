python -m flake8 common/*.py hloc/*.py hloc/matchers/*.py hloc/extractors/*.py 
python -m isort common/*.py hloc/*.py hloc/matchers/*.py hloc/extractors/*.py --check-only --diff
python -m black common/*.py hloc/*.py hloc/matchers/*.py hloc/extractors/*.py --check --diff