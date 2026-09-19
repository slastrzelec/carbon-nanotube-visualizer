"""
Empty on purpose. Its presence makes pytest add this project's root directory
to sys.path during test collection, so `tests/test_*.py` can `from lib import
...` regardless of whether pytest is invoked as `pytest`, `python -m pytest`,
from this folder, from CI, or from an IDE's test runner. (Same fix applied to
erythrocyte-shape-analyzer after hitting this exact issue there.)
"""
