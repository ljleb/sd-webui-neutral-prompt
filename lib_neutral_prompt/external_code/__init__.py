import contextlib


@contextlib.contextmanager
def fix_path():
    import sys
    from pathlib import Path

    extension_path = str(Path(__file__).parent.parent.parent)
    added = False
    if extension_path not in sys.path:
        sys.path.insert(0, extension_path)
        added = True

    yield

    if added:
        sys.path.remove(extension_path)


with fix_path():
    del fix_path
    from .api import *
