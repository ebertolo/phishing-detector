"""Make the ``src`` layout importable when Streamlit runs from the repo root.

Streamlit executes ``app/streamlit_app.py`` and ``app/pages/*.py`` directly, so
the ``src`` directory is not automatically on ``sys.path``. Importing this module
(``import _bootstrap``) at the top of each page adds it once. In the container the
package is pip/uv-installed, so this is a no-op there.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
