"""Local HTTP API that orchestrates the engine and evidence layers for the review application.

The API process never loads models or runs analysis; see ``README.md`` in this package.
"""

from spectra import __version__ as API_VERSION

__all__ = ["API_VERSION"]
