"""Context compression using retrieval heads.

This module exposes a convenience function around the QR‑HEAD reducer defined
in :mod:`app.services.qr_retriever`.  It returns the condensed context
alongside the number of tokens before and after compression, which are
useful for rate limiting and logging.
"""

from typing import Tuple

from .services import qr_retriever


def compress(query: str, context: str) -> Tuple[str, int, int]:
    """Run the QR‑HEAD reducer to condense context.

    Parameters
    ----------
    query: str
        The user's current query.
    context: str
        Concatenation of previous messages (if any).

    Returns
    -------
    tuple[str, int, int]
        The condensed context, the number of tokens before compression,
        and the number of tokens after compression.
    """
    condensed, before, after = qr_retriever.reduce(query, context)
    return condensed, before, after