# D:\Tinnitus-Search\tinnitus_search\index\chroma_store_mcp.py
from __future__ import annotations

from typing import Any, Dict, List

from tinnitus_search.index.chroma_store import ChromaStore


class ChromaStoreMCP(ChromaStore):
    """
    MCP adapter over the existing ChromaStore.

    Goal:
      - Keep original project API intact
      - Add a fast "get by ids" method for MCP tools (chunk_id -> exact doc+metadata)
    """

    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """
        Fetch exact documents+metadatas by Chroma 'ids'.

        Returns ChromaDB GetResult-like dict (ids/documents/metadatas/...).
        """
        if not hasattr(self, "collection"):
            raise AttributeError(
                "ChromaStoreMCP expects self.collection to exist. "
                "Your ChromaStore should keep a reference to the Chroma collection."
            )

        # Most reliable include-set for your case
        return self.collection.get(ids=ids, include=["documents", "metadatas"])