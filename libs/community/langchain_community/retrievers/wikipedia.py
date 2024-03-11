from pdb import run
from typing import List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_core.tools import BaseTool


class WikipediaRetriever(BaseTool, WikipediaAPIWrapper):
    """`Wikipedia API` retriever.

    It wraps load() to get_relevant_documents().
    It uses all WikipediaAPIWrapper arguments without any change.
    """

    name = "wikipedia"
    description = "Search for a query on Wikipedia"

    def _run(
        self, query: str
    ) -> List[Document]:
        return self.load(query=query)
