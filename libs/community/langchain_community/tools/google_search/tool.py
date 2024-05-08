"""Tool for the Google search API."""

import json
from typing import Any, List, Optional
from langchain_community.document_loaders.chromium import AsyncChromiumLoader
from langchain_community.document_transformers.html2text import Html2TextTransformer

from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents.base import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool

from langchain_community.utilities.google_search import GoogleSearchAPIWrapper


@deprecated(
    since="0.0.33",
    removal="0.2.0",
    alternative_import="langchain_google_community.GoogleSearchRun",
)
class GoogleSearchRun(BaseTool):
    """Tool that queries the Google search API."""

    name: str = "google_search"
    description: str = (
        "A wrapper around Google Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )
    api_wrapper: GoogleSearchAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)


@deprecated(
    since="0.0.33",
    removal="0.2.0",
    alternative_import="langchain_google_community.GoogleSearchResults",
)
class GoogleSearchResults(BaseTool):
    """Tool that queries the Google Search API and gets back json."""

    name: str = "google_search_results"
    description: str = (
        "A wrapper around Google Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query. Output is a JSON array of the query results"
    )
    num_results: int = 4
    api_wrapper: GoogleSearchAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:

        print("SEARCH_QUERY: ", query) # DEBUG

        search_results = self.api_wrapper.results(query, self.num_results)

        print("SEARCH_RESULTS: ", search_results) # DEBUG

        if len(search_results) == 1 and "Result" in search_results[0]:
            return json.dumps(["No search results found."], ensure_ascii=False)

        return json.dumps(search_results, ensure_ascii=False)

class GoogleSearchRetriever(BaseRetriever):
    """Tool that queries the Google Search API and gets back json."""

    name: str = "google_search_retriever"
    description: str = (
        "A wrapper around Google Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query. Output is a Document list of the query results"
    )
    num_results: int = 2
    api_wrapper: GoogleSearchAPIWrapper

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:

        print("SEARCH_QUERY: ", query) # DEBUG

        search_results = self.api_wrapper.results(query, self.num_results)

        print("SEARCH_RESULTS: ", search_results) # DEBUG

        if len(search_results) == 1 and "Result" in search_results[0]:
            return [Document(page_content="No search results found.")]

        loader = AsyncChromiumLoader([result["link"] for result in search_results])
        htmls = loader.load()
        html2text = Html2TextTransformer()
        docs = html2text.transform_documents(htmls)

        return [doc for doc in docs]
