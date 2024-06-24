# Strategy      
# Convert search results to documents
# Use bm25 retriever to get the most relevant results based on the query
# From those results, try to parse the data from the websites that are linked
# Convert the parsed data into Document objects

import requests
from bs4 import BeautifulSoup, Tag
from fake_useragent import UserAgent
from requests import Session
from time import sleep
from loguru import logger
from langchain_core.documents import Document
from langchain_core.runnables.base import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from typing import List, Dict, Union, Optional, Any
from urllib.parse import urlparse, parse_qs, unquote
from langchain_core.pydantic_v1 import BaseModel, Field


class WebSearchSource(BaseModel):
    headline: str = Field(None, description="The headline tag for the source.")
    url: str = Field(None, description="The URL for the source.")

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "WebSearchSource":
        return cls(**data)
        
class WebSearchResult(BaseModel):
    response: str = Field(None, description="The final, summarized response from the web search.")
    sources: List[WebSearchSource] = Field(None, description="The page title and URLs used to generate the response.")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Union[str, WebSearchSource]]) -> "WebSearchResult":
        return cls(**data)
    
class WebSearchTool:
    """A tool that searches the web for you and gives you a direct answer to your question along with the sources it used. Powered by Langchain and DuckDuckGo. 

    ### Parameters

    - `llm` (BaseChatModel| Required): The LLM Chat Model that you would like to use (e.g. `ChatOpenAI`). 
                                       See https://python.langchain.com/v0.1/docs/integrations/chat/ for the full list of supported models.
    - `session` (requests.Session| Optional): A Session object used to connection pool our requests to websites. If not provided, one is created by the tool.
    - `user_agent` (str| Optional): A User-Agent to be used when navigating the web. The default is a randomized User-Agent.
    - `k` (int | Optional): Defines the number of ranked search results to return from our ranking process. Default is to return the top 5 relevant documents.
    - `summary_prompt` (str | Optional): The prompt used to instruct the LLM on how it should approach summarizing the results. There is a default prompt provided.

    ### Usage
    
    Passing in a session object:
    
    ```python
    with requests.Session() as session:
        WST = WebSearchTool(llm=llm, session=session)
        answer = WST.search("What does it take to become an AI engineer?")
        print(answer.response)
    ```

    Omitting a session object:

    ```python
    with WebSearchTool(llm=llm) as WST:
        answer = WST.search("What does it take to become an AI engineer?")
        print(answer.response)
    ```
    """
    def __init__(self, llm: BaseChatModel, session: Optional[Session] = None, user_agent: str = None, k: int = 5, summary_prompt: str = None):
        self.llm: BaseChatModel = llm
        self.base_url: str = "https://html.duckduckgo.com/html/?q={query}"
        self.user_agent: str = user_agent or UserAgent().random
        self.session: Session = session or Session()
        self.k: int = k
        self.default_summary_prompt: str = """You are a conversational, expert web researcher who is skilled in searching the web and synthesizing the results of 
                                           your search.
    
                                         Task:
    
                                         Take user input in the form of a collection of HTML search results and generate a summary based on 
                                         what you have found there.
    
                                         Instructions:
                                         - Do not make up any answers. Only use the information that is provided as input to generate your response.
                                         - Your response should be no more than 5 sentences long.
                                         - Be succinct and don't leave out anything important.
                                         - Generate your answer based on its relevance to the original question.
                                         - Parse the HTML as best as possible
                                         - Remove all HTML tags from your output
                                         - Phrase your answers in a conversational style always
                                         """
        self.prompt_template: str = summary_prompt or self.default_summary_prompt
        self.summary_chain: Runnable = self._create_summary_chain(self.prompt_template, self.llm)
        self.current_question: str = None
        self.last_response: str = None
        self.latest_ranked_documents: List[Document] = None
        self.latest_final_documents: List[Document] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.session.close()
    
    def search(self, question: str, user_agent: Optional[str] = None, delay: Optional[int] = 1, **header_params) -> WebSearchResult:
        """Perform a web search and get back a structured response

        Args:
            question (str): Your web search term or question
            user_agent (str | Optional): A User-Agent to use for sending requests. If not provided, a randomized agent is supplied by the tool.
            delay (int | Optional): The number of seconds to wait between searches for ranked results. Default is 1 second between each additional search.
            header_params (**kwargs): Additional parameters you wish to supply to the header for all requests
        Returns:
            A valid WebSearchResult object.
            
        """
        # Set up the initial search
        self.current_question: str = question
        query: str = question.replace(" ", "%20")
        url: str = self.base_url.format(query=query)
        headers: Dict[str, str] = {"User-Agent": user_agent or self.user_agent}
        headers.update(header_params)
        # Get the initial search results
        initial_search_results: List[Document] = self._get_search_results(url=url, headers=headers)
        # Get the top 5 ranked results
        ranked_results: List[Document] = self._rank_search_results(query=query, search_results=initial_search_results)
        self.latest_ranked_documents = ranked_results
        if ranked_results:
            # Extract the detail summaries from those results
            content_with_detail_summaries: List[Document] = self._parse_ranked_results(ranked_results, delay)
            self.latest_final_documents = content_with_detail_summaries
            sources: List[WebSearchSource] = self._extract_sources_from_metadata(content_with_detail_summaries)
            source_text: str = self._source_to_text(sources)
            response: str = f"{self._generate_final_response(content_with_detail_summaries)}\n\n{source_text}"
            search_result: WebSearchResult = WebSearchResult.from_dict({"response": response, "sources": sources})
            return search_result
        logger.info("There are no search results to rank. Ending search...")


    def _extract_sources_from_metadata(self, docs: List[Document]) -> List[WebSearchSource]:
        """Extracts source information from Document metadata and returns a list of validated WebSearchSource objects."""
        sources = [WebSearchSource.from_dict(
                        {"headline": doc.metadata['title'], 
                         "url": doc.metadata['url']}) 
                   for doc in docs]
        return sources

    
    def _source_to_text(self, sources: List[WebSearchSource]) -> str:
        """Takes a list of WebSearchSource objects and translates them into a list of 
            sources to be appended as text to the end of a response."""
        source_text = "\n".join([f"{source.headline}: {source.url}" for source in sources])
        result = f"Sources:\n{source_text}"
        return result
        
        
    def _create_summary_chain(self, prompt_template: str, llm: BaseChatModel) -> Runnable:
        """Creates the LLM chain used for summarizing documents and generating a response"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_template),
            ("human", "{question} {document}")
        ])
        summary_chain = prompt | llm | StrOutputParser()
        return summary_chain

    
    def _process_search_result(self, result: Tag) -> Union[Document, None]:
        """Processes a raw search result into a Langchain Document."""
        title_tag = result.find("a", class_="result__a")
        description_tag = result.find("a", class_="result__snippet")
        if all([title_tag, description_tag]):
            title = title_tag.get_text()
            link = self._extract_navigable_link(title_tag["href"])
            description = description_tag.get_text()
            doc = Document(
                page_content = f"{title}:\nDescription: {description}\nURL:{link}",
                metadata = {"title": title, "desc": description, "url": link}
            )
            return doc

        return None

    
    def _get_search_results(self, url: str, headers: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Searches the web using the supplied url and headers, then returns a list of search results as Documents."""
        # Fetch the HTML content
        response = self.session.get(url, headers=headers)
        # Raise an error for bad status codes
        if response.status_code != 200:
            response.raise_for_status()
        if response.status_code == 202:
            logger.info("Getting blocked by DuckDuckGo... consider retrying later...")
            logger.info(f"Status Code: {response.status_code}\n\nContent:\n\n{response.content}")
        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find all search results
        results = soup.find_all('div', class_='result__body')
        # Extract titles and descriptions
        search_results = [self._process_search_result(result) for result in results]
        filtered_results = list(filter(lambda doc: doc is not None, search_results))
        
        return filtered_results

    
    def _rank_search_results(self, 
                            query: str, 
                            search_results: List[Document]) -> Union[List[Document], None]:
        """Takes in a list of search results and finds the top K most relevant 
           results based on the given query using the BM25 algorithm."""
        if search_results:
            retriever = BM25Retriever.from_documents(search_results)
            ranked_docs = retriever.invoke(query)
            return ranked_docs[:self.k]
        logger.info("No search results found")
        return None

    
    def _heuristic_page_parse(self, content: str) -> str:
        """Uses basic heuristics to attempt to scrape the content from a search result for additional information."""
        soup = BeautifulSoup(content, "html.parser")
        # Heuristically identify main content
        main_content = soup.find_all(['h1', 'h2', 'p', 'a'])
        combined_content = " ".join([content.get_text() 
                                     for content in main_content]) 
        
        return combined_content

    
    def _extract_navigable_link(self, duckduckgo_url: str) -> str:
        """Takes in a DuckDuckGo url and strips away the metacharacters to return the original link."""
        # Parse the URL
        parsed_url = urlparse(duckduckgo_url)
        # Extract query parameters
        query_params = parse_qs(parsed_url.query)
        # Get the 'uddg' parameter which contains the actual URL
        if 'uddg' in query_params:
            navigable_url = query_params['uddg'][0]
            # Decode the URL-encoded string
            return unquote(navigable_url)
        
        return None
        

    def _parse_result(self, 
                      result: Document, 
                      session: Session, 
                      delay: int = 1, 
                      headers: Dict[str, Any] = None) -> Union[Document, None]:
        """Heuristically parses a search result url to add more context to the given Document."""
        parsed_result = None
        try:
            url = result.metadata["url"]
            if url:
                content = session.get(url, headers=headers)
                parsed_content = self._heuristic_page_parse(content.text)
                combined_content = f"{result.page_content}\n{parsed_content}"
                summarized_content = self.summary_chain.invoke({
                    "question": self.current_question,
                    "document": combined_content
                })
                parsed_result = Document(page_content=summarized_content, metadata=result.metadata)

        except KeyError:
            logger.info("Missing `url` from metadata")

        except AttributeError:
            logger.info("Document not initialized.")
        # Pause for delay seconds between requests
        sleep(delay)
        return parsed_result
        

    def _parse_ranked_results(self, ranked_results: List[Document], delay: int = 1) -> List[Document]:
        """Parses a list of ranked search results and scrapes each page to add more detail to our final response."""
        ranked_parsed_docs = []
        headers = {"User-Agent": self.user_agent}
        parsed_results = [self._parse_result(result, self.session, delay, headers)
                          for result in ranked_results]
        filtered_results = list(filter(lambda doc: doc is not None, parsed_results))
        return filtered_results

    
    def _generate_final_response(self, final_results: List[Document]) -> str:
        """Takes the final results and generates a response to the current question."""
        final_result = self.summary_chain.invoke({
            "question": self.current_question,
            "document": final_results
        })
        return final_result
