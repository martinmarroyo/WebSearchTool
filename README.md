# Web Search Tool

A tool that searches the web for you and gives you a direct answer to your question along with the sources it used. Powered by Langchain and DuckDuckGo. 

## Parameters

- `llm` (BaseChatModel| Required): The LLM Chat Model that you would like to use (e.g. `ChatOpenAI`). 
                                    See https://python.langchain.com/v0.1/docs/integrations/chat/ for the full list of supported models.
- `session` (requests.Session| Optional): A Session object used to connection pool our requests to websites. If not provided, one is created by the tool.
- `user_agent` (str| Optional): A User-Agent to be used when navigating the web. The default is a randomized User-Agent.
- `k` (int | Optional): Defines the number of ranked search results to return from our ranking process. Default is to return the top 5 relevant documents.
- `summary_prompt` (str | Optional): The prompt used to instruct the LLM on how it should approach summarizing the results. There is a default prompt provided.

## Usage

**Passing in a `Session` object:**

```python
with requests.Session() as session:
    WST = WebSearchTool(llm=llm, session=session)
    answer = WST.search("What does it take to become an AI engineer?")
    print(answer.response)
```

**Omitting a `Session` object:**

```python
with WebSearchTool(llm=llm) as WST:
    answer = WST.search("What does it take to become an AI engineer?")
    print(answer.response)
```