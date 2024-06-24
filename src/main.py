from tools.web_search_tool import WebSearchTool
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from typing import List
import sys

def run_chat(llm: BaseChatModel, user_agent: str):
        
    with WebSearchTool(llm=llm, user_agent=user_agent) as WST:
        while True:
            question = input("Ask me a question or have me look something up for you: ")

            if question.lower() in ["exit", "quit", "/q", "stop"]:
                break
            try:
                answer = WST.search(question)

                print(answer.response)
            except AttributeError:
                print("Sorry, there were no search results for your question. Please try again.")
                continue 

if __name__ == "__main__":
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-1106")
    print("Starting chat...")
    run_chat(llm=llm, user_agent=user_agent)
