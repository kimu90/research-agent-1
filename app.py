import logging
import dotenv
dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

from research_agent import ResearchAgent
from tools import *

# Define the tools
tools = [YouComSearch(), SimilarWebSearch(), ExaCompanySearch(), NewsSearch()]

# Create an instance of the ResearchAgent class and pass the tools list to it
research_agent = ResearchAgent(tools)

# Create a simple mock context for demonstration
class MockContext:
    def new_message(self):
        return MockMessage()

class MockMessage:
    def add(self, type, text=""):
        print(f"Message added: {text}")
        return self

    def notify(self):
        print("Message notified")

def main():
    # Demonstrate the research agent with a sample query
    print("Starting research agent demo...")
    
    # Example queries for different tools
    queries = [
        {"query": "What is the latest trend in artificial intelligence?", "tool": handle_you_com_search},
        {"query": "Anthropic company overview", "tool": handle_similar_web_search},
        {"query": "OpenAI recent developments", "tool": handle_exa_company_search},
        {"query": "AI technology news", "tool": handle_news_search}
    ]

    for query_info in queries:
        print(f"\nPerforming research on: {query_info['query']}")
        try:
            context = MockContext()
            query_info['tool'](context, query=query_info['query'])
        except Exception as e:
            print(f"Error during research: {e}")

    print("\nResearch demo completed.")

# These functions simulate event handlers
def handle_you_com_search(context, **kwargs):
    result = YouComSearch(include_summary=True).invoke(input=kwargs)
    m = context.new_message()
    m.add("text", text=result.summary)
    m.notify()

def handle_similar_web_search(context, **kwargs):
    result = SimilarWebSearch(include_summary=True).invoke(input=kwargs)
    m = context.new_message()
    m.add("text", text=result.summary)
    m.notify()

def handle_exa_company_search(context, **kwargs):
    result = ExaCompanySearch(include_summary=True).invoke(input=kwargs)
    m = context.new_message()
    m.add("text", text=result.summary)
    m.notify()

def handle_news_search(context, **kwargs):
    result = NewsSearch(include_summary=True).invoke(input=kwargs)
    m = context.new_message()
    m.add("text", text=result.summary)
    m.notify()

if __name__ == "__main__":
    main()