"""
AI Agent using LangChain's dynamic tool calling system.
This version uses LangChain agents for truly dynamic tool selection.
"""
import os
import logging
from dotenv import load_dotenv
from pydantic import SecretStr

from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from tool_loader import ToolLoader

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LangChainAIAgent:
    """An AI agent that uses LangChain's dynamic tool calling system."""
    
    def __init__(self):
        """Initialize the AI agent with Azure OpenAI and dynamic tools."""
        self.llm = self._setup_llm()
        self.tool_loader = ToolLoader()
        self.agent_executor = self._setup_agent()
        self.conversation_history = []
        
    def _setup_llm(self) -> AzureChatOpenAI:
        """Setup Azure OpenAI LLM."""
        try:
            # Check for required environment variables
            required_vars = [
                'AZURE_OPENAI_API_KEY',
                'AZURE_OPENAI_ENDPOINT',
                'AZURE_OPENAI_DEPLOYMENT_NAME'
            ]
            
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                raise ValueError(f"Missing required environment variables: {missing_vars}")
            
            api_key_str = os.getenv('AZURE_OPENAI_API_KEY')
            llm = AzureChatOpenAI(
                api_key=SecretStr(api_key_str) if api_key_str else None,
                azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview'),
                azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
                temperature=0.1,
                max_tokens=1500,
            )
            
            logger.info("✅ Azure OpenAI LLM initialized successfully")
            return llm
        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure OpenAI: {str(e)}")
            raise
    
    def _setup_agent(self) -> AgentExecutor:
        """Setup LangChain agent with dynamic tools."""
        try:
            # Get LangChain tools from configuration
            tools = self.tool_loader.get_langchain_tools()
            
            if not tools:
                logger.warning("⚠️ No LangChain tools found. Agent will work with LLM only.")
            
            # Create agent prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful AI assistant with access to various tools for gathering information.
                
When answering questions:
1. Think step by step about what information you need
2. Use the available tools to gather relevant information
3. If you need current/recent information, use the web search tool
4. If you need factual/encyclopedic information, use the Wikipedia search tool
5. If you need detailed content from a specific web page, use the content extraction tool
6. Provide comprehensive answers based on the information gathered
7. Always cite your sources when using tool results
8. If you can't find relevant information with the tools, say so honestly

Available tools: {tool_names}
"""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            # Create the agent
            agent = create_openai_tools_agent(
                llm=self.llm,
                tools=tools,
                prompt=prompt
            )
            
            # Create agent executor
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=self.tool_loader.get_agent_config().get('max_iterations', 3),
                return_intermediate_steps=True
            )
            
            tool_names = [tool.name for tool in tools]
            logger.info(f"✅ LangChain agent initialized with tools: {tool_names}")
            return agent_executor
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LangChain agent: {str(e)}")
            raise
    
    def process_query(self, query: str) -> str:
        """Process a user query using LangChain agent."""
        try:
            print(f"\n🤖 Processing your query: '{query}'")
            print("=" * 60)
            
            # Prepare chat history for the agent
            chat_history = []
            for entry in self.conversation_history[-5:]:  # Keep last 5 exchanges
                chat_history.append(HumanMessage(content=entry['query']))
                chat_history.append(AIMessage(content=entry['answer']))
            
            # Let the agent decide which tools to use
            logger.info("🧠 Agent analyzing query and selecting tools...")
            
            result = self.agent_executor.invoke({
                "input": query,
                "chat_history": chat_history,
                "tool_names": [tool.name for tool in self.tool_loader.get_langchain_tools()]
            })
            
            answer = result.get('output', 'I was unable to process your query.')
            
            # Add to conversation history
            self.conversation_history.append({
                'query': query,
                'answer': answer,
                'intermediate_steps': result.get('intermediate_steps', [])
            })
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {str(e)}")
            error_msg = f"I encountered an error while processing your query: {str(e)}"
            
            # Still add to history for debugging
            self.conversation_history.append({
                'query': query,
                'answer': error_msg,
                'intermediate_steps': []
            })
            
            return error_msg
    
    def show_tool_status(self):
        """Display the status of all available tools."""
        print("\n🔧 Tool Status:")
        print("=" * 40)
        
        langchain_tools = self.tool_loader.get_langchain_tools()
        if not langchain_tools:
            print("❌ No LangChain tools are currently enabled.")
            return
        
        for tool in langchain_tools:
            print(f"✅ {tool.name}: {tool.description}")
        
        print("\n📋 Agent Configuration:")
        agent_config = self.tool_loader.get_agent_config()
        for key, value in agent_config.items():
            print(f"   {key}: {value}")
        
        print("\n🔍 Search Strategy:")
        search_strategy = self.tool_loader.get_search_strategy()
        for key, value in search_strategy.items():
            print(f"   {key}: {value}")
        print("=" * 40)
    
    def run_interactive_loop(self):
        """Run the interactive loop for user queries."""
        
        print("🤖 AI Agent with Dynamic Tool Calling")
        print("=" * 50)
        print("I'm an AI agent that can dynamically select and use tools")
        print("to answer your questions. Ask me anything!")
        print("\nCommands:")
        print("- Type 'quit' or 'exit' to stop")
        print("- Type 'history' to see conversation history")
        print("- Type 'clear' to clear conversation history")
        print("- Type 'tools' to see available tools and their status")
        print("- Type 'reload' to reload tool configuration")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n💬 Your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye! Thanks for using the AI Agent!")
                    break
                
                elif user_input.lower() == 'history':
                    self._show_history()
                    continue
                
                elif user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("🗑️  Conversation history cleared!")
                    continue
                
                elif user_input.lower() == 'tools':
                    self.show_tool_status()
                    continue
                
                elif user_input.lower() == 'reload':
                    try:
                        self.tool_loader.reload_config()
                        # Recreate agent with new configuration
                        self.agent_executor = self._setup_agent()
                        print("🔄 Tool configuration reloaded successfully!")
                    except Exception as e:
                        print(f"❌ Failed to reload configuration: {str(e)}")
                    continue
                
                elif not user_input:
                    print("⚠️  Please enter a question or command.")
                    continue
                
                # Process the query
                answer = self.process_query(user_input)
                
                print("\n" + "=" * 60)
                print("📋 FINAL ANSWER:")
                print("=" * 60)
                print(answer)
                print("=" * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for using the AI Agent!")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {str(e)}")
                logger.error(f"Error in interactive loop: {str(e)}")
    
    def _show_history(self):
        """Show conversation history."""
        if not self.conversation_history:
            print("📝 No conversation history yet.")
            return
        
        print("\n📝 Conversation History:")
        print("-" * 40)
        
        for i, entry in enumerate(self.conversation_history, 1):
            print(f"\n{i}. Query: {entry['query'][:100]}...")
            print(f"   Answer: {entry['answer'][:200]}...")
            
            # Show intermediate steps if available
            if entry.get('intermediate_steps'):
                print(f"   Tools used: {len(entry['intermediate_steps'])} steps")
            
            print("-" * 40)


def main():
    """Main function to run the AI agent."""
    try:
        # Check for required environment variables
        required_vars = [
            'AZURE_OPENAI_API_KEY',
            'AZURE_OPENAI_ENDPOINT',
            'AZURE_OPENAI_DEPLOYMENT_NAME'
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print("❌ Missing required environment variables:")
            for var in missing_vars:
                print(f"   - {var}")
            print("\nPlease copy .env.example to .env and fill in your Azure OpenAI credentials.")
            return
        
        # Initialize and run the agent
        agent = LangChainAIAgent()
        agent.run_interactive_loop()
        
    except Exception as e:
        logger.error(f"Failed to start AI agent: {str(e)}")
        print(f"❌ Failed to start AI agent: {str(e)}")


if __name__ == "__main__":
    main()
