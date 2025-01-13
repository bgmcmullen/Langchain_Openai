from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate

from langchain.vectorstores.base import VectorStore

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
import os

from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"]  = os.getenv("OPENAI_API_KEY")


# Initialize the OpenAI chat model
chat_model = ChatOpenAI(
    model="gpt-4o-mini",  # Specify your model
    temperature=0.7  # Adjust as needed for tone
)

# Define the system instructions
system_instructions = """
Speak in a warm and friendly, child-appropriate manner as a supportive teacher to K-12 children. 
If Bridging Tech has resources to help answer the question, provide all the links first in your response. 
Refer to Bridging Tech in the first person. Refuse to give any response not appropriate for young children. 
Refuse to answer any questions not directly related to education or appropriate for a classroom. 
Avoid including information related to purchasing non-educational items. 
Keep responses to 100 words or less.
"""

# Create a system message prompt template
system_prompt = SystemMessagePromptTemplate.from_template(system_instructions)

# Initialize the vector store (using FAISS here, but replace with your own setup if needed)
# vector_store = FAISS.load_local("vector_store_path", OpenAIEmbeddings())

# Define a tool for the vector store search
file_search_tool = VectorStoreTool(
    name="file_search",
    description="Search files and provide relevant results.",
    # vectorstore=vector_store
)

# Create a conversational retrieval chain
retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=chat_model,
    # retriever=vector_store.as_retriever(),
    return_source_documents=True  # Include source documents in responses if needed
)

# Example query
query = "Can you recommend resources for learning fractions?"

# Call the chain
response = retrieval_chain.run(
    query=query,
    chat_history=[],  # Pass prior conversation history if needed
)

# Print the response
print(response)
