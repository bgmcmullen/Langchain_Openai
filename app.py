import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import numpy as np
from annoy import AnnoyIndex
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
from flask import Flask, request, jsonify
from flask_cors import CORS
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import trim_messages
from langchain_core.messages import RemoveMessage

workflow = StateGraph(state_schema=MessagesState)

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return jsonify("Hello World!")

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the single document
file_path = "about-BT/About-Bridging-tech.txt"
loader = TextLoader(file_path)
documents = loader.load()

with open(file_path, 'r') as file:
    data = file.read()

sentences = sent_tokenize(data)

cached_embeddings = [model.encode(sentence) for sentence in sentences]

# Initialize Annoy index
dimension = len(cached_embeddings[0])
annoy_index = AnnoyIndex(dimension, 'angular')  # Angular distance for similarity

# Add embeddings to Annoy index
for i, embedding in enumerate(cached_embeddings):
    annoy_index.add_item(i, embedding)

# Build the index
annoy_index.build(10)  # Number of trees

# Split the document into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(split_docs, embeddings)



def trim(messages):
    if len(messages) > 2:
        messages = messages[-2:]
    return messages

# Define the retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})  # Retrieve the most relevant chunk

# System prompt for assistant behavior
system_prompt = SystemMessagePromptTemplate.from_template(
    "Speak in a warm and frendly and child appropriate manor as supportive teacher to K-12 children. You work for briding tech to refer to briding tech in first person. Refuse to give any response not appropriate for young children. Avoid including information related to purchasing non-education items. Provide url links when they are given in context, never provide any other links. If not given a link refuse to provide a link if asked. Keep response to 100 words or less and respond in markdown."
)

# Human input prompt
human_prompt = HumanMessagePromptTemplate.from_template("{question}")

# Combine system and human prompts
chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

# trimmer = trim_messages(strategy="last", max_tokens=2, token_counter=len)

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=1.0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Create a simple LLM chain using the chat prompt
# qa_chain = LLMChain(
#     llm=llm,
#     prompt=chat_prompt
# )


def delete_messages(state):
    messages = state["messages"]
    return {"messages": [RemoveMessage(id=m.id) for m in messages[:-2]]}


# Define the function that calls the model
def call_model(state: MessagesState):
    # state["messages"] = summarize_history(state["messages"])
    
    # if len(state["messages"]) > 2:
    #     compiled_workflow.update_state(config, {"messages": [HumanMessage("Test")]})
    # print(state)
    system_prompt = (
        "Speak in a warm and frendly and child appropriate manor as supportive teacher to K-12 children. You work for briding tech to refer to briding tech in first person. Refuse to give any response not appropriate for young children. Avoid including information related to purchasing non-education items. Provide url links when they are given in context, never provide any other links. If not given a link refuse to provide a link if asked. Keep response to 100 words or less and respond in markdown and Latex in $$ for equations."
    )

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    # Set the temperature for the model
    temperature = 1.0  # Adjust this value as needed
    response = llm.invoke(messages)
    return {"messages": response}


# Define the node and edge
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# Add simple in-memory checkpointer
memory = MemorySaver()
compiled_workflow = workflow.compile(checkpointer=memory)


# Retrieve the most relevant chunk from the single document

@app.route("/chat", methods=["POST"])
def chat():
    new_prompt = request.json.get("input")
    threadId = request.json.get("threadId")



    print(threadId)

    
    
    # Encode the new prompt
    new_embedding = model.encode(new_prompt)

    

    # Find the closest match
    k = 10  # Number of nearest neighbors
    indices = annoy_index.get_nns_by_vector(new_embedding, k, include_distances=True)
    min_distance = indices[1][0]

    context = ''

    config = {"configurable": {"thread_id": threadId}}

    # Retrieve matched prompts
    for idx, distance in zip(indices[0], indices[1]):
        if(distance < 1.2):
            context += f"{sentences[idx]} "


    # Combine the retrieved context with the user query
    response = compiled_workflow.invoke({"messages": [HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{new_prompt}")]},
    config={"configurable": {"thread_id": threadId}})


    # Extract the AI response
    ai_response = next(
        (message.content for message in reversed(response['messages']) if isinstance(message, AIMessage)),
        None  # Default to None if no AIMessage is found
    )

    # print(response)

    

    messages = compiled_workflow.get_state(config).values["messages"]

    

    if len(messages) > 4:
        compiled_workflow.update_state(config, {"messages": RemoveMessage(id=messages[0].id)})
        compiled_workflow.update_state(config, {"messages": RemoveMessage(id=messages[1].id)})

    print(messages)


    return ai_response

if __name__ == "__main__":
    app.run(debug=True)
