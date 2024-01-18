from flask import Flask, request, jsonify
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from langchain.memory import CassandraChatMessageHistory, ConversationBufferMemory
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()

cloud_config = {
    'secure_connect_bundle': os.environ.get("BUNDLE_NAME")
}

with open(os.environ.get("TOKEN_NAME")) as f:
    secrets = json.load(f)

CLIENT_ID = secrets["clientId"]
CLIENT_SECRET = secrets["secret"]
ASTRA_DB_KEYSPACE = os.environ.get("KEYSPACE_NAME")
auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()

Message_history = CassandraChatMessageHistory(
    session_id="APVS",
    session=session,
    keyspace=ASTRA_DB_KEYSPACE,
    ttl_seconds=3600
)

Message_history.clear()
cass_buff_memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=Message_history
)

template = """
You are now interacting with the knowledge bot named AI. Please provide your query or request based on the chat history for a more context-aware response.
Chat History: {chat_history}
User: {input}
AI:
"""

prompt = PromptTemplate(
    input_variables=["chat_history", "input"],
    template=template
)

llm = ChatGoogleGenerativeAI(model="gemini-pro", api_key=os.environ.get("GOOGLE_API_KEY"))

llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=cass_buff_memory
)


@app.route('/ask', methods=['get'])
def ask_query():
    data = request.json
    query = data.get('query', 'Start to ask query')

    response = llm_chain.invoke(input=query)

    if response['input'] == "exit":
        return jsonify({'response': response['text'], 'exit': True})

    return jsonify({'response': response['text'], 'exit': False})


if __name__ == '__main__':
    app.run(debug=True)
