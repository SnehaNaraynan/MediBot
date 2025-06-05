import os
import json
import boto3
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM
from langchain_core.language_models.llms import LLM
from typing import Optional

# --- Claude via Bedrock Setup ---
def get_bedrock_client(region: str = 'us-west-2', runtime: bool = True):
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        raise Exception("AWS credentials not found in environment.")
    return boto3.client(
        service_name="bedrock-runtime" if runtime else "bedrock",
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

bedrock = get_bedrock_client()

# --- Custom LLM Wrapper ---
class ClaudeLLM(LLM):
    model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    region_name: str = "us-west-2"
    temperature: float = 0.5
    max_tokens: int = 1000

    def _call(self, prompt: str, **kwargs) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

        import boto3
        client = boto3.client("bedrock-runtime", region_name=self.region_name)
        response = client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )

        result = json.loads(response["body"].read().decode())
        return result["content"][0]["text"].strip()

    def _llm_type(self) -> str:
        return "bedrock-claude"

# --- Prompt ---
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know — don't make it up.
Only use information from the provided context.

Context: {context}
Question: {question}

Start the answer directly. Be concise.
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# --- FAISS Vectorstore ---
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# --- Create RetrievalQA Chain ---
llm = ClaudeLLM()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# --- CLI Prompt Loop ---
user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})
print("\nRESULT:\n", response["result"])