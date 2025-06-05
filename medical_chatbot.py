import os
import json
import boto3
import streamlit as st

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --------------------------
# Claude LLM via Bedrock
# --------------------------
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

# --------------------------
# Prompt Template
# --------------------------
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know — don't make it up.
Only use information from the provided context.

Context: {context}
Question: {question}

Start the answer directly. Be concise.
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# --------------------------
# Load FAISS vectorstore
# --------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# --------------------------
# Streamlit App
# --------------------------
def main():
    st.title("Ask MediBot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Ask me anything about your documents...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error(" Failed to load the vector store.")
                return

            llm = ClaudeLLM()
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=False,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]

            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f" Error: {str(e)}")

if __name__ == "__main__":
    main()
