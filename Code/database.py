from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DATA_PATH="data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    
    documents=loader.load()
    return documents
documents=load_pdf_files(data=DATA_PATH)
#print("Length of PDF pages: ", len(documents))
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
print(f"Total chunks created: {len(text_chunks)}")
print(" Sample chunk preview:")
print(text_chunks[0].page_content[:300])  # Show first 300 characters of first chunk
def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model=get_embedding_model()
sample_embedding = embedding_model.embed_documents([text_chunks[0].page_content])
print(f"Sample embedding length: {len(sample_embedding[0])}")


#DB_FAISS_PATH = "vectorstore/db_faiss"
#db = FAISS.from_documents(text_chunks, embedding_model)
# db.save_local(DB_FAISS_PATH)
