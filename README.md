# MediBot

MediBot is an AI-powered healthcare assistant built using a Retrieval-Augmented Generation (RAG) architecture. It combines semantic search and large language model generation to deliver medically grounded, conversational answers to user queries.

Designed to be approachable and trustworthy, MediBot helps users:

Navigate symptoms
Understand medications
Receive daily wellness tips
While MediBot is not a substitute for professional medical advice, it serves as a fast, reliable first layer of support for general health information.

Features

Chat interface built with Streamlit for easy interaction
Claude 3 via AWS Bedrock as the core LLM
FAISS vector database for fast semantic search over pre-loaded medical content
RAG architecture for combining retrieved knowledge with LLM response generation
Pre-indexed data from vetted sources like medical encyclopedias and symptom guides
System Architecture

1. Data Processing (database.py)
>Loads medical PDFs
>Splits text into overlapping chunks
>Generates sentence-transformer embeddings
>Saves them into a FAISS vector store

3. Backend Logic (llm_data_connect.py)
>Loads vector database
>Retrieves top relevant text chunks based on user query
>Structures a prompt with context using LangChain
>Sends the prompt to Claude via AWS Bedrock for response generation

4. Frontend Interface (medibot.py)
>Built in Streamlit
>Provides a chat UI with support for topic shortcuts and live history
>Handles conversation flow, prompt submission, and response display

