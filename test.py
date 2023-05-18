import pandas as pd
from flask import Flask, render_template, request
import os
from uuid import uuid4
from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain import VectorDBQA, OpenAI
from langchain.prompts import PromptTemplate
import pinecone



if __name__ == "__main__":
    loader = PyPDFLoader("knowledge/Sharpstack-RPpaper.pdf")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    #docs = [t.page_content for t in texts]

    prompt_template = """Write a concise summary of the following:

    {text}

    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(OpenAI(temperature=0), chain_type="map_reduce", return_intermediate_steps=True,
                                 map_prompt=PROMPT, combine_prompt=PROMPT)
    chain({"input_documents": texts}, return_only_outputs=True)

    print('test')
