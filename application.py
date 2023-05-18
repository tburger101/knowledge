import pandas as pd
from flask import Flask, render_template, request
import os
from uuid import uuid4
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain import VectorDBQA, OpenAI
import pinecone
import embed
import core_chat as cc

app = Flask(__name__)
chat_history = []

pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY'),
    environment="us-central1-gcp",
)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        chat_history.append(('User', user_input))
        output = process_input(user_input)  # Call your function to process the input text
        chat_history.append(('Chatbot', output))
        return render_template('index.html', chat_history=chat_history)
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        pdf_file = request.files['file']
        file_name = pdf_file.filename

        upload_folder = os.path.join(app.root_path, 'knowledge')
        os.makedirs(upload_folder, exist_ok=True)

        # Generate a unique filename
        pdf_file_name = os.path.join(upload_folder, file_name)
        pdf_file.save(pdf_file_name)

        # loader = PyPDFLoader(pdf_file_name)
        # document = loader.load()
        #
        # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        # texts = text_splitter.split_documents(document)
        #
        # embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
        # Pinecone.from_documents(texts, embeddings, index_name="my-knowledgebase")

        emd = embed.PDFEmbeddings(pdf_path=pdf_file_name, openai_key=os.environ.get("OPENAI_API_KEY"),
                                  max_tokens=1600)
        emd.create_text_chunks()

        embedding_df = emd.create_embeddings_df()
        previous_embedding_df = pd.read_csv("static/embeddings.csv")

        combo_embedding_df = pd.concat([embedding_df, previous_embedding_df])
        combo_embedding_df.drop_duplicates(keep='first', inplace=True)
        combo_embedding_df.to_csv("static/embeddings.csv", index=False, escapechar='\\')

        return render_template('upload.html')
    else:
        return render_template('upload.html')


@app.route('/summary', methods=['GET', 'POST'])
def summary():
    # Path to the knowledge folder
    knowledge_folder = 'knowledge'

    # List to store file names
    files = []

    # Loop through the files in the knowledge folder

    if request.method == 'POST':
        print('test')
    else:
        for filename in os.listdir(knowledge_folder):
            # Check if the item is a file
            if os.path.isfile(os.path.join(knowledge_folder, filename)):
                # Add the file name to the list
                files.append(filename)
        return render_template('summary.html', files=files)


def process_input(user_input):
    key = os.environ.get('OPENAI_API_KEY')
    emd = embed.PDFEmbeddings(pdf_path="", openai_key=key,
                              max_tokens=1600)
    emd.embeddings_df = pd.read_csv("static/embeddings.csv")
    model = 'gpt-3.5-turbo'

    content_message = "Fullfil the users request or answer the question. The context section is provided to give you " \
                      "more context and might not be needed to answer the question. "
    top_results_strings, scores = emd.top_embeddings(user_input, 5)
    gpt_message = cc.combo_query_string(user_input, top_results_strings, model, 1700)
    answer = cc.ask_chat(key, gpt_message, content_message, .7, model)

    # embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    # docsearch = Pinecone.from_existing_index('my-knowledgebase', embeddings)
    # qa = VectorDBQA.from_chain_type(
    #     llm=OpenAI(temperature=.7), chain_type="stuff", vectorstore=docsearch, return_source_documents=True, k=3
    # )
    # answer = qa({"query": user_input}).get('result')
    #
    return f"Answer: {answer}"


if __name__ == '__main__':
    app.run(debug=True)
