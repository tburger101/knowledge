# Knowledge Chat- Store, Chat, and Summarize Personal Documents 

This Flask App allows you to upload your personal PDFs, chat with your documents, and obtain a summary of them.

## Prerequisites
* langchain
* pinecone
* langchai
* Pinecone API Key
* OpenAI API key

## Installation
1. Clone the repository to your local machine:
2. Install the required Python packages using pip: pip install -r requirements.txt
3. Set up your OpenAI API key:
   1. Visit the OpenAI website and create an account.
   2. Obtain an API key.
   3. Set the environment variable OPENAI_API_KEY to your API key.
4. Create a new pinecone account/index/ and API key
   1. Visit the pincone website and create an account.
   2. Obtain an API key.
   3. Create a new index called "my-knowledgebase" with 1536 dimensions.
   4. Set the environment variable PINECONE_API_KEY to your API Key.
   5. Alter the application.py pinecone.init to reflect your pinecone environment. 

## Getting Started
1. Run the Flask application: application.py.run()
2. Access the application in your web browser at http://localhost:5000.
3. Upload your first PDF by navigating to the upload tab and adding your document to your vault.
   1. The document will now be converted to embedding vectors in your pinecone index.
   2. The PDF will also be saved in the knowledge folder to be able to access it later.
4. After your documents have been uploaded you know have the option to ask them questions or summarize them.
   1. To ask questions: Navigate to the home page Knowledge Chat and start typing
   2. To summarize documents: Navigate to the Summarize chat and select your document with the prompt of how you are looking to summarize it.

## License
This project is licensed under the MIT License.

