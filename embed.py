import openai
import pandas as pd
import PyPDF2
import numpy as np
import tiktoken
import re
from scipy import spatial


def embeddings_from_string(embedding_string):
    # Remove brackets and split into list of values
    embedding_string = embedding_string.strip('[]')
    embedding_list = embedding_string.split(', ')

    # Convert values to floats and create numpy array
    embedding_array = np.array([float(val) for val in embedding_list])

    return embedding_array


def num_tokens(text, model):
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def halved_by_delimiter(string, delimiter, model):
    """Split a string in two, on a delimiter, trying to balance tokens on each side."""
    chunks = string.split(delimiter)
    if len(chunks) == 1:
        return [string, ""]  # no delimiter found
    elif len(chunks) == 2:
        return chunks  # no need to search for halfway point
    else:
        total_tokens = num_tokens(string, model)
        halfway = total_tokens // 2
        best_diff = halfway
        for i, chunk in enumerate(chunks):
            left = delimiter.join(chunks[: i + 1])
            left_tokens = num_tokens(left, model)
            diff = abs(halfway - left_tokens)
            if diff >= best_diff:
                break
            else:
                best_diff = diff
        left = delimiter.join(chunks[:i])
        right = delimiter.join(chunks[i:])
        return [left, right]


def truncated_string(string, model, max_tokens):
    """Truncate a string to a maximum number of tokens."""
    encoding = tiktoken.encoding_for_model(model)
    encoded_string = encoding.encode(string)
    truncated_string_text = encoding.decode(encoded_string[:max_tokens])
    if len(encoded_string) > max_tokens:
        print(f"Warning: Truncated string from {len(encoded_string)} tokens to {max_tokens} tokens.")
    return truncated_string_text


class PDFEmbeddings:
    def __init__(self, openai_key, max_tokens=None, pdf_path=None):
        self.pdf_path = pdf_path
        self.openai_key = openai_key
        self.max_tokens = max_tokens
        self.model = "text-embedding-ada-002"
        self.text_chunks = None
        self.embeddings_df = None

    def create_text_chunks(self):
        # Extract text from the PDF
        with open(self.pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ''
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()

        # Split text into chunks based on max_tokens
        # Remove extra white spaces and newlines
        text = re.sub(r'\s+', ' ', text).strip()

        # Split the text into chunks based on headings and new paragraphs
        chunks = self.split_text(text, max_recursion=8)
        self.text_chunks=chunks
        return chunks

    def split_text(self, string, max_recursion):
        """
        Split a subsection into a list of subsections, each with no more than max_tokens.
        Each subsection is a tuple of parent titles [H1, H2, ...] and text (str).
        """

        num_tokens_in_string = num_tokens(string, self.model)
        # if length is fine, return string
        if num_tokens_in_string <= self.max_tokens:
            return [string]

        # if recursion hasn't found a split after X iterations, just truncate
        elif max_recursion == 0:
            return [truncated_string(string, model=self.model, max_tokens=self.max_tokens)]
        # otherwise, split in half and recurse
        else:
            for delimiter in ["\n\n", "\n", ". "]:
                left, right = halved_by_delimiter(string, delimiter=delimiter, model=self.model)
                if left == "" or right == "":
                    # if either half is empty, retry with a more fine-grained delimiter
                    continue
                else:
                    # recurse on each half
                    results = []
                    for half in [left, right]:
                        half_strings = self.split_text(half, max_recursion - 1)
                        results.extend(half_strings)
                    return results

        # otherwise no split was found, so just truncate (should be very rare)
        return [truncated_string(string, self.model, self.max_tokens)]

    def create_embeddings_df(self):
        # Generate OpenAI embeddings for each text chunk
        embeddings = []
        openai.api_key = self.openai_key
        for chunk in self.text_chunks:
            response = openai.Embedding.create(
                model=self.model,
                input=chunk,
            )
            embedding = response['data'][0]['embedding']
            embeddings.append([chunk, embedding])

        # Create dataframe with embeddings
        df = pd.DataFrame(embeddings, columns=['raw_text', 'embedding'])

        return df

    def chunk_embed(self):
        self.create_text_chunks()
        self.create_embeddings_df()

    def top_embeddings(self, string, top_results):
        """Returns a list of strings and relatednesses, sorted from most related to least."""
        openai.api_key = self.openai_key
        relatedness_fn = lambda x, y: 1 - spatial.distance.cosine(x, y)
        self.embeddings_df['embedding']=self.embeddings_df['embedding'].apply(embeddings_from_string)

        query_embedding_response = openai.Embedding.create(
            model=self.model,
            input=string,
        )
        query_embedding = query_embedding_response["data"][0]["embedding"]

        strings_and_relatednesses = [
            (row["raw_text"], relatedness_fn(query_embedding, row["embedding"]))
            for i, row in self.embeddings_df.iterrows()
        ]
        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        strings, relatednesses = zip(*strings_and_relatednesses)
        return strings[:top_results], relatednesses[:top_results]
