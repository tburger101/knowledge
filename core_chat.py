import time

import openai
import embed


def combo_query_string(client_query, background_strings, gpt_model, token_budget):
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""

    introduction = "The context section is provided to " \
                   "give you more context and might not be needed to answer the question. "
    question = f"\n\nQuestion: {client_query}"
    message = introduction
    for string in background_strings:
        next_article = f'\n\nContext Section:\n"""\n{string}\n"""'
        if (
                embed.num_tokens(message + next_article + question, model=gpt_model)
                > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def ask_chat(key, client_query, content_message, temperature, model):
    openai.api_key = key

    messages = [
        {"role": "system", "content": content_message},
        {"role": "user", "content": client_query},
    ]

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        response_message = response["choices"][0]["message"]["content"]

    except openai.error.APIError as e:
        time.sleep(.5)
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        response_message = response["choices"][0]["message"]["content"]

    return response_message
