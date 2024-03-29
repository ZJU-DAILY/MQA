import os
import sys
import time

from openai import OpenAI

root = os.getcwd()
if sys.platform.startswith('win'):
    root = '\\\\?\\' + root
dataset_path = os.path.join(root, 'dataset')
query_path = os.path.join(dataset_path, 'query')
meta_path = os.path.join(dataset_path, 'meta')
base_path = os.path.join(dataset_path, 'base')
embedding_config = os.path.join(dataset_path, 'config.json')


def get_llm(model, temperature, history, search, text_path):
    if model == 'none':
        return NoLlm(search=search, text_path=text_path)
    elif model == 'gpt-3.5-turbo':
        return OpenAiLlm(model=model, temperature=temperature, history=history, search=search, text_path=text_path)
    elif model == 'gpt-4(dall-e-2)':
        return DALLE(model=model, temperature=temperature)


class BaseLlm:
    def __init__(self):
        pass

    def generate_answer(self, content):
        raise NotImplementedError


class NoLlm(BaseLlm):
    def __init__(self, search, text_path):
        super().__init__()
        self.search = search
        self.text_path = text_path

    def generate_answer(self, content):
        with open(os.path.join(root, 'dataset', 'search', self.text_path), 'w') as file:
            file.write(content)

        images = self.search.search()
        reply = {
            'images': images,
            'reply': "Here are the images."
        }
        return reply


class OpenAiLlm(BaseLlm):
    def __init__(self, model, temperature, history, search, text_path):
        super(OpenAiLlm, self).__init__()
        self.history = history
        self.model = 'gpt-3.5-turbo'
        self.temperature = temperature
        self.client = OpenAI()
        self.search = search
        self.text_path = text_path

    def is_search(self, content):
        messages = self.history.copy()
        messages.extend([
            {'role': 'user', 'content': content},
            {'role': 'system',
             'content': "As an image searching system, users may also ask you unrelated questions for help." +
                        "Determine whether the user intends to find an image related to their content. " +
                        "Output only 'yes' or 'no'. For example, if the user says 'I want an ancient building.', " +
                        "it is likely that the user wants an image of an ancient building, so output 'yes'."}
        ])
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        response = completion.choices[0].message.content
        print(response)
        return response != 'no'

    def extract_keywords(self, content):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {'role': 'user', 'content': content},
                {'role': 'system',
                 'content': 'As a text analysis expert, please extract the keywords from the following sentences. ' +
                            'Keywords should be limited to adjectives or nouns, avoiding verbs and adverbs. ' +
                            'Additionally, negative keywords should be converted into positive ones using synonyms. ' +
                            'Output only the possessed keywords, separated by whitespace without any punctuate. ' +
                            'For example, if the sentence is "I want red apples but not fresh", ' +
                            'the output should be "red apples ripe"'}]
        )
        keyword = completion.choices[0].message.content
        with open(self.text_path, 'w') as file:
            file.write(keyword)

    def reply(self, content, prompt):
        self.history.append({'role': 'user', 'content': content})
        messages = self.history.copy()
        messages.append({'role': 'system', 'content': prompt})
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages
        )
        response = completion.choices[0].message.content
        self.history.append({'role': 'assistant', 'content': response})
        return response

    def generate_answer(self, content):
        if not self.is_search(content) or self.search is None:
            reply = self.reply(
                content=content,
                prompt='You should engage in conversation with the user about their preferences and needs. ' +
                       'Use only plain text to describe the interaction. Output text only.')
            return {
                'images': [],
                'reply': reply
            }

        images = self.search.search()
        self.extract_keywords(content=content)
        reply = self.reply(
            content=content,
            prompt='I want you to act as a retrieval-augmented generation assistant. ' +
                   'When users provide a query, there will a retriever sourcing relevant images, ' +
                   'and it is your job to complement them with appropriate text. ' +
                   'The text you provide shall not add a new attribute to the user\'s query, ' +
                   'but to enrich and provide context to the images returned by the retriever. ' +
                   'Your mission is to make sure that your text aligns seamlessly with the user\'s ' +
                   'query and vibe of the images. ' +
                   'All responses should appear as one brief and clear explanation to the user, ' +
                   'enhance their understanding and improve their overall experience. ' +
                   'Your sole responsibility is to offer relevant text corresponding to the user\'s query. ' +
                   'Please refrain from mentioning any limitations regarding the provision of images.')
        return {
            'images': images,
            'reply': reply
        }


class DALLE(BaseLlm):
    def __init__(self, model, temperature):
        super(DALLE, self).__init__()
        self.model = model
        self.temperature = temperature
        self.client = OpenAI()

    def generate_answer(self, content):
        response = self.client.images.generate(
            model='dall-e-2',
            prompt=content,
            size="1024x1024",
            quality="standard",
            n=2,
        )
        # image = response.data[0].url
        images = [response.data[i].url for i in range(len(response.data))]
        completion = self.client.chat.completions.create(
            model='gpt-4-vision-preview',
            temperature=self.temperature,
            messages=[
                {'role': 'user',
                 'content': [
                     {"type": "text", "text": content},
                     {
                         "type": "image_url",
                         "image_url": {
                             "url": images[0]
                         },
                     },
                 ], },
                {'role': 'system',
                 'content': 'I want you to act as a retrieval-augmented generation assistant. ' +
                            'When users provide a query, there will a retriever sourcing relevant images, ' +
                            'and it is your job to complement them with appropriate text. ' +
                            'The text you provide shall not add a new attribute to the user\'s query, ' +
                            'but to enrich and provide context to the images returned by the retriever. ' +
                            'Your mission is to make sure that your text aligns seamlessly with the user\'s ' +
                            'query and vibe of the images. ' +
                            'All responses should appear as one brief and clear explanation to the user, ' +
                            'enhance their understanding and improve their overall experience. ' +
                            'Your sole responsibility is to offer relevant text corresponding to the user\'s query. ' +
                            'Please refrain from mentioning any limitations regarding the provision of images.'}
            ]
        )
        reply = completion.choices[0].message.content
        return {
            'images': images,
            'reply': reply
        }
