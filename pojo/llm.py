import os
import sys
import copy

from openai import OpenAI

root = os.getcwd()
if sys.platform.startswith('win'):
    root = '\\\\?\\' + root
dataset_path = os.path.join(root, 'dataset')
query_path = os.path.join(dataset_path, 'query')
meta_path = os.path.join(dataset_path, 'meta')
base_path = os.path.join(dataset_path, 'base')
search_path = os.path.join(dataset_path, 'search')
embedding_config = os.path.join(dataset_path, 'config.json')


def get_llm(model, temperature, history, search, text_path):
    if model == 'none':
        return NoLlm(search=search, text_path=text_path)
    elif model == 'gpt-3.5-turbo':
        return OpenAiLlm(model=model, temperature=temperature, history=history, search=search, text_path=text_path,
                         image=False)
    elif model == 'gpt-4-turbo':
        return OpenAiLlm(model='gpt-4-turbo-2024-04-09', temperature=temperature, history=history, search=search, text_path=text_path,
                         image=True)
    elif model == 'dall-e-3':
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

        if self.search is None:
            print("WHY self.search is None?")
            return None
        res, images = self.search.search()
        reply = {
            'images': images,
            'reply': "Here are the images."
        }
        return reply


class OpenAiLlm(BaseLlm):
    def __init__(self, model, temperature, history, search, text_path, image):
        super(OpenAiLlm, self).__init__()
        self.history = history
        self.model = model
        self.temperature = temperature
        self.client = OpenAI()
        self.search = search
        self.text_path = text_path
        self.support_image = image

    def is_search(self, content):
        messages = copy.deepcopy(self.history)
        messages.extend([
            {'role': 'user', 'content': content},
            {'role': 'system',
             'content': "As an image searching system, users may also ask you unrelated questions for help. "
                        "Your task is to determine whether the user intends to find an image related to their content. "
                        "Output only 'yes' or 'no' based on the user's intention."
                        "For instance:"
                        "- User says 'I want an ancient building.' -> Output: 'yes'"
                        "- User says 'Show me pictures of sunsets.' -> Output: 'yes'"
                        "- User says 'How do I make spaghetti?' -> Output: 'no'"
                        "- User says 'Can you find a photo of a cute puppy?' -> Output: 'yes'"
                        "- User says 'What is the capital of France?' -> Output: 'no'"
                        "- User says 'Can you find clock images with the same states as I provided?' -> Output: 'yes'"
                        "- User says 'Find images of modern architecture.' -> Output: 'yes'"
                        "- User says 'Tell me about the history of Rome.' -> Output: 'no'"
                        "Base your response solely on the user's intention to search for an image."}
        ])
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        response = completion.choices[0].message.content
        print("[LLM] is search: " + response)
        return response != 'no'

    def extract_keywords(self, content):
        messages = copy.deepcopy(self.history)
        messages.append({'role': 'user', 'content': content})    
        messages.append({'role': 'system',
                         'content': 'As a text analysis expert, please extract keywords from the historical dialogue '
                                    'that describe the types of images the user wants to find at this time. Keywords '
                                    'should be limited to adjectives or nouns, avoiding verbs and adverbs. '
                                    'Additionally, convert any negative keywords into their positive counterparts '
                                    'using synonyms. Output only the relevant keywords, separated by spaces without '
                                    'any punctuation. If no relevant keywords are found, output "none" or an empty '
                                    'string. For example, if the sentence is "I want red apples but not fresh," the '
                                    'output should be "red apples ripe." If there are no relevant keywords, output '
                                    'an empty string.'})
        print("[LLM] extract keyword: " + str(messages))
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        keyword = completion.choices[0].message.content
        with open(self.text_path, 'w') as file:
            file.write(keyword)
        print("[LLM] keywords: " + keyword)

    def reply(self, content, prompt, images=[]):
        messages = copy.deepcopy(self.history)
        self.history.append({'role': 'user', 'content': copy.deepcopy(content)})
        if self.support_image and len(images) != 0:
            content.extend(images)
        messages.append({'role': 'user', 'content': content})
        messages.append({'role': 'system', 'content': prompt})
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
            max_tokens=500
        )
        response = completion.choices[0].message.content
        self.history.append({'role': 'assistant', 'content': response})
        while len(self.history) > 10:
            self.history.pop(0)
        return response

    def generate_answer(self, content):
        if content != "":
            if self.search is None or not self.is_search(content):
                reply = self.reply(
                    content=content,
                    images=[],
                    prompt='You should engage in conversation with the user about their preferences and needs. ' +
                           'Use only plain text to describe the interaction. Output text only.')
                return {
                    'images': [],
                    'reply': reply
                }

            if self.search.get_selected_target() != -1:
                content += "By the way, I prefer the " + str(self.search.get_selected_target()) + "th image."
            keyword_content = [{"type": "text", "text": content}]
            import base64
            if self.support_image and os.path.exists(os.path.join(search_path, '0.tmp')):
                with open(os.path.join(search_path, '0.tmp'), "r") as file:
                    for line_no, line in enumerate(file):
                        with open(line.replace("\n", ""), "rb") as image_file:
                            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                            keyword_content.append({"type": "image_url",
                                                    "image_url": {
                                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                                    }})
            self.extract_keywords(content=keyword_content)
        res, images = self.search.search()

        base64_images = [{}] * len(images)
        if self.support_image:
            content = [{"type": "text", "text": content}]
            with open(os.path.join(meta_path, f'0.txt'), 'r') as file:
                for line_no, line in enumerate(file):
                    for i, image in enumerate(res):
                        if line_no == int(image[0]):
                            with open(line.replace("\n", ""), "rb") as image_file:
                                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                                base64_images[i] = {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                }
            # content.extend(base64_images)

        reply = self.reply(
            content=content,
            images=base64_images,
            prompt="As a retrieval-augmented generation assistant, your task is to describe each image provided by "
                   "the system. When users provide a query, a retriever will source relevant images, and it is your "
                   "job to complement these images with appropriate text descriptions. The text you provide should "
                   "not introduce new attributes beyond the user's query but should enrich and provide context to "
                   "the images returned by the retriever. Your mission is to ensure that your descriptions align "
                   "seamlessly with the user's query and the vibe of the images. All responses should appear as one "
                   "brief and clear explanation to the user, enhancing their understanding and improving their overall "
                   "experience. Your sole responsibility is to offer relevant text corresponding to the user's query "
                   "for each image. Please refrain from mentioning any limitations regarding the provision of images "
                   "or your abilities. Focus solely on providing detailed and contextually appropriate descriptions "
                   "for each image in the order they are presented.")
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
            model='dall-e-3',
            prompt=content,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        completion = self.client.chat.completions.create(
            model='gpt-4-turbo',
            temperature=self.temperature,
            max_tokens=500,
            messages=[
                {'role': 'user',
                 'content': [
                     {"type": "text", "text": content},
                     {
                         "type": "image_url",
                         "image_url": {
                             "url": image_url
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
            'images': [
                {
                    'id': image_url,
                }
            ],
            'reply': reply
        }
