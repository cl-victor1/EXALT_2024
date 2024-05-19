from openai import OpenAI

class OpenAIProxy:
    def __init__(self):
        self.openai_client = OpenAI(max_retries=10)
        self.system_prompt = None
    
    def set_system_promt(self, system_prompt):
        self.system_prompt = system_prompt
    
    def call_chat_completion_api(self, role_content, model_name='gpt-4-turbo-2024-04-09', max_tokens=None):
        if self.system_prompt is None:
            messages = []
        else:
            messages = [{"role": "system", "content": self.system_prompt}]
        for (role, content) in role_content:
            messages.append({"role": role, "content": content})
        response = self.openai_client.chat.completions.create(model=model_name, messages=messages, max_tokens=max_tokens)
        # only return the content, discard everything else
        return response.choices[0].message.content
    
    def call_embeddings_api(self, text, dimensions, model_name='text-embedding-3-small'):
        # only return the embedding, discard everything else
        return self.openai_client.embeddings.create(input = [text], model=model_name, dimensions=dimensions).data[0].embedding
