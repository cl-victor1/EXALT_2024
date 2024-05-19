import anthropic

class AnthropicProxy:
    def __init__(self):
        self.anthropic_client = anthropic.Anthropic(max_retries=10)
        self.system_prompt = None
    
    def set_system_promt(self, system_prompt):
        self.system_prompt = system_prompt
    
    def call_message_api(self, role_content, model_name='claude-3-opus-20240229', max_tokens=1024):
        messages = []
        for (role, content) in role_content:
            messages.append({"role": role, "content": content})
        response = self.anthropic_client.messages.create(model=model_name, system=self.system_prompt, messages=messages, max_tokens=max_tokens)
        # only return the content, discard everything else
        return response.content[0].text
