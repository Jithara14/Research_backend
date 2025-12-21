from llama_cpp import Llama
from config import Settings


settings = Settings()
class LLMService:
    def __init__(self):
        self.model = Llama(
            model_path="model/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=0,
        )

    def generate_response(self,query:str , search_results:list[dict]):

        context_text = "\n\n".join(
            f"Source {i+1}({result['url']}):\n{result['content']}"
            for i,result in enumerate(search_results)
        )

        full_prompt = f"""
        You are a helpful legal assistant.

        Context from web search:
        {context_text}

        User query:
        {query}

        Please provide a comprehensive, detailed, well-cited, accurate response using the above context. 
        Think step by step and make sure you fully answer the user’s question. 
        Do not use outside knowledge unless it is clearly necessary.
        """


        output = self.model(
            full_prompt,
            max_tokens=512,
            temperature=0.3,
            top_p=0.9,
            stop=["</s>"],
        )

        response_text = output["choices"][0]["text"].strip()
        return response_text
