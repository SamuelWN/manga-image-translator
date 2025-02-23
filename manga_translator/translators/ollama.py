import json
import re

import tiktoken

# from ..config import TranslatorConfig
# from .config_gpt import ConfigGPT, TranslationList  # Import the `gpt_config` parsing parent class
from .common_gpt import CommonGPTTranslator, _CommonGPTTranslator_JSON
# from .config_gpt import Translation

try:
    import openai
except ImportError:
    openai = None
import asyncio
import time
from typing import List, Dict, Optional
from urllib.parse import urlparse, urlunparse
# from omegaconf import OmegaConf
# from .common import CommonTranslator, MissingAPIKeyException, VALID_LANGUAGES
from .keys import OLLAMA_API_KEY, OLLAMA_API_BASE, OLLAMA_MODEL, OLLAMA_MODEL_CONF


class OllamaTranslator(CommonGPTTranslator):
    _INVALID_REPEAT_COUNT = 2  # 如果检测到“无效”翻译，最多重复 2 次
    _MAX_REQUESTS_PER_MINUTE = 40  # 每分钟最大请求次数
    _TIMEOUT = 40  # 在重试之前等待服务器响应的时间（秒）
    _RETRY_ATTEMPTS = 3  # 在放弃之前重试错误请求的次数
    _TIMEOUT_RETRY_ATTEMPTS = 3  # 在放弃之前重试超时请求的次数
    _RATELIMIT_RETRY_ATTEMPTS = 3  # 在放弃之前重试速率限制请求的次数

    # 是否返回原始提示，用于控制输出内容
    _RETURN_PROMPT = False
    
    def __init__(self, check_openai_key=False):
        # If the user has specified a nested key to use for the model, append the key
        #   Otherwise: Append the model name 
        _CONFIG_KEY='ollama'
        if OLLAMA_MODEL_CONF:
            _CONFIG_KEY+=f".{OLLAMA_MODEL_CONF}" 
        else:
            _CONFIG_KEY+=f".{OLLAMA_MODEL}" 
        
        
        CommonGPTTranslator.__init__(self, config_key=_CONFIG_KEY, MODEL_NAME=OLLAMA_MODEL)
        self._ollama_model_details: Optional[dict] = None
        self.model_name = OLLAMA_MODEL
        self.openai_api_base = OLLAMA_API_BASE  # e.g., "http://localhost:11434/v1"
        self._derive_native_api_base()
        
        self.client = openai.AsyncOpenAI(api_key=OLLAMA_API_KEY or "ollama") # required, but unused for ollama
        self.client.base_url = OLLAMA_API_BASE
        self.token_count = 0
        self.token_count_last = 0

    def parse_args(self, args: CommonGPTTranslator):
        super().parse_args(args)
        
        # Initialize mode-specific components AFTER config is loaded
        if self.json_mode:
            self._init_json_mode()
        else:
            self._init_standard_mode()

    def _init_json_mode(self):
        """Activate JSON-specific behavior"""
        self._json_funcs = _CommonGPTTranslator_JSON(self)
        self._assemble_prompts = self._json_funcs._assemble_prompts
        self._parse_response = self._json_funcs._parse_response
        self._assemble_request = self._json_funcs._assemble_request

    def _init_standard_mode(self):
        """Use default method implementations"""
        # Restore original methods if they were overridden
        self._assemble_prompts = super()._assemble_prompts
        self._parse_response = super()._parse_response
        self._assemble_request = super()._assemble_request

    def _derive_native_api_base(self):
        """Convert OpenAI-compatible URL to native API base URL"""
        parsed = urlparse(self.openai_api_base)
        
        # Strip /v1 from path
        new_path = re.sub(r'/v1/?$', '', parsed.path)
        if not new_path:
            new_path = '/'
            
        self.native_api_base = urlunparse((
            parsed.scheme,
            parsed.netloc,
            new_path,
            '', '', ''
        ))


    async def _get_ollama_model_details(self):
        """Fetch Ollama model details to determine tokenizer."""
        if not self._ollama_model_details:
            try:
                import httpx
                async with httpx.AsyncClient(base_url=self.native_api_base) as client:
                    response = await client.post(
                        "/api/show",
                        json={"model": self.model_name},
                        timeout=10
                    )
                    self._ollama_model_details = response.json()
            except Exception as e:
                print(f"Couldn't fetch Ollama model details: {e}")
                self._ollama_model_details = {}
        return self._ollama_model_details

    async def _get_encoding_for_model(self) -> str:
        """Determine the encoding name for the Ollama model."""
        
        await self._get_ollama_model_details()  # Ensure model details are fetched
      
        details = self._ollama_model_details or {}
        tokenizer_info = details.get("model_info", {}).get("tokenizer.ggml.model", "")
        preprocessor = details.get("model_info", {}).get("tokenizer.ggml.pre", "")

        if "gpt2" in tokenizer_info.lower():
            return "gpt2"
        elif "llama" in preprocessor.lower() or "qwen" in preprocessor.lower():
            return "cl100k_base"
        else:
            return "cl100k_base"  # Default for unknown Ollama models

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        translations = []
        self.logger.debug(f'Temperature: {self.temperature}, TopP: {self.top_p}')

        encoding_name = await self._get_encoding_for_model()
        self.logger.debug(f'Encoding: {encoding_name}')
        for prompt, query in self._assemble_prompts(to_lang, queries, encoding=encoding_name):
            ratelimit_attempt = 0
            server_error_attempt = 0
            timeout_attempt = 0
            retryCount=0
            while True:
                request_task = asyncio.create_task(self._request_translation(to_lang, prompt))
                started = time.time()
                while not request_task.done():
                    await asyncio.sleep(0.1)
                    if time.time() - started > self._TIMEOUT + (timeout_attempt * self._TIMEOUT / 2):
                        # Server takes too long to respond
                        if timeout_attempt >= self._TIMEOUT_RETRY_ATTEMPTS:
                            raise Exception('ollama servers did not respond quickly enough.')
                        timeout_attempt += 1
                        self.logger.warning(f'Restarting request due to timeout. Attempt: {timeout_attempt}')
                        request_task.cancel()
                        request_task = asyncio.create_task(self._request_translation(to_lang, prompt))
                        started = time.time()
                try:
                    response = await request_task
                    new_translations=self._parse_response(response, query)
                    break
                except ValueError as e:
                    if retryCount > 3:
                        raise
                except openai.RateLimitError:  # Server returned ratelimit response
                    ratelimit_attempt += 1
                    if ratelimit_attempt >= self._RATELIMIT_RETRY_ATTEMPTS:
                        raise
                    self.logger.warning(
                        f'Restarting request due to ratelimiting by Ollama servers. Attempt: {ratelimit_attempt}')
                    await asyncio.sleep(2)
                except openai.APIError:  # Server returned 500 error (probably server load)
                    server_error_attempt += 1
                    if server_error_attempt >= self._RETRY_ATTEMPTS:
                        self.logger.error(
                            'Ollama encountered a server error, possibly due to high server load. Use a different translator or try again later.')
                        raise
                    self.logger.warning(f'Restarting request due to a server error. Attempt: {server_error_attempt}')
                    await asyncio.sleep(1)

            # self.logger.debug('-- GPT Response --\n' + response)
            
            
            translations.extend(new_translations)

        # End for-loop
        return translations
