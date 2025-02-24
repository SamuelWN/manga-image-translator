import re
import asyncio
import time
import tiktoken

from typing import List
from .common import MissingAPIKeyException
from .keys import OPENAI_API_KEY, OPENAI_HTTP_PROXY, OPENAI_API_BASE, OPENAI_MODEL
from .common_gpt import CommonGPTTranslator, _CommonGPTTranslator_JSON

try:
    import openai
except ImportError:
    openai = None

class OpenAITranslator(CommonGPTTranslator):
    _MAX_REQUESTS_PER_MINUTE = 200
    _TIMEOUT = 40
    _RETRY_ATTEMPTS = 3

    def __init__(self, check_openai_key=True):
        _CONFIG_KEY = 'chatgpt.' + OPENAI_MODEL
        CommonGPTTranslator.__init__(self, config_key=_CONFIG_KEY, MODEL_NAME=OPENAI_MODEL)

        if not OPENAI_API_KEY and check_openai_key:
            raise MissingAPIKeyException('OPENAI_API_KEY environment variable required')

        client_args = {
            "api_key": OPENAI_API_KEY,
            "base_url": OPENAI_API_BASE
        }
        
        if OPENAI_HTTP_PROXY:
            from httpx import AsyncClient
            client_args["http_client"] = AsyncClient(proxies={
                "all://*openai.com": f"http://{OPENAI_HTTP_PROXY}"
            })

        self.client = openai.AsyncOpenAI(**client_args)
        self._MAX_TOKENS = 8192
        self.token_count = 0
        self.token_count_last = 0
        self._last_request_ts = 0

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

    def _get_encoding_for_model(self) -> str:
        """Determine the encoding name for the OpenAI model."""
        self.logger.debug("OPENAI_MODEL: ")
        self.logger.debug(OPENAI_MODEL)

        try:
            # Use tiktoken's built-in mapping for OpenAI models
            encoding = tiktoken.encoding_for_model(OPENAI_MODEL)
            return encoding.name
        except KeyError:
            return "cl100k_base"  # Fallback for unknown OpenAI models

    def _count_tokens(self, text: str) -> int:
        """Count tokens for OpenAI models."""
        encoding_name = self._get_encoding_for_model()
        self.logger.debug("encoding_name: ")
        self.logger.debug(encoding_name)
        return self.count_tokens(text, encoding_name)

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        translations = []
        
        for prompt, query in self._assemble_prompts(to_lang, queries, encoding=self._get_encoding_for_model()):
            for attempt in range(self._RETRY_ATTEMPTS):
                try:
                    response_text = await self._request_translation(to_lang, prompt)
                    translations.extend(self._parse_response(response_text, query))
                    # return translations
                except Exception as e:
                    self.logger.warning(f"Translation attempt {attempt+1} failed: {str(e)}")
                    if attempt == self._RETRY_ATTEMPTS - 1:
                        raise
                    await asyncio.sleep(1)
        
        return translations

    def _parse_response(self, response: str, queries: List[str]) -> List[str]:
        # Initialize output list as a copy of the input
        #   Any skipped/omitted values will be filtered out as:
        #       `Translation identical to queries`
        translations = queries.copy()

        # Testing suggests ChatGPT refusals are all-or-nothing. 
        #   If partial responses do occur, this should may benefit from revising.
        if self._cannot_assist(response):
            self.logger.error(f'Refusal message detected in response. Skipping.')  
            return translations


        expected_count=len(translations)

        # Use translation ID to position value in list `translations`
        #   Parse output to grab translation ID
        #   Use translation ID to position in a list

        # Use regex to extract response 
        response=self.extract_capture_groups(response, rf"{self.rgx_capture}")

        # Extract IDs and translations from the response
        translation_matches = list(re.finditer(r'<\|(\d+)\|>(.*?)(?=(<\|\d+\|>|$))', 
                                    response, re.DOTALL)
                                )

        # Insert translations into their respective positions based on IDs:
        for match in translation_matches:
            id_num = int(match.group(1))
            translation = match.group(2).strip()
            
            # Ensure the ID is within the expected range
            if id_num < 1 or id_num > expected_count:
                raise ValueError(f"ID {id_num} in response is out of range (expected 1 to {expected_count})")
            
            # Insert the translation at the correct position
            translations[id_num - 1] = translation
        
        return translations
        
    async def _ratelimit_sleep(self):
        if self._MAX_REQUESTS_PER_MINUTE > 0:
            now = time.time()
            delay = 60 / self._MAX_REQUESTS_PER_MINUTE
            if now - self._last_request_ts < delay:
                await asyncio.sleep(delay - (now - self._last_request_ts))
            self._last_request_ts = time.time()