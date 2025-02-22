import json
import re

# from ..config import TranslatorConfig
# from .config_gpt import ConfigGPT, TranslationList  # Import the `gpt_config` parsing parent class
from .common_gpt import CommonGPTTranslator, TranslationList
from .config_gpt import Translation

try:
    import openai
except ImportError:
    openai = None
import asyncio
import time
from typing import List, Dict, Tuple, Union
from omegaconf import OmegaConf
from .common import CommonTranslator, MissingAPIKeyException, VALID_LANGUAGES
from .keys import OLLAMA_API_KEY, OLLAMA_API_BASE, OLLAMA_MODEL, OLLAMA_MODEL_CONF


class OllamaTranslator(CommonGPTTranslator):
    _LANGUAGE_CODE_MAP=VALID_LANGUAGES

    _INVALID_REPEAT_COUNT = 2  # 如果检测到“无效”翻译，最多重复 2 次
    _MAX_REQUESTS_PER_MINUTE = 40  # 每分钟最大请求次数
    _TIMEOUT = 40  # 在重试之前等待服务器响应的时间（秒）
    _RETRY_ATTEMPTS = 3  # 在放弃之前重试错误请求的次数
    _TIMEOUT_RETRY_ATTEMPTS = 3  # 在放弃之前重试超时请求的次数
    _RATELIMIT_RETRY_ATTEMPTS = 3  # 在放弃之前重试速率限制请求的次数

    # 最大令牌数量，用于控制处理的文本长度
    _MAX_TOKENS = 4096

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
        
        CommonGPTTranslator.__init__(self, config_key=_CONFIG_KEY)

        self.client = openai.AsyncOpenAI(api_key=OLLAMA_API_KEY or "ollama") # required, but unused for ollama
        self.client.base_url = OLLAMA_API_BASE
        self.token_count = 0
        self.token_count_last = 0

    def parse_args(self, args: CommonTranslator):
        super().parse_args(args)
        
        # Initialize mode-specific components AFTER config is loaded
        if self.json_mode:
            self._init_json_mode()
        else:
            self._init_standard_mode()


    def _init_json_mode(self):
        """Activate JSON-specific behavior"""
        self._json_funcs = _OllamaTranslator_JSON(self)
        self._assemble_prompts = self._json_funcs._assemble_prompts
        self._parse_response = self._json_funcs._parse_response
        self._assemble_request = self._json_funcs._assemble_request

    def _init_standard_mode(self):
        """Use default method implementations"""
        # Restore original methods if they were overridden
        self._assemble_prompts = super()._assemble_prompts
        self._parse_response = super()._parse_response
        self._assemble_request = super()._assemble_request


    def _assemble_prompts(self, to_lang: str, queries: List[str]) -> List[Tuple[str, List[str]]]:
        """
        Assemble prompts while respecting token limits.
        Returns a list of tuples containing (prompt, queries_included) for each chunk.
        """
        # Base case: no queries left
        if not queries:
            return []

        # Initialize prompt and token count
        prompt = ''
        token_count = 0

        # Add template if enabled
        if self.include_template:
            template = self.prompt_template.format(to_lang=to_lang)
            prompt += template
            token_count += self._count_tokens(template)

        # Add queries until token limit is reached
        included_queries = []
        for i, query in enumerate(queries):
            query_section = f'\n<|{i + 1}|>{query}'
            query_tokens = self._count_tokens(query_section)

            # Check if adding this query would exceed the token limit
            if token_count + query_tokens > self._MAX_TOKENS:
                # If there's more than one query, split recursively
                if len(queries) > 1:
                    remaining_prompts = self._assemble_prompts(to_lang, queries[i:])
                    return [(prompt.lstrip(), included_queries)] + remaining_prompts
                else:
                    # If only one query remains and it exceeds the limit, yield it alone
                    return [(prompt.lstrip(), included_queries), (query_section.lstrip(), [query])]

            # Add the query to the prompt
            prompt += query_section
            token_count += query_tokens
            included_queries.append(query)

        # Return the final prompt
        return [(prompt.lstrip(), included_queries)]

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        translations = []
        self.logger.debug(f'Temperature: {self.temperature}, TopP: {self.top_p}')

        for prompt, query in self._assemble_prompts(to_lang, queries):
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
                    new_translations=self._parse_response(response, prompt, query)
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

    def _parse_response(self, response: str, prompt, queries: List[str]) -> List[str]:
        translations = queries.copy()

        # Use regex to extract response 
        response=self.extract_capture_groups(response, rf"{self.rgx_capture}")

        # Testing suggests ChatGPT refusals are all-or-nothing. 
        #   If partial responses do occur, this should may benefit from revising.
        if self._cannot_assist(response):
            self.logger.error(f'Refusal message detected in response. Skipping.')  
            return translations

        # Sometimes it will return line like "<|9>demo", and we need to fix it.
        def add_pipe(match):
            number = match.group(1)
            return f"<|{number}|>"
        response = re.sub(r"<\|?(\d+)\|?>", add_pipe, response)
        

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
        
        if self.token_count_last:
            self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')
        
        return translations

        # # self.logger.debug('-- GPT Response (filtered) --\n' + response)

        # # @NOTE: This should *should* be superflous now, due to `extract_capture_groups`:
        # # 
        # # Remove any text preceeding the first translation.
        # new_translations = re.split(r'<\|\d+\|>', 'pre_1\n' + response)[1:]
        # # new_translations = re.split(r'<\|\d+\|>', response)

        # # When there is only one query LLMs likes to exclude the <|1|>
        # if not new_translations[0].strip():
        #     new_translations = new_translations[1:]

        # if len(new_translations) <= 1 and query_size > 1:
        #     # Try splitting by newlines instead
        #     new_translations = re.split(r'\n', response)

        # if len(new_translations) > query_size:
        #     new_translations = new_translations[: query_size]
        # elif len(new_translations) < query_size:
        #     new_translations = new_translations + [''] * (query_size - len(new_translations))

        # return [t.strip() for t in new_translations]

        # self.logger.debug(translations)

        # return translations
  
    
    def _assemble_request(self, to_lang: str, prompt: str) -> Dict:
        messages = [{'role': 'system', 'content': self.chat_system_template.format(to_lang=to_lang)}]

        if to_lang in self.chat_sample:
            messages.append({'role': 'user', 'content': self.chat_sample[to_lang][0]})
            messages.append({'role': 'assistant', 'content': self.chat_sample[to_lang][1]})

        messages.append({'role': 'user', 'content': prompt})

        # Arguments for the API call:
        kwargs = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "max_tokens": self._MAX_TOKENS // 2,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout": self._TIMEOUT
        }

        return kwargs


    async def _request_translation(self, to_lang: str, prompt, isRetry=False) -> str:
        kwargs = self._assemble_request(to_lang, prompt)

        self.logger.debug("-- GPT prompt --\n" + 
                "\n".join(f"{msg['role'].capitalize()}:\n {msg['content']}" for msg in kwargs["messages"]) +
                "\n"
            )

        # import pprint
        # pp = pprint.PrettyPrinter(depth=4)

        self.logger.debug("-- kwargs --")
        self.logger.debug(kwargs)
        # self.logger.debug(pp.pprint(kwargs))
        self.logger.debug("------------")

        try:
            response = await self.client.beta.chat.completions.parse(**kwargs)

            self.logger.debug("\n-- GPT Response --\n" +
                                response.choices[0].message.content +
                                "\n------------------\n"
                            )

            if response.usage:
                self.token_count += response.usage.total_tokens
                self.token_count_last = response.usage.total_tokens
            
            if not response.choices:
                raise ValueError("Empty response from OpenAI API")
            
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error in _request_translation: {str(e)}")
            raise e

class _OllamaTranslator_JSON:
    """Internal helper class for JSON mode logic"""

    def __init__(self, translator: OllamaTranslator):
        self.translator = translator


    def _assemble_prompts(self, to_lang: str, queries: List[str]) -> List[Tuple[str, List[str]]]:
        queryList = []

        for input_ID, input_text in enumerate(queries):
            queryList.append(
                        Translation(
                            ID=input_ID,
                            text=input_text
                        )
                    )

        # Create TranslationList
        queryTL = TranslationList(Translated=queryList).model_dump_json()

        return [(queryTL, queries)]

    def _assemble_request(self, to_lang: str, prompt: str) -> Dict:
        messages = [{'role': 'system', 'content': self.translator.chat_system_template.format(to_lang=to_lang)}]
        
        if to_lang in self.translator.json_sample:
            messages.append({'role': 'user', 'content': self.translator.json_sample[to_lang][0]})
            messages.append({'role': 'assistant', 'content': self.translator.json_sample[to_lang][1]})
        
        messages.append({'role': 'user', 'content': prompt})

        # Arguments for the API call:
        kwargs = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "max_tokens": self.translator._MAX_TOKENS // 2,
            "temperature": self.translator.temperature,
            "top_p": self.translator.top_p,
            "timeout": self.translator._TIMEOUT,
            "response_format": TranslationList
        }

        return kwargs


    def _parse_response(self, response: json, prompt, queries: List[str]) -> List[str]:
        """
        Parses a JSON response from the API and maps translations to their respective positions.

        Args:
            response (json): The JSON response from the API.
            queries (List[str]): The original input values

        Returns:
            List[str]: A list of translations in the same order as the input queries.
                       If a translation is missing, the original query is preserved.
        """

        translations = queries.copy()  # Initialize with the original queries
        # translations = list(aQuery.text for aQuery in queries.Translated)
        expected_count = len(translations)

        try:
            # Parse the JSON response
            response_data = json.loads(response)

            # Validate the JSON structure
            if not isinstance(response_data, dict) or "Translated" not in response_data:
                raise ValueError("Invalid JSON structure: Missing 'Translated' key")

            translated_items = response_data["Translated"]

            # Validate that 'Translated' is a list
            if not isinstance(translated_items, list):
                raise ValueError("Invalid JSON structure: 'Translated' must be a list")

            # Process each translated item
            for item in translated_items:
                # Validate item structure
                if not isinstance(item, dict) or "ID" not in item or "text" not in item:
                    raise ValueError("Invalid translation item: Missing 'ID' or 'text'")

                id_num = item["ID"]
                translation = item["text"].strip()

                # Check if the ID is within the expected range
                if id_num < 0 or id_num > (expected_count - 1):
                    raise ValueError(f"ID {id_num} out of range (expected 0 to {(expected_count-1)})")

                # Update the translation at the correct position
                translations[id_num] = translation

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {str(e)}") from e

        return translations

