import re
import asyncio
import time
import json
import logging
from typing import List, Dict

from .config_gpt import ConfigGPT
from .common import CommonTranslator, MissingAPIKeyException
from .keys import OPENAI_API_KEY, OPENAI_HTTP_PROXY, OPENAI_API_BASE, OPENAI_MODEL

try:
    import openai
except ImportError:
    openai = None

class OpenAITranslator(ConfigGPT, CommonTranslator):
    _LANGUAGE_CODE_MAP = {
        'CHS': 'Simplified Chinese',
        'CHT': 'Traditional Chinese',
        'CSY': 'Czech',
        'NLD': 'Dutch',
        'ENG': 'English',
        'FRA': 'French',
        'DEU': 'German',
        'HUN': 'Hungarian',
        'ITA': 'Italian',
        'JPN': 'Japanese',
        'KOR': 'Korean',
        'PLK': 'Polish',
        'PTB': 'Portuguese',
        'ROM': 'Romanian',
        'RUS': 'Russian',
        'ESP': 'Spanish',
        'TRK': 'Turkish',
        'UKR': 'Ukrainian',
        'VIN': 'Vietnamese',
        'CNR': 'Montenegrin',
        'SRP': 'Serbian',
        'HRV': 'Croatian',
        'ARA': 'Arabic',
        'THA': 'Thai',
        'IND': 'Indonesian'
    }
    
    _MAX_REQUESTS_PER_MINUTE = 200
    _TIMEOUT = 40
    _RETRY_ATTEMPTS = 3
    _MAX_TOKENS = 8192

    def __init__(self, check_openai_key=True):
        _CONFIG_KEY = 'chatgpt.' + OPENAI_MODEL
        ConfigGPT.__init__(self, config_key=_CONFIG_KEY)
        CommonTranslator.__init__(self)
        
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
        self.token_count = 0
        self.token_count_last = 0
        self._last_request_ts = 0


    def parse_args(self, args: CommonTranslator):
        self.config = args.chatgpt_config


    def _cannot_assist(self, response: str) -> bool:
        # Common refusal terms
        ERROR_KEYWORDS = [
            # ENG_KEYWORDS
            r"I must decline",
            r'(i(\'m| am)?\s+)?sorry(.|\n)*?(can(\'t|not)|unable to|cannot)\s+(assist|help)',
            # CHINESE_KEYWORDS (using regex patterns)
            r"抱歉，?我(无法|不能)",  # Matches "抱歉，我无法" or "抱歉我不能"
            r"对不起，?我(无法|不能)",  # Matches "对不起，我无法" or "对不起我不能"
            r"我无法(满足|回答|处理)",  # Matches "我无法满足" or "我无法回答" or "我无法处理"
            r"这超出了我的范围",  # Matches "这超出了我的范围"
            r"我不便回答",  # Matches "我不便回答"
            r"我不能提供相关建议",  # Matches "我不能提供相关建议"
            r"这类内容我不能处理",  # Matches "这类内容我不能处理"
            r"我需要婉拒",  # Matches "我需要婉拒"
            # JAPANESE_KEYWORDS
            r"申し訳ありませんが",
        ]

        # Use regex to check for common variants of refusal phrases.
        #   Check for `ERROR_KEYWORDS` for other variants, languages
        refusal_pattern = re.compile(
            '|'.join(ERROR_KEYWORDS),re.IGNORECASE
        )

        # Check if any refusal pattern matches the response
        return bool(refusal_pattern.search(response.strip().lower()))

    def _assemble_prompts(self, from_lang: str, to_lang: str, queries: List[str]):
        prompt=''

        if self.include_template:
            prompt = self.prompt_template.format(to_lang=to_lang)
        
        for i, query in enumerate(queries):
            prompt += f"\n<|{i+1}|>{query}"
        
        return [prompt.lstrip()], len(queries)

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        translations = [''] * len(queries)
        prompt, _ = self._assemble_prompts(from_lang, to_lang, queries)
        
        for attempt in range(self._RETRY_ATTEMPTS):
            try:
                response_text = await self._request_translation(to_lang, prompt[0])
                if not self.json_mode:
                    translations = self._parse_response(response_text, queries)
                else:
                    translations = self._parse_json_response(response_text, queries)
                return translations
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
        
    def _parse_json_response(self, response: str, queries: List[str]) -> List[str]:
        """
        Parses a JSON response from the API and maps translations to their respective positions.

        Args:
            response (str): The JSON response from the API.
            queries (List[str]): The original list of queries/input lines.

        Returns:
            List[str]: A list of translations in the same order as the input queries.
                       If a translation is missing, the original query is preserved.
        """
        translations = queries.copy()  # Initialize with the original queries
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

                # Extract and validate the ID format
                id_str = item["ID"]
                id_match = re.match(r'^(\d+)$', id_str)  # Match numeric IDs without <| |>
                if not id_match:
                    raise ValueError(f"Invalid ID format: {id_str} (expected numeric ID)")

                id_num = int(id_match.group(1))
                translation = item["text"].strip()

                # Check if the ID is within the expected range
                if id_num < 1 or id_num > expected_count:
                    raise ValueError(f"ID {id_num} out of range (expected 1 to {expected_count})")

                # Update the translation at the correct position
                translations[id_num - 1] = translation

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {str(e)}") from e

        return translations


    async def _request_translation(self, to_lang: str, prompt: str) -> str:
        messages = [{
            "role": "system",
            "content": self.chat_system_template.format(to_lang=to_lang)
        },{
            'role': 'user', 'content': self.chat_sample[to_lang][0]
        },{
            'role': 'assistant', 'content': self.chat_sample[to_lang][1]
        },{
            "role": "user",
            "content": prompt
        }]

        self.logger.debug("-- GPT prompt --\n" + 
                "\n".join(f"{msg['role'].capitalize()}:\n {msg['content']}" for msg in messages) +
                "\n"
            )

        # Arguments for the API call:
        kwargs = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "max_tokens": self._MAX_TOKENS // 2,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout": self._TIMEOUT
        }

        # If requesting JSON output:
        if self.json_mode:
            messages[1] =  {'role': 'user', 'content': self.json_sample[to_lang][0]}
            messages[2] =  {'role': 'user', 'content': self.json_sample[to_lang][1]}
            kwargs["messages"] = messages

            kwargs["response_format"] =  self.json_schema


        self.logger.debug("-- kwargs --")
        self.logger.debug(kwargs)
        self.logger.debug("------------")

        try:
            response = await self.client.chat.completions.create(**kwargs)

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

        except openai.RateLimitError as e:
            self.logger.error("Rate limit exceeded. Consider upgrading your plan or adding payment method.")
            raise
        except openai.APIError as e:
            self.logger.error(f"API error: {e.message}")
            raise
        except Exception as e:
            self.logger.error(f"Error in _request_translation: {str(e)}")
            raise

    async def _ratelimit_sleep(self):
        if self._MAX_REQUESTS_PER_MINUTE > 0:
            now = time.time()
            delay = 60 / self._MAX_REQUESTS_PER_MINUTE
            if now - self._last_request_ts < delay:
                await asyncio.sleep(delay - (now - self._last_request_ts))
            self._last_request_ts = time.time()