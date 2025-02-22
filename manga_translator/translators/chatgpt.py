import re
import asyncio
import time
import json

from typing import List, Dict, Tuple
from .config_gpt import ConfigGPT, TranslationList
from .common import CommonTranslator, MissingAPIKeyException, VALID_LANGUAGES
from .keys import OPENAI_API_KEY, OPENAI_HTTP_PROXY, OPENAI_API_BASE, OPENAI_MODEL
from .common_gpt import CommonGPTTranslator

try:
    import openai
except ImportError:
    openai = None

class OpenAITranslator(CommonGPTTranslator):
    _LANGUAGE_CODE_MAP = VALID_LANGUAGES
    _MAX_REQUESTS_PER_MINUTE = 200
    _TIMEOUT = 40
    _RETRY_ATTEMPTS = 3
    _MAX_TOKENS = 8192

    def __init__(self, check_openai_key=True):
        _CONFIG_KEY = 'chatgpt.' + OPENAI_MODEL
        CommonGPTTranslator.__init__(self, config_key=_CONFIG_KEY)
        # ConfigGPT.__init__(self, config_key=_CONFIG_KEY)
        # CommonTranslator.__init__(self)
        # super().__init__(config_key=_CONFIG_KEY)
        

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


    def _get_encoding_for_model(self) -> str:
        """Determine the encoding name for the OpenAI model."""
        self.logger.debug("OPENAI_MODEL: ")
        self.logger.debug(OPENAI_MODEL)
        self.logger.debug(type(OPENAI_MODEL))

        import tiktoken

        try:
            # Use tiktoken's built-in mapping for OpenAI models
            encoding = tiktoken.encoding_for_model(OPENAI_MODEL)
            return encoding.name
        except KeyError:
            return "cl100k_base"  # Fallback for unknown OpenAI models

    def _count_tokens(self, text: str) -> int:
        """Count tokens for OpenAI models."""
        encoding_name = self._get_encoding_for_model()
        return self.count_tokens(text, encoding_name)


    # def _assemble_prompts(self, from_lang: str, to_lang: str, queries: List[str]):
    #     prompt=''

    #     if self.include_template:
    #         prompt = self.prompt_template.format(to_lang=to_lang)
        
    #     for i, query in enumerate(queries):
    #         prompt += f"\n<|{i+1}|>{query}"
        
    #     return [prompt.lstrip()], len(queries)

    # async def _assemble_prompts(self, from_lang: str, to_lang: str, queries: List[str]):
    #     """
    #     Assemble prompts while respecting token limits.
    #     Yields tuples of (prompt, queries_included) for each chunk.
    #     """
    #     prompt = ''
    #     token_count = 0
    #     i_offset = 0  # Tracks the offset for query numbering

    #     # Add template if enabled
    #     if self.include_template:
    #         template = self.prompt_template.format(to_lang=to_lang)
    #         prompt += template
    #         token_count += await self._count_tokens(template)


    #     # Process each query
    #     for i, query in enumerate(queries):
    #         query_section = f'\n<|{i + 1 - i_offset}|>{query}'
    #         query_tokens = await self._count_tokens(query_section)

    #         # Check if adding this query would exceed the token limit
    #         if token_count + query_tokens > self._MAX_TOKENS:
    #             # Yield the current prompt and reset
    #             if self._RETURN_PROMPT:
    #                 prompt += '\n<|1|>'
    #                 token_count += await self._count_tokens('\n<|1|>')

    #             yield prompt.lstrip(), queries[i_offset:i]  # Yield queries included so far

    #             # Reset for the next prompt
    #             prompt = self.prompt_template.format(to_lang=to_lang) if self.include_template else ''
    #             token_count = await self._count_tokens(prompt)
    #             i_offset = i  # Reset query numbering

    #             # Re-add the current query to the new prompt
    #             query_section = f'\n<|1|>{query}'
    #             query_tokens = await self._count_tokens(query_section)

    #         # Add the query to the prompt
    #         prompt += query_section
    #         token_count += query_tokens

    #     # Yield the final prompt
    #     if self._RETURN_PROMPT:
    #         prompt += '\n<|1|>'
    #         token_count += await self._count_tokens('\n<|1|>')

    #     yield prompt.lstrip(), queries[i_offset:]  # Yield remaining queries


    def _assemble_prompts(self, from_lang: str, to_lang: str, queries: List[str]) -> List[Tuple[str, List[str]]]:
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
                    remaining_prompts = self._assemble_prompts(from_lang, to_lang, queries[i:])
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
        translations = [''] * len(queries)
        
        for prompt, query in self._assemble_prompts(from_lang, to_lang, queries):
            for attempt in range(self._RETRY_ATTEMPTS):
                try:
                    response_text = await self._request_translation(to_lang, prompt[0])
                    if not self.json_mode:
                        translations.append(self._parse_response(response_text, queries))
                    else:
                        translations.append(self._parse_json_response(response_text, queries))
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
                # id_str = item["ID"]
                # id_match = re.match(r'^(\d+)$', id_str)  # Match numeric IDs without <| |>
                # if not id_match:
                #     raise ValueError(f"Invalid ID format: {id_str} (expected numeric ID)")

                id_num = item["ID"]
                # id_num = int(id_match.group(1))
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
        messages = [{'role': 'system', 'content': self.chat_system_template.format(to_lang=to_lang)}]

        # Arguments for the API call:
        kwargs = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "max_tokens": self._MAX_TOKENS // 2,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout": self._TIMEOUT
        }

        if self.json_mode:
            kwargs["response_format"] =  TranslationList
            
            if to_lang in self.json_sample:
                kwargs["messages"].append({'role': 'user', 'content': self.json_sample[to_lang][0]})
                kwargs["messages"].append({'role': 'assistant', 'content': self.json_sample[to_lang][1]})
        elif to_lang in self.chat_sample:
            kwargs["messages"].append({'role': 'user', 'content': self.chat_sample[to_lang][0]})
            kwargs["messages"].append({'role': 'assistant', 'content': self.chat_sample[to_lang][1]})

        kwargs["messages"].append({'role': 'user', 'content': prompt})

        self.logger.debug("-- GPT prompt --\n" + 
                "\n".join(f"{msg['role'].capitalize()}:\n {msg['content']}" for msg in messages) +
                "\n"
            )

        self.logger.debug("-- kwargs --")
        self.logger.debug(kwargs)
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