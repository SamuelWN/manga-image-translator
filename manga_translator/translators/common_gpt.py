import json
import re
import tiktoken

from .config_gpt import ConfigGPT, TextValue, TranslationList
from .common import CommonTranslator, VALID_LANGUAGES
from typing import List, Dict, Tuple

import openai

class CommonGPTTranslator(ConfigGPT, CommonTranslator):
    _LANGUAGE_CODE_MAP=VALID_LANGUAGES # Assume that GPT translators support all languages

    def __init__(self, config_key: str, MODEL_NAME=''):
        self._tokenizer_cache: dict = {}
        ConfigGPT.__init__(self, config_key=config_key)
        CommonTranslator.__init__(self)
        self.MODEL=MODEL_NAME

    def parse_args(self, args: CommonTranslator):
        self.config = args.chatgpt_config


    """Utility functions for GPT-based translators"""
    def extract_capture_groups(self, text: str, regex=r"(.*)"):
        """
        Extracts all capture groups from matches and concatenates them into a single string.
        
        :param text: The multi-line text to search.
        :param regex: The regex pattern with capture groups.
        :return: A concatenated string of all matched groups.
        """
        pattern = re.compile(regex, re.DOTALL)  # DOTALL to match across multiple lines
        matches = pattern.findall(text)  # Find all matche

        # Ensure matches are concatonated (handles multiple groups per match)
        extracted_text = "\n".join(
            "\n".join(m) if isinstance(m, tuple) else m for m in matches
        )
        
        return extracted_text.strip() if extracted_text else None

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

    def count_tokens(self, text: str, encoding_name: str) -> int:
        """Count tokens using the specified encoding."""
        if encoding_name not in self._tokenizer_cache:
            self._tokenizer_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
        encoding = self._tokenizer_cache[encoding_name]
        return len(encoding.encode(text))


    """Common functions for GPT-based translators"""
    def _assemble_request(self, to_lang: str, prompt: str) -> Dict:
        messages = [{'role': 'system', 'content': self.chat_system_template.format(to_lang=to_lang)}]

        if to_lang in self.chat_sample:
            messages.append({'role': 'user', 'content': self.chat_sample[to_lang][0]})
            messages.append({'role': 'assistant', 'content': self.chat_sample[to_lang][1]})

        messages.append({'role': 'user', 'content': prompt})

        # Arguments for the API call:
        kwargs = {
            "model": self.MODEL,
            "messages": messages,
            "max_tokens": self._MAX_TOKENS // 2,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout": self._TIMEOUT
        }

        return kwargs

    async def _request_translation(self, to_lang: str, prompt) -> str:
        kwargs = self._assemble_request(to_lang, prompt)

        self.logger.debug("-- GPT prompt --\n" + 
                "\n".join(f"{msg['role'].capitalize()}:\n {msg['content']}" for msg in kwargs["messages"]) +
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

    def _assemble_prompts(self, to_lang: str, queries: List[str], encoding='cl100k_base') -> List[Tuple[str, List[str]]]:
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
            token_count += self.count_tokens(template)

        # Add queries until token limit is reached
        included_queries = []
        for i, query in enumerate(queries):
            query_section = f'\n<|{i + 1}|>{query}'
            query_tokens = self.count_tokens(query_section, encoding_name=encoding)

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
 
    def _parse_response(self, response: str, queries: List[str]) -> List[str]:
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


class _CommonGPTTranslator_JSON:
    """Internal helper class for JSON mode logic"""

    def __init__(self, translator: CommonGPTTranslator):
        self.translator = translator

    def _assemble_prompts(self, to_lang: str, queries: List[str], encoding='cl100k_base') -> List[Tuple[str, List[str]]]:
        retList = []  # Fixed initialization
        queryList = []
        queryChunk = []
        input_ID = 0

        for input_text in queries:
            # temp list, to check if it exceeds token limit:
            temp_list = queryList + [TextValue(ID=input_ID, text=input_text)]
            temp_json = TranslationList(TextList=temp_list).model_dump_json()
            total_tokens = self.translator.count_tokens(temp_json, encoding)
            
            if total_tokens <= self.translator._MAX_TOKENS:
                # Add to current chunk
                queryList = temp_list
                queryChunk.append(input_text)
                input_ID += 1
            else:
                # Finalize current chunk if it has content
                if queryList:
                    retList.append((
                        TranslationList(TextList=queryList).model_dump_json(),
                        queryChunk
                    ))
                # Start new chunk with current item (even if it exceeds limit)
                queryList = [TextValue(ID=0, text=input_text)]
                queryChunk = [input_text]
                input_ID = 0  # Reset ID counter for new chunk

        # Add remaining items
        if queryList:
            retList.append((
                TranslationList(TextList=queryList).model_dump_json(),
                queryChunk
            ))

        return retList

    def _assemble_request(self, to_lang: str, prompt: str) -> Dict:
        messages = [{'role': 'system', 'content': self.translator.chat_system_template.format(to_lang=to_lang)}]
        
        jSample=self.translator.get_json_sample(to_lang)
        if jSample:
            messages.append({'role': 'user', 'content': jSample[0].model_dump_json()})
            messages.append({'role': 'assistant', 'content': jSample[1].model_dump_json()})
        else:
            # If no appropriate `json_sample` is available, but a `chat_sample` is found: 
            #   Convert and use the `chat_sample`
            chatSample=self.translator.chat_sample.get(to_lang)
            if chatSample:
                asJSON = [
                    self.text2json(self.translator.chat_sample[0]).model_dump_json(),
                    self.text2json(self.translator.chat_sample[1]).model_dump_json()
                ]

                messages.append({'role': 'user', 'content': asJSON[0]})
                messages.append({'role': 'assistant', 'content': asJSON[1]})


        messages.append({'role': 'user', 'content': prompt})

        # Arguments for the API call:
        kwargs = {
            "model": self.translator.MODEL,
            "messages": messages,
            "max_tokens": self.translator._MAX_TOKENS // 2,
            "temperature": self.translator.temperature,
            "top_p": self.translator.top_p,
            "timeout": self.translator._TIMEOUT,
            "response_format": TranslationList
        }

        return kwargs

    def _parse_response(self, response: json, queries: List[str]) -> List[str]:
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
        expected_count = len(translations)

        try:
            # Parse the JSON response
            response_data = json.loads(response)
            # Validate the JSON structure
            if not isinstance(response_data, dict) or "TextList" not in response_data:
                raise ValueError("Invalid JSON structure: Missing 'TextList' key")

            translated_items = response_data["TextList"]

            # Validate that 'TextList' is a list
            if not isinstance(translated_items, list):
                raise ValueError("Invalid JSON structure: 'TextList' must be a list")

            rangeOffset = min([val['ID'] for val in translated_items])

            # Process each translated item
            for item in translated_items:
                # Validate item structure
                if not isinstance(item, dict) or "ID" not in item or "text" not in item:
                    raise ValueError("Invalid translation item: Missing 'ID' or 'text'")

                id_num = item["ID"]
                translation = item["text"].strip()

                # Check if the ID is within the expected range
                if (id_num < 0) or (id_num > (expected_count - (rangeOffset+1))):
                    raise ValueError(f"ID {id_num} out of range (expected 0 to {(expected_count-1)})")

                # Update the translation at the correct position
                translations[id_num - rangeOffset] = translation

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {str(e)}") from e

        return translations
 
    def text2json(text: str) -> TranslationList:
        """
        Convert text samples to TranslationList format.
        Assists with backwards compatiblity for `<|ID|>`-based samples.
        
        Args:
            input_data: Text samples, keyed by `<|ID|>` tags
            
        Returns:
            Text samples stored as a TranslationList
        """

        segment_pattern = re.compile(r'<\|(\d+)\|>(.*?)(?=<\|(\d+)\|>|$)', re.DOTALL)
        segments = segment_pattern.findall(text)

        jsonified=TranslationList(
                            TextList=[
                                TextValue(
                                    ID=int(seg[0]),
                                    text=seg[1].strip()
                                ) for seg in segments
                            ]
                        )

        return jsonified
