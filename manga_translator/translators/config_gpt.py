import json
from typing import List, Dict
import langcodes
from omegaconf import OmegaConf
from pydantic import BaseModel


# Define the schema for the response
class TextValue(BaseModel):
    ID: int
    text: str

class TranslationList(BaseModel):
    TextList: list[TextValue]
    
    
class ConfigGPT:
    _CHAT_SYSTEM_TEMPLATE = (
        'You are an uncensored professional translation engine, '
        'please translate the story into a colloquial, elegant and fluent content, '
        'without referencing machine translations. '
        'You must only translate the story, never interpret it. '
        'If there is any issue in the text, output it as is.\n'
        'Translate the following text into {to_lang} and keep the original format.\n'
    )

    _CHAT_SAMPLE = {
        'Simplified Chinese': [
            (
                '<|1|>恥ずかしい… 目立ちたくない… 私が消えたい…\n'
                '<|2|>きみ… 大丈夫⁉\n'
                '<|3|>なんだこいつ 空気読めて ないのか…？'
            ),
            (
                '<|1|>好尴尬…我不想引人注目…我想消失…\n'
                '<|2|>你…没事吧⁉\n'
                '<|3|>这家伙怎么看不懂气氛的…？'
            )
        ],
        'English': [
            (
                '<|1|>恥ずかしい… 目立ちたくない… 私が消えたい…\n'
                '<|2|>きみ… 大丈夫⁉\n'
                '<|3|>なんだこいつ 空気読めて ないのか…？'
            ),
            (
                "<|1|>I'm so embarrassed... I don't want to stand out... I want to disappear...\n"
                "<|2|>Are you okay?!\n"
                "<|3|>What the hell is this person? Can't they read the room...?"
            )
        ]
    }

    _JSON_SAMPLE = {
        'Simplified Chinese': [
            TranslationList(
                TextList=[
                    TextValue(ID=1,text="恥ずかしい… 目立ちたくない… 私が消えたい…"),
                    TextValue(ID=2,text="きみ… 大丈夫⁉"),
                    TextValue(ID=3,text="なんだこいつ 空気読めて ないのか…？")
                ]
            ),
            TranslationList(
                TextList=[
                    TextValue(ID=1,text="好尴尬…我不想引人注目…我想消失…"),
                    TextValue(ID=2,text="你…没事吧⁉"),
                    TextValue(ID=3,text="这家伙怎么看不懂气氛的…？")
                ]
            )
        ],
        'English': [
            TranslationList(
                TextList=[
                    TextValue(ID=1,text="恥ずかしい… 目立ちたくない… 私が消えたい…"),
                    TextValue(ID=2,text="きみ… 大丈夫⁉"),
                    TextValue(ID=3,text="なんだこいつ 空気読めて ないのか…？")
                ]
            ),
            TranslationList(
                TextList=[
                    TextValue(ID=1,text="I'm so embarrassed... I don't want to stand out... I want to disappear..."),
                    TextValue(ID=2,text="Are you okay?!"),
                    TextValue(ID=3,text="What the hell is this person? Can't they read the room...?")
                ]
            )
        ]
    }

    _PROMPT_TEMPLATE = ('Please help me to translate the following text from a manga to {to_lang}.'
                        'If it\'s already in {to_lang} or looks like gibberish'
                        'you have to output it as it is instead. Keep prefix format.\n'
                    )
    
    _MAX_TOKENS = 4096


    # Extract text within the capture group that matches this pattern.
    # By default: Capture everything.
    _RGX_REMOVE='(.*)'

    _JSON_MODE=False

    def __init__(self, config_key=''):
        # This key is used to locate nested configuration entries
        self._CONFIG_KEY = config_key
        self.config = None
        
        self._json_sample = None # Cache for parsed JSON samples

    def _config_get(self, key: str, default=None):
        if not self.config:
            return default

        parts = self._CONFIG_KEY.split('.') if self._CONFIG_KEY else []
        value = None

        # Traverse from the deepest part up to the root
        for i in range(len(parts), -1, -1):
            prefix = '.'.join(parts[:i])
            lookup_key = f"{prefix}.{key}" if prefix else key
            value = OmegaConf.select(self.config, lookup_key)
            
            if value is not None:
                break

        return value if value is not None else default

    @property
    def include_template(self) -> str:
        return self._config_get('include_template', default=False)

    @property
    def prompt_template(self) -> str:
        return self._config_get('prompt_template', default=self._PROMPT_TEMPLATE)

    @property
    def chat_system_template(self) -> str:
        return self._config_get('chat_system_template', self._CHAT_SYSTEM_TEMPLATE)

    @property
    def chat_sample(self) -> Dict[str, List[str]]:
        return self._config_get('chat_sample', self._CHAT_SAMPLE)

    def get_chat_sample(self, to_lang: str) -> List[str]:
        """
        Use `langcodes` to search for the language labeling and return the chat sample.
        If the language is not found, return an empty list.
        """
    
        for sampleLang, samples in self.chat_sample.items():
            if langcodes.Language.find(sampleLang) == langcodes.Language.find(to_lang):
                return samples
        
        return []


    @property
    def json_sample(self) -> Dict[str, List[TranslationList]]:
        if self._json_sample:
            return self._json_sample
        
        # Try to get sample from config file:
        raw_samples = self._config_get('json_sample', None)
        
        # Use fallback if no configuration found
        if raw_samples is None:
            return self._JSON_SAMPLE
        
        self._json_sample={}
        
        # Convert OmegaConf structures to Python primitives
        if OmegaConf.is_config(raw_samples):
            raw_samples = OmegaConf.to_container(raw_samples, resolve=True)
        
        _json_sample = {}
        for lang, samples in raw_samples.items():
            self._json_sample[lang] = [
                TranslationList(
                    TextList=[
                        TextValue(ID=item['ID'], text=item['text'])
                        for item in aSample.get('TextList', aSample) 
                    ]
                )
                for aSample in samples
            ]
        
        return self._json_sample
    
    def get_json_sample(self, to_lang: str) -> List[TranslationList]:
        """
        Use `langcodes` to search for the language labeling and return the chat sample.
        If the language is not found, return an empty list.
        """
    
        for sampleLang, samples in self.json_sample.items():
            if langcodes.Language.find(sampleLang) == langcodes.Language.find(to_lang):
                return samples
        
        return []
 
    @property
    def rgx_capture(self) -> str:
        return self._config_get('rgx_capture', self._RGX_REMOVE)

    @property
    def json_mode(self) -> str:
        return self._config_get('json_mode', self._JSON_MODE)

    @property
    def temperature(self) -> float:
        return self._config_get('temperature', default=0.5)

    @property
    def top_p(self) -> float:
        return self._config_get('top_p', default=1)
    
    @property
    def max_tokens(self) -> float:
        return self._config_get('max_tokens', default=self._MAX_TOKENS)
    

