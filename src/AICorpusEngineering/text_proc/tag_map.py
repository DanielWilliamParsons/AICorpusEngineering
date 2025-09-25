
from types import MappingProxyType

class TagMapper:
    def __init__(self, mapping=None):
        default = {
            "/ADV": "adverb",
            "_ADV": "adverb",
        }
        self.tag_map = MappingProxyType(mapping or default)

    def map_tag(self, tag: str) -> str:
        return self.tag_map.get(tag, "word") # "word" is the fallback tag in case we get asked for a tag that doesn't exist.