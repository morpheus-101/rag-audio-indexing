import json
from typing import List, Tuple
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode


def wrap_text(text, max_width=79):
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + len(current_line) <= max_width:
            current_line.append(word)
            current_length += len(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)

    if current_line:
        lines.append(' '.join(current_line))
    return '\n'.join(lines)


def default_parser(json_string: str) -> Tuple[List, List]:
    data = json.loads(json_string)
    entities = data["Entities"] if data["Entities"] else []
    relationships = data["Relationships"] if data["Relationships"] else []
    result = (entities, relationships)
    return result


class TranscriptionParser:
    def __init__(self, transcription_with_char_timestamps):
        self.transcription_with_char_timestamps = (
            transcription_with_char_timestamps)

    def _get_full_text_from_char_timestamps(self):
        full_text_string: str = ""
        for char, _ in self.transcription_with_char_timestamps:
            full_text_string += char
        return full_text_string

    def get_nodes(self):
        transcription_text = self._get_full_text_from_char_timestamps()
        text_chunks = []
        text_parser = SentenceSplitter(chunk_size=300, chunk_overlap=100)
        text_chunks = text_parser.split_text(transcription_text)
        nodes = []
        for idx, text_chunk in enumerate(text_chunks):
            node = TextNode(text=text_chunk,)
            nodes.append(node)
        return nodes
