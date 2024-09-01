import json
from typing import List, Tuple
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode


def wrap_text(text: str, max_width: int = 79) -> str:
    """
    Wrap text to a specified maximum width.

    Args:
        text (str): The input text to be wrapped.
        max_width (int, optional): The maximum width of each line. Defaults to 79.

    Returns:
        str: The wrapped text with lines separated by newline characters.
    """
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


def default_parser(json_string: str) -> Tuple[List[str], List[str]]:
    """
    Parse a JSON string to extract entities and relationships.

    Args:
        json_string (str): A JSON-formatted string containing 'Entities' and 'Relationships' keys.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists:
            - The first list contains entities.
            - The second list contains relationships.
    """
    data = json.loads(json_string)
    entities = data["Entities"] if data["Entities"] else []
    relationships = data["Relationships"] if data["Relationships"] else []
    result = (entities, relationships)
    return result


class TranscriptionParser:
    """
    A class for parsing transcriptions with character timestamps and generating
    text nodes.
    """

    def __init__(self, transcription_with_char_timestamps: List[Tuple[str, float]]):
        """
        Initialize the TranscriptionParser with character-level timestamps.

        Args:
            transcription_with_char_timestamps (List[Tuple[str, float]]): A list
                of tuples containing characters and their timestamps.
        """
        self.transcription_with_char_timestamps = transcription_with_char_timestamps

    def _get_full_text_from_char_timestamps(self) -> str:
        """
        Extract the full text from the character-level timestamps.

        Returns:
            str: The full text of the transcription.
        """
        full_text_string: str = ""
        for char, _ in self.transcription_with_char_timestamps:
            full_text_string += char
        return full_text_string

    def get_nodes(self) -> List[TextNode]:
        """
        Generate text nodes from the transcription.

        Returns:
            List[TextNode]: A list of TextNode objects representing chunks of
            the transcription.
        """
        transcription_text: str = self._get_full_text_from_char_timestamps()
        text_parser: SentenceSplitter = SentenceSplitter(chunk_size=300,
                                                         chunk_overlap=100)
        text_chunks: List[str] = text_parser.split_text(transcription_text)
        nodes: List[TextNode] = []
        for text_chunk in text_chunks:
            node: TextNode = TextNode(text=text_chunk)
            nodes.append(node)
        return nodes
