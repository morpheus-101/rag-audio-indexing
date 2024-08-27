from typing import List, Tuple, Dict, Any
import whisper  # type: ignore
from llama_index.core.node_parser import SentenceSplitter


def transcribe(audio_file_path: str) -> List[Tuple[int, str, float, float]]:
    """
    Transcribes the audio file located at the given path.

    Args:
        audio_file_path (str): The path to the audio file.

    Returns:
        List[Tuple[int, str, float, float]]: A list of tuples containing
            the transcribed segments.
        Each tuple consists of the segment ID, transcribed text,
        start time, and end time.
    """
    model = whisper.load_model("base")
    result: Dict[str, Any] = model.transcribe(audio_file_path)

    segments: List[Dict[str, Any]] = result["segments"]

    return [
        (
            int(segment.get("id", 0)),
            str(segment.get("text", "")),
            float(segment.get("start", 0.0)),
            float(segment.get("end", 0.0)),
        )
        for segment in segments
    ]


def map_characters_to_timestamps(
    transcription: List[Tuple[int, str, float, float]]
) -> List[Tuple[str, float]]:
    """
    Maps characters to their corresponding timestamps in a transcription.

    Args:
        transcription (List[Tuple[int, str, float, float]]): A list of tuples
            representing the transcription.
            Each tuple contains the following elements:
            - id (int): The identifier of the transcription.
            - text (str): The text of the transcription.
            - start (float): The start timestamp of the transcription.
            - end (float): The end timestamp of the transcription.

    Returns:
        List[Tuple[str, float]]: A list of tuples representing the characters
            and their corresponding timestamps.
            Each tuple contains the following elements:
            - char (str): The character.
            - char_time (float): The timestamp of the character.

    """
    char_timestamps = []
    for id, text, start, end in transcription:
        duration = end - start
        char_count = len(text)
        time_per_char = duration / char_count
        for i, char in enumerate(text):
            char_time = start + (i * time_per_char)
            char_timestamps.append((char, char_time))
    return char_timestamps


class CreateCustomTextChunks:
    def __init__(
        self,
        transcription_with_char_timestamps: List[Tuple[str, float]],
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
    ) -> None:
        """
        Initialize the DataPreparation object.

        Args:
            transcription_with_char_timestamps (str): The transcription
                with character timestamps.
            chunk_size (int, optional): The size of each chunk for
                text parsing. Defaults to 1024.
        """
        self.transcription_with_char_timestamps = (
            transcription_with_char_timestamps)
        self.full_text_string: str = ""
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap

        self._get_full_text_from_char_timestamps()
        self.text_parser: SentenceSplitter = SentenceSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def _get_full_text_from_char_timestamps(self) -> None:
        full_text_string: str = ""
        for char, _ in self.transcription_with_char_timestamps:
            full_text_string += char
        self.full_text_string = full_text_string

    def _get_start_end_idx(
            self, all_text_chunks: List[str]) -> List[Tuple[int, int]]:
        text_chunks_start_end_idx = []
        for i, text_chunk in enumerate(all_text_chunks):
            if text_chunk not in self.full_text_string:
                assert f"Chunk {i} not found in full text"
            else:
                start_idx = self.full_text_string.find(text_chunk)
                end_idx = start_idx + len(text_chunk) - 1
                text_chunks_start_end_idx.append((start_idx, end_idx))
        return text_chunks_start_end_idx

    def _get_time_stamps(
        self, start_end_idx: List[Tuple[int, int]]
    ) -> List[Tuple[float, float]]:
        text_chunks_time_stamps = []
        for start_idx, end_idx in start_end_idx:
            start_time = self.transcription_with_char_timestamps[start_idx][1]
            end_time = self.transcription_with_char_timestamps[end_idx][1]
            text_chunks_time_stamps.append((start_time, end_time))
        return text_chunks_time_stamps

    def _combine_text_chunks_with_timestamps(
        self,
        all_text_chunks: List[str],
        all_text_chunk_timestamps: List[Tuple[float, float]],
    ) -> List[Tuple[str, Tuple[float, float]]]:
        text_chunks_with_timestamps = []
        for i in range(len(all_text_chunks)):
            text_chunks_with_timestamps.append(
                (all_text_chunks[i], all_text_chunk_timestamps[i])
            )
        return text_chunks_with_timestamps

    def create_custom_text_chunks(
            self) -> List[Tuple[str, Tuple[float, float]]]:
        text_chunks = []
        text_chunks = self.text_parser.split_text(self.full_text_string)
        start_end_idx = self._get_start_end_idx(text_chunks)
        time_stamps = self._get_time_stamps(start_end_idx)
        text_chunks_with_timestamps = (
            self._combine_text_chunks_with_timestamps(
                text_chunks, time_stamps))
        return text_chunks_with_timestamps
