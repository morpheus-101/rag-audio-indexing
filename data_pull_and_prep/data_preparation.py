from typing import List, Tuple
import whisper


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
    result = model.transcribe(audio_file_path)
    return [(segment["id"],
             segment["text"],
             segment["start"],
             segment["end"])
            for segment in result["segments"]]


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
