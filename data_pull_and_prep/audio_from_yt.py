from pytubefix import YouTube
from pytubefix.exceptions import VideoUnavailable


def download_audio(video_url: str,
                   video_name: str,
                   output_dir: str) -> None:
    """
    Download audio from a YouTube video.

    Args:
        video_url (str): The URL of the YouTube video.
        video_name (str): The name to be used for the output audio file 
            (without extension).
        output_dir (str): The directory where the audio file will be saved.

    Returns:
        None

    Raises:
        VideoUnavailable: If the specified video is not available.
        Exception: For any other errors that occur during the download process.

    Example:
        download_audio("https://www.youtube.com/watch?v=dQw4w9WgXcQ", 
            "never_gonna_give_you_up", "/downloads/")
        Downloading audio...
        Audio downloaded: /downloads/never_gonna_give_you_up.mp3
    """
    output_audio_file_name = output_dir + video_name + ".mp3"
    try:
        yt = YouTube(video_url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        print("Downloading audio...")
        audio_file = audio_stream.download(filename=output_audio_file_name)
        print(f"Audio downloaded: {audio_file}")
    except VideoUnavailable:
        print("The video is unavailable.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
