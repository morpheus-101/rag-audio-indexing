from pytubefix import YouTube
from pytubefix.cli import on_progress
from pytubefix.exceptions import VideoUnavailable


def download_audio(video_url: str,
                   video_name: str,
                   output_dir: str):
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
