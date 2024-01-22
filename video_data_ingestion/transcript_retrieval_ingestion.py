from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi

from vector_db.chroma import EnhVectorDatabase
from video_data_ingestion.ingestion_utils import load_transcript_into_vector_storage


def download_and_load_yt_transcript_from_url(vector_db: EnhVectorDatabase, url: str):
    """
    Manages to download the transcript from the url and save it into the vector storage.
    """
    video_id = get_yt_id_from_url(url=url)

    if video_id:
        transcript_filename = 'transcripts/trascript_' + video_id + '.txt'

        print("YT: downloading transcript...")
        download_transcript(video_id=video_id, filename=transcript_filename)

        print("YT: loading transcript into vector storage...")
        load_transcript_into_vector_storage(filename=transcript_filename, vector_db=vector_db)
    else:
        print(f'Impossible to retrieve video ID of {url}.')


def get_yt_id_from_url(url: str):
    """
    Gets the ID related to the video from the url.
    """
    try:
        yt = YouTube(url)
        video_id = yt.video_id
        return video_id
    except Exception as e:
        print(f'Error while retrieving ID of the video with URL: {e}')
        return None


def download_transcript(video_id, filename):
    """
    Retrieves the transcript for the video with ID=video_id and saves it
    in a txt file with the given filename.
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])

        if transcript:
            with open(filename, 'w', encoding='utf-8') as file:
                for entry in transcript:
                    file.write(entry['text'] + '\n')
            print(f'Transcript of the video {video_id} saved successfully.')
            return True
        else:
            print(f'No transcript for the video {video_id}.')
            return False

    except Exception as e:
        print(f'Error during transcript download of video {video_id}: {e}')
        return False
