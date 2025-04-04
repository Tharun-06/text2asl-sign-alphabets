def get_youtube_subtitles(video_url: str) -> str | None:
    print(f"\n--- Fetching Subtitles for: {video_url} ---")
    try:
        from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
    except ImportError:
        print("Error: 'youtube-transcript-api' not installed.")
        print("Please install it: pip install youtube-transcript-api")
        return None
    try:
        video_id = None
        if "v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1].split("?")[0]
        if not video_id:
            print("Error: Could not extract video ID.")
            return None
        print(f"Extracted Video ID: {video_id}")
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        print(f"Available transcript languages: {[t.language for t in transcript_list]}")
        transcript = None
        try:
            transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
            print("Found English transcript.")
        except NoTranscriptFound:
            print("No English transcript found. Trying auto-generated...")
            try:
                transcript = transcript_list.find_generated_transcript(['en'])
                print("Found auto-generated English transcript.")
            except NoTranscriptFound:
                print("No auto-generated English transcript found. Trying first available...")
                try:
                    available_langs = list(transcript_list._manually_created_transcripts.keys()) + list(transcript_list._generated_transcripts.keys())
                    if not available_langs:
                        raise NoTranscriptFound("No transcripts available at all.")
                    transcript = transcript_list.find_transcript(available_langs)
                    print(f"Found transcript in language: {transcript.language}")
                except NoTranscriptFound:
                    print("Error: No suitable transcripts found.")
                    return None
        transcript_data = transcript.fetch()
        full_text = " ".join([item.text for item in transcript_data if hasattr(item, 'text')])
        print(f"Subtitles fetched and combined successfully (Length: {len(full_text)} chars).")
        return full_text
    except TranscriptsDisabled:
        print("Error: Transcripts are disabled.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred fetching subtitles: {e}")
        return None
