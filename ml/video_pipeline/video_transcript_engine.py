import yt_dlp
import os

# =========================
# OUTPUT DIRECTORY
# =========================

OUTPUT_DIR = "downloads"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# =========================
# YOUTUBE URL
# =========================

video_url = input(
    "Enter YouTube Video URL: "
)

# =========================
# YT-DLP OPTIONS
# =========================

ydl_opts = {

    "skip_download": True,

    "writesubtitles": True,

    "writeautomaticsub": True,

    "subtitleslangs": ["en"],

    "subtitlesformat": "vtt",

    "outtmpl":
    f"{OUTPUT_DIR}/%(title)s.%(ext)s"

}

# =========================
# DOWNLOAD TRANSCRIPT
# =========================

print("\nExtracting transcript...\n")

with yt_dlp.YoutubeDL(ydl_opts) as ydl:

    info = ydl.extract_info(
        video_url,
        download=True
    )

    video_title = info["title"]

# =========================
# SUCCESS
# =========================

print("\nTranscript extraction completed!")

print(f"\nVideo Title: {video_title}")

print(
    f"\nSaved inside: {OUTPUT_DIR}"
)