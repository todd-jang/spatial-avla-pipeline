import subprocess
from pathlib import Path

RAW = Path("raw_videos")
RAW.mkdir(exist_ok=True)
urls = {
    "zth2QuoHscY": "https://www.youtube.com/watch?v=zth2QuoHscY",
    "UpleTeyFPJ0": "https://www.youtube.com/watch?v=UpleTeyFPJ0",
}
for vid, url in urls.items():
    out = RAW / f"{vid}.mp4"
    if out.exists():
        print("SKIP", out)
        continue
    cmd = [
        "yt-dlp",
        "-f",
        "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "--merge-output-format",
        "mp4",
        "-o",
        str(out),
        url,
    ]
    subprocess.run(cmd, check=True)
print("DONE")
