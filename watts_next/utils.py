from pathlib import Path

import requests


def download_file_from_url(url: str, output_path: Path) -> None:
    """Download file from url and store at desired path."""
    # download the file from this url to a tmp location @ tmp_path
    response = requests.get(url=url, stream=True, timeout=20)
    response.raise_for_status()
    with open(output_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
