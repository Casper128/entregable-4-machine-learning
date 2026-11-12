"""Descarga el CSV público indicado en la guía desde Kaggle."""
from pathlib import Path
from urllib.request import urlopen, Request
from zipfile import ZipFile
from io import BytesIO
import hashlib

URL = "https://www.kaggle.com/api/v1/datasets/download/shivamb/real-or-fake-fake-jobposting-prediction"
destination = Path(__file__).resolve().parent / "data" / "fake_job_postings.csv"
if destination.exists():
    print("Se conserva el archivo existente:", destination)
else:
    request = Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=120) as response:
        archive_bytes = response.read()
    with ZipFile(BytesIO(archive_bytes)) as archive:
        matches = [name for name in archive.namelist() if Path(name).name == "fake_job_postings.csv"]
        if len(matches) != 1:
            raise ValueError("El archivo descargado no contiene el CSV esperado. Descárgalo manualmente de Kaggle.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive.read(matches[0]))
    print("Dataset guardado:", destination)
print("SHA-256:", hashlib.sha256(destination.read_bytes()).hexdigest())
