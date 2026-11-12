
"""Descarga y valida el dataset de ofertas laborales falsas de Kaggle."""

from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from zipfile import ZipFile, BadZipFile
from io import BytesIO, StringIO
import csv
import hashlib
import os
import tempfile

URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "shivamb/real-or-fake-fake-jobposting-prediction"
)

BASE_DIR = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd()
)

DESTINATION = BASE_DIR / "data" / "fake_job_postings.csv"

REQUIRED_COLUMNS = {
    "job_id",
    "title",
    "description",
    "requirements",
    "fraudulent",
}

MAX_DOWNLOAD_BYTES = 150 * 1024 * 1024
MAX_CSV_BYTES = 150 * 1024 * 1024


def calculate_sha256(path):
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def validate_csv(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("El CSV no existe o está vacío.")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns

        if missing:
            raise ValueError(
                f"Faltan columnas obligatorias: {sorted(missing)}"
            )

        rows = 0
        labels = set()

        for row in reader:
            rows += 1
            labels.add(row["fraudulent"])

        if rows == 0:
            raise ValueError("El CSV no contiene registros.")

        if not labels.issubset({"0", "1"}):
            raise ValueError(
                f"Valores inesperados en fraudulent: {labels}"
            )

    print(f"Registros encontrados: {rows:,}")
    print(f"Columnas encontradas: {len(columns)}")
    print(f"Etiquetas encontradas: {sorted(labels)}")


def download_dataset():
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)

    if DESTINATION.exists():
        print("Archivo existente:", DESTINATION)
        validate_csv(DESTINATION)
        print("SHA-256:", calculate_sha256(DESTINATION))
        return

    print("Descargando dataset desde Kaggle...")

    request = Request(
        URL,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    try:
        with urlopen(request, timeout=120) as response:
            archive_bytes = response.read(MAX_DOWNLOAD_BYTES + 1)

        if len(archive_bytes) > MAX_DOWNLOAD_BYTES:
            raise ValueError("Descarga superior al límite permitido.")

        with ZipFile(BytesIO(archive_bytes)) as archive:
            matches = [
                item for item in archive.infolist()
                if not item.is_dir()
                and Path(item.filename).name == "fake_job_postings.csv"
            ]

            if len(matches) != 1:
                raise ValueError(
                    "El ZIP no contiene exactamente un CSV esperado."
                )

            item = matches[0]

            if item.file_size > MAX_CSV_BYTES:
                raise ValueError("CSV superior al límite permitido.")

            # El temporal se crea en el mismo directorio.
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=DESTINATION.parent,
                suffix=".tmp",
                delete=False
            ) as temp_file:
                temp_path = Path(temp_file.name)

                try:
                    with archive.open(item) as source:
                        while True:
                            block = source.read(1024 * 1024)
                            if not block:
                                break
                            temp_file.write(block)
                except Exception:
                    temp_path.unlink(missing_ok=True)
                    raise

        try:
            validate_csv(temp_path)
            os.replace(temp_path, DESTINATION)
        finally:
            temp_path.unlink(missing_ok=True)

        print("Dataset guardado:", DESTINATION)
        print("SHA-256:", calculate_sha256(DESTINATION))

    except HTTPError as error:
        print(f"Error HTTP de Kaggle: {error.code}")
        raise

    except URLError as error:
        print(f"Error de conexión: {error.reason}")
        raise

    except BadZipFile:
        print("El archivo descargado no es un ZIP válido.")
        raise


if __name__ == "__main__":
    download_dataset()
