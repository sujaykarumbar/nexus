import hashlib
import os
import re
import shutil
import uuid
from typing import Tuple, BinaryIO
from apps.api.core.config import settings
from apps.api.core.exceptions import ValidationException


def calculate_sha256(file_path: str) -> str:
    """Calculate SHA-256 checksum of a file on disk."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def sanitize_filename(filename: str) -> str:
    """Strip unsafe characters, path traversals, and normalize name."""
    filename = os.path.basename(filename)
    filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
    return filename[:200]


def save_secure_upload(file_obj: BinaryIO, original_filename: str) -> Tuple[str, str, int, str]:
    """
    Securely store uploaded dataset file.
    Returns: (stored_file_path, sanitized_filename, file_size_bytes, sha256_checksum)
    """
    clean_name = sanitize_filename(original_filename)
    ext = clean_name.split(".")[-1].lower() if "." in clean_name else ""
    
    allowed_exts = ["csv", "xlsx", "xls", "json", "parquet"]
    if ext not in allowed_exts:
        raise ValidationException(f"Unsupported file format '.{ext}'. Allowed: {', '.join(allowed_exts)}")

    file_uuid = str(uuid.uuid4())
    stored_name = f"{file_uuid}_{clean_name}"
    target_path = os.path.join(settings.UPLOAD_DIR, stored_name)

    # Save to disk
    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file_obj, buffer)

    file_size_bytes = os.path.getsize(target_path)
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if file_size_bytes > max_bytes:
        if os.path.exists(target_path):
            os.remove(target_path)
        raise ValidationException(f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB}MB.")

    checksum = calculate_sha256(target_path)
    return target_path, clean_name, file_size_bytes, checksum
