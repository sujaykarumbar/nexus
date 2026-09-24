"""
NEXUS Document Processor
Multi-format extraction pipeline for PDF, DOCX, TXT, Markdown, CSV, JSON, and HTML.
Preserves structural headings, sections, page numbers, metadata, and SHA-256 provenance hashes.
"""

import os
import re
import json
import hashlib
import zipfile
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from typing import Dict, Any, List, Optional


class SimpleHTMLTextExtractor(HTMLParser):
    """Clean text extractor from HTML ignoring script/style tags."""

    def __init__(self):
        super().__init__()
        self.reset()
        self.fed: List[str] = []
        self.ignore = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() in ["script", "style"]:
            self.ignore = True

    def handle_endtag(self, tag):
        if tag.lower() in ["script", "style"]:
            self.ignore = False
        elif tag.lower() in ["p", "div", "h1", "h2", "h3", "h4", "li", "tr"]:
            self.fed.append("\n")

    def handle_data(self, data):
        if not self.ignore and data.strip():
            self.fed.append(data.strip() + " ")

    def get_text(self) -> str:
        return "".join(self.fed).strip()


class DocumentProcessor:
    """
    Modular document extraction pipeline extracting text, headings, sections,
    and metadata while preserving provenance.
    """

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".json", ".html", ".pdf", ".docx"}

    @staticmethod
    def compute_hash(content: bytes) -> str:
        """Compute SHA256 hash of raw file content."""
        return hashlib.sha256(content).hexdigest()

    def process_file(
        self,
        file_path: str,
        document_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a document file from path and extract structured sections and text.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        filename = os.path.basename(file_path)
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        file_size = os.path.getsize(file_path)

        with open(file_path, "rb") as f:
            raw_bytes = f.read()

        doc_hash = self.compute_hash(raw_bytes)

        return self.process_bytes(
            raw_bytes=raw_bytes,
            filename=filename,
            file_type=ext.lstrip("."),
            file_size=file_size,
            doc_hash=doc_hash,
            document_id=document_id,
            project_id=project_id
        )

    def process_bytes(
        self,
        raw_bytes: bytes,
        filename: str,
        file_type: str,
        file_size: int,
        doc_hash: Optional[str] = None,
        document_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process raw bytes and extract structured metadata and sections.
        """
        if not doc_hash:
            doc_hash = self.compute_hash(raw_bytes)

        normalized_type = file_type.lower().lstrip(".")
        sections: List[Dict[str, Any]] = []
        page_count = 1
        metadata: Dict[str, Any] = {
            "filename": filename,
            "file_type": normalized_type,
            "file_size": file_size,
            "hash": doc_hash
        }

        if normalized_type in ["txt", "text"]:
            text = raw_bytes.decode("utf-8", errors="replace")
            sections = self._parse_plaintext_sections(text)

        elif normalized_type in ["md", "markdown"]:
            text = raw_bytes.decode("utf-8", errors="replace")
            sections = self._parse_markdown_sections(text)

        elif normalized_type == "csv":
            text = raw_bytes.decode("utf-8", errors="replace")
            sections = self._parse_csv_summary(text, filename)

        elif normalized_type == "json":
            text = raw_bytes.decode("utf-8", errors="replace")
            sections = self._parse_json_summary(text, filename)

        elif normalized_type == "html":
            text = raw_bytes.decode("utf-8", errors="replace")
            sections = self._parse_html_sections(text)

        elif normalized_type == "docx":
            sections, page_count = self._parse_docx(raw_bytes)

        elif normalized_type == "pdf":
            sections, page_count = self._parse_pdf(raw_bytes)

        else:
            # Fallback text decoder
            text = raw_bytes.decode("utf-8", errors="replace")
            sections = [{"title": "Content", "page": 1, "text": text}]

        # Compute full aggregated text
        full_text = "\n\n".join([f"## {s['title']}\n{s['text']}" if s['title'] else s['text'] for s in sections])

        return {
            "document_id": document_id or f"doc_{doc_hash[:12]}",
            "project_id": project_id,
            "filename": filename,
            "file_type": normalized_type,
            "file_size": file_size,
            "hash": doc_hash,
            "page_count": max(1, page_count),
            "sections": sections,
            "full_text": full_text.strip(),
            "metadata": metadata
        }

    def _parse_markdown_sections(self, text: str) -> List[Dict[str, Any]]:
        """Parse markdown into sections based on # and ## headings."""
        lines = text.splitlines()
        sections: List[Dict[str, Any]] = []
        current_title = "Introduction"
        current_lines: List[str] = []
        page = 1

        for line in lines:
            # Check for heading
            heading_match = re.match(r"^(#{1,3})\s+(.+)$", line)
            if heading_match:
                if current_lines:
                    sections.append({
                        "title": current_title,
                        "page": page,
                        "text": "\n".join(current_lines).strip()
                    })
                    current_lines = []
                current_title = heading_match.group(2).strip()
            else:
                current_lines.append(line)

        if current_lines:
            sections.append({
                "title": current_title,
                "page": page,
                "text": "\n".join(current_lines).strip()
            })

        if not sections:
            sections = [{"title": "General", "page": 1, "text": text.strip()}]

        return sections

    def _parse_plaintext_sections(self, text: str) -> List[Dict[str, Any]]:
        """Split plaintext by major paragraph breaks or page markers."""
        paras = re.split(r"\n\s*\n\s*\n", text)
        sections = []
        for idx, para in enumerate(paras):
            if para.strip():
                sections.append({
                    "title": f"Section {idx + 1}",
                    "page": (idx // 3) + 1,
                    "text": para.strip()
                })
        return sections if sections else [{"title": "Document", "page": 1, "text": text.strip()}]

    def _parse_html_sections(self, text: str) -> List[Dict[str, Any]]:
        """Extract clean text from HTML."""
        parser = SimpleHTMLTextExtractor()
        parser.feed(text)
        cleaned = parser.get_text()
        return self._parse_plaintext_sections(cleaned)

    def _parse_csv_summary(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """Extract tabular sample and schema summary from CSV text."""
        lines = text.strip().splitlines()
        header = lines[0] if lines else ""
        sample_rows = lines[1:6] if len(lines) > 1 else []
        total_rows = len(lines) - 1

        summary_text = f"CSV Dataset: {filename}\nColumns: {header}\nTotal Rows: {total_rows}\n"
        if sample_rows:
            summary_text += "Sample Records:\n" + "\n".join(sample_rows)

        return [{
            "title": f"Tabular Overview: {filename}",
            "page": 1,
            "text": summary_text
        }]

    def _parse_json_summary(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """Parse and summarize JSON documents."""
        try:
            data = json.loads(text)
            if isinstance(data, list):
                summary = f"JSON Document Array: {filename} with {len(data)} records.\nSample: {json.dumps(data[:2], indent=2)}"
            elif isinstance(data, dict):
                keys = list(data.keys())
                summary = f"JSON Document Object: {filename} with keys: {keys}.\nContent: {json.dumps(data, indent=2)}"
            else:
                summary = f"JSON Value: {data}"
        except Exception:
            summary = text

        return [{
            "title": f"JSON Data: {filename}",
            "page": 1,
            "text": summary
        }]

    def _parse_docx(self, raw_bytes: bytes) -> tuple[List[Dict[str, Any]], int]:
        """Extract text from docx via XML document.xml."""
        try:
            import io
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
                xml_content = z.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            # Namespace for Word
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            paragraphs = []
            for p in tree.iterfind(".//w:p", ns):
                texts = [node.text for node in p.iterfind(".//w:t", ns) if node.text]
                if texts:
                    paragraphs.append("".join(texts))
            full_text = "\n\n".join(paragraphs)
            return self._parse_plaintext_sections(full_text), max(1, len(paragraphs) // 10)
        except Exception:
            # Fallback
            text = raw_bytes.decode("utf-8", errors="ignore")
            # Extract readable ascii chunks
            readable = "".join([c if ord(c) < 128 else " " for c in text])
            return [{"title": "DOCX Document", "page": 1, "text": readable.strip()}], 1

    def _parse_pdf(self, raw_bytes: bytes) -> tuple[List[Dict[str, Any]], int]:
        """Extract text from PDF using pypdf if available, else standard text stream extraction."""
        try:
            import pypdf
            import io
            reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
            page_count = len(reader.pages)
            sections = []
            for idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    sections.append({
                        "title": f"Page {idx + 1}",
                        "page": idx + 1,
                        "text": page_text.strip()
                    })
            if sections:
                return sections, page_count
        except ImportError:
            pass
        except Exception:
            pass

        # Fallback text stream parser for PDF
        text = raw_bytes.decode("latin1", errors="ignore")
        # Extract text within parentheses in stream objects
        matches = re.findall(r"\((.*?)\)\s*Tj", text)
        if matches:
            extracted = " ".join(matches)
            return [{"title": "PDF Document", "page": 1, "text": extracted}], 1

        # Generic fallback
        clean_text = re.sub(r"[^\x20-\x7E\n]", " ", text)
        return [{"title": "PDF Extract", "page": 1, "text": clean_text[:5000].strip()}], 1
