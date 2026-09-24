"""
NEXUS Document Chunker
Intelligent boundary-aware chunking respecting headings, paragraphs, and sentence boundaries.
Maintains strict provenance (document_id, chunk_id, section, page, token_count).
"""

import re
from typing import List, Dict, Any, Optional


class DocumentChunker:
    """
    Intelligent document chunking engine preserving semantic continuity,
    heading structure, and document provenance.
    """

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        """
        :param chunk_size: Approximate token/word count per chunk.
        :param chunk_overlap: Approximate token/word overlap between sequential chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, processed_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split processed document sections into structured, traceable chunks.
        """
        document_id = processed_doc.get("document_id", "doc_unknown")
        project_id = processed_doc.get("project_id")
        filename = processed_doc.get("filename", "unknown")
        sections = processed_doc.get("sections", [])

        all_chunks: List[Dict[str, Any]] = []
        global_chunk_idx = 0

        for sec in sections:
            section_title = sec.get("title", "")
            page_number = sec.get("page", 1)
            section_text = sec.get("text", "")

            if not section_text.strip():
                continue

            sec_chunks = self._split_section_text(
                text=section_text,
                section_title=section_title,
                page_number=page_number,
                document_id=document_id,
                project_id=project_id,
                filename=filename,
                start_index=global_chunk_idx
            )

            all_chunks.extend(sec_chunks)
            global_chunk_idx += len(sec_chunks)

        return all_chunks

    def _split_section_text(
        self,
        text: str,
        section_title: str,
        page_number: int,
        document_id: str,
        project_id: Optional[str],
        filename: str,
        start_index: int
    ) -> List[Dict[str, Any]]:
        """Split section into sentence-aware chunks respecting size bounds."""
        # Split text into paragraphs first
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]

        sentences: List[str] = []
        for p in paragraphs:
            # Split paragraph into sentences
            raw_sents = re.split(r"(?<=[.?!])\s+", p)
            for s in raw_sents:
                if s.strip():
                    sentences.append(s.strip())

        chunks: List[Dict[str, Any]] = []
        current_words: List[str] = []
        current_chunk_idx = start_index

        for sent in sentences:
            sent_words = sent.split()
            if len(current_words) + len(sent_words) > self.chunk_size and len(current_words) > 0:
                chunk_str = " ".join(current_words)
                chunk_id = f"{document_id}_chk_{current_chunk_idx}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "project_id": project_id,
                    "filename": filename,
                    "chunk_index": current_chunk_idx,
                    "section_title": section_title,
                    "page_number": page_number,
                    "text": chunk_str,
                    "token_count": len(current_words),
                    "metadata": {
                        "section": section_title,
                        "page": page_number,
                        "filename": filename
                    }
                })
                current_chunk_idx += 1

                # Carry over overlap
                if self.chunk_overlap > 0 and len(current_words) > self.chunk_overlap:
                    current_words = current_words[-self.chunk_overlap:]
                else:
                    current_words = []

            current_words.extend(sent_words)

        if current_words:
            chunk_str = " ".join(current_words)
            chunk_id = f"{document_id}_chk_{current_chunk_idx}"
            chunks.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "project_id": project_id,
                "filename": filename,
                "chunk_index": current_chunk_idx,
                "section_title": section_title,
                "page_number": page_number,
                "text": chunk_str,
                "token_count": len(current_words),
                "metadata": {
                    "section": section_title,
                    "page": page_number,
                    "filename": filename
                }
            })

        return chunks
