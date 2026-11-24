"""
pdf_extractor.py
----------------
Extract text from an uploaded PDF using pypdf.
pypdf is imported inside the function so that the rest of the app
keeps working even if the library has a problem.
"""

from .utils import FriendlyError


def extract_text_from_pdf(uploaded_file):
    """Return all text found in a PDF (Streamlit UploadedFile or file object)."""
    name = getattr(uploaded_file, "name", "")
    if name and not name.lower().endswith(".pdf"):
        raise FriendlyError("Unsupported file type. Please upload a .pdf file.")

    try:
        from pypdf import PdfReader
    except ImportError:
        raise FriendlyError(
            "PDF support is not available (pypdf is not installed). "
            "Run: pip install pypdf  - or paste the article text instead."
        )

    try:
        reader = PdfReader(uploaded_file)

        if reader.is_encrypted:
            try:
                unlocked = reader.decrypt("")
            except Exception:
                unlocked = 0
            if not unlocked:
                raise FriendlyError(
                    "This PDF is password protected. Please upload an unlocked PDF."
                )

        pages_text = []
        for page in reader.pages:
            try:
                pages_text.append(page.extract_text() or "")
            except Exception:
                continue                       # skip a damaged page

    except FriendlyError:
        raise
    except Exception:
        raise FriendlyError(
            "This file is not a valid PDF or it is damaged. "
            "Please upload another PDF or paste the text instead."
        )

    text = "\n\n".join(pages_text).strip()
    if not text:
        raise FriendlyError(
            "No text could be extracted from this PDF. It may be a scanned image. "
            "Please paste the article text instead."
        )
    return text
