from pathlib import Path
import fitz 
import docx
import re

def clean_text(text: str) -> str:
    text = re.sub(r'\n+', '\n', text) 
    text = re.sub(r'[ \t]+', ' ', text)  
    text = text.strip()
    return text

def read_pdf(file_path: str) -> str:
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"{file_path} does not exist.")
    
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    
    text = clean_text(text)
    return text

def read_word(file_path: str) -> str:
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"{file_path} does not exist.")
    
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    text = clean_text(text)
    return text

def read_text_file(file_path: str) -> str:
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"{file_path} does not exist.")
    
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    text = clean_text(text)
    return text

def read_file(file_path: str) -> str:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"{file_path} does not exist.")
    
    ext = path.suffix.lower()
    if ext == ".pdf":
        return read_pdf(file_path)
    elif ext in [".docx", ".doc"]:
        return read_word(file_path)
    elif ext == ".txt":
        return read_text_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
