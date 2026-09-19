
import PyPDF2
import sys

pdf_path = "DiffMark-main/diffmark_paper_v2.pdf"
output_path = "exact_pdf_text.txt"

print(f"Extracting EXACT text from {pdf_path}...")

with open(pdf_path, 'rb') as f:
    reader = PyPDF2.PdfReader(f)
    full_text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        full_text += f"\n--- PAGE {page_num} ---\n"
        full_text += page_text

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(full_text)

print(f"✅ EXACT text saved to {output_path}!")
print("\n--- First 2000 characters ---")
print(full_text[:2000])
