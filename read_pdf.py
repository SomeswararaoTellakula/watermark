from PyPDF2 import PdfReader

pdf_path = "/Users/apple/Documents/watermark/Diffmark_Final_Paper.pdf"
reader = PdfReader(pdf_path)
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

with open("/Users/apple/Documents/watermark/paper_text.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Paper text extracted to paper_text.txt")
print(text[:10000])
