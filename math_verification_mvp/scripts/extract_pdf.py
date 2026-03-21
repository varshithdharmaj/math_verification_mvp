import PyPDF2
import sys

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.extract_text()
        return text

if __name__ == "__main__":
    pdf_file = r"c:\Users\Varshith Dharmaj\Downloads\major\MVM2-COMPLETE-MVP-MULTIMODAL.pdf"
    extracted_text = extract_text_from_pdf(pdf_file)
    
    # Write to file with UTF-8 encoding
    output_file = r"c:\Users\Varshith Dharmaj\Downloads\major\pdf_content.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(extracted_text)
    
    print(f"PDF content extracted to: {output_file}")
