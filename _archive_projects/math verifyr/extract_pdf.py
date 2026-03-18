
import sys
import importlib.util

def check_install(package_name):
    spec = importlib.util.find_spec(package_name)
    return spec is not None

pdf_path = r"c:\Users\Varshith Dharmaj\Downloads\math verifyr\MAJOR_PROJECT_PAPER_I (5).pdf"

try:
    if check_install("pypdf"):
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        print("SUCCESS: pypdf")
        print(text)
    elif check_install("PyPDF2"):
        import PyPDF2
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        print("SUCCESS: PyPDF2")
        print(text)
    else:
        print("FAILURE: No PDF library found (pypdf or PyPDF2).")
        # Fallback to simple string extraction if possible (not reliable for PDF but worth a try?)
        # unique strings
        pass

except Exception as e:
    print(f"ERROR: {e}")

