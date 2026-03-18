# How to Access the MathVerify Interface

## 🌐 Web Interface (Gradio)

### Quick Start

**Method 1: Using main.py**
```bash
python main.py --mode gradio
```

**Method 2: Using demo script**
```bash
python demo_gradio.py
```

### Access URLs

After launching, the interface will be available at:

- **Local Access**: http://localhost:7860
- **Network Access**: http://0.0.0.0:7860 (if configured)
- **Public Link**: Available if `share=True` is set

### Features Available

Once the interface loads, you'll see tabs for:

1. **✅ Answer Verification**
   - Enter gold and prediction answers
   - See verification results with LaTeX rendering
   - View error taxonomy breakdown

2. **📝 Handwritten Math OCR**
   - Upload InkML files
   - Upload images (PNG, JPG)
   - Transcribe to LaTeX

3. **📊 Batch Verification**
   - Upload files with multiple answers
   - See batch results

4. **📊 Error Taxonomy**
   - Reference guide for error types

5. **ℹ️ About**
   - Project information

## 💻 Command Line Interface (CLI)

If you prefer command-line:

```bash
# Single verification
python main.py --mode cli verify --gold "1/2" --pred "0.5"

# Batch verification
python main.py --mode cli batch-verify --gold-file gold.txt --pred-file pred.txt
```

## 🐍 Python API

For programmatic access:

```python
from core_verification import MathVerifier

verifier = MathVerifier()
result = verifier.verify_answer(gold="1/2", prediction="0.5")
print(result)
```

## 🔧 Troubleshooting

### Gradio Not Installed

If you see an error about Gradio:

```bash
pip install gradio
```

### Port Already in Use

If port 7860 is busy, use a different port:

```bash
python main.py --mode gradio --port 7861
```

### Interface Won't Open

1. Check if the server started:
   - Look for "Running on local URL: http://127.0.0.1:7860"
   
2. Manually open browser:
   - Copy the URL from terminal
   - Paste into browser

3. Check firewall settings:
   - Ensure port 7860 is not blocked

## 📸 Screenshots

The interface includes:
- Clean, modern design
- LaTeX rendering for math expressions
- Color-coded error taxonomy
- Step-by-step reasoning display
- Image upload support

## 🚀 Quick Test

To quickly test if everything works:

```bash
# 1. Start the interface
python main.py --mode gradio

# 2. Open browser to http://localhost:7860

# 3. Try an example:
#    Gold: 1/2
#    Prediction: 0.5
#    Click "Verify Answer"

# 4. You should see: ✓ CORRECT
```

## 📝 Notes

- The interface runs until you press `Ctrl+C`
- All data is processed locally
- No internet required (except for optional API features)
- Works on Windows, Mac, and Linux

---

**Ready to use!** Launch the interface and start verifying mathematical expressions.

