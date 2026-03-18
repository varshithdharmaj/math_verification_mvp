# Enhanced Gradio Web Interface - Features

## 🚀 Launch the Interface

### Quick Start:
```bash
# Option 1: Using main.py
python main.py --mode gradio

# Option 2: Using standalone demo
python demo_gradio.py
```

The interface will open at: `http://localhost:7860`

## ✨ Features

### 1. **Clean Web Interface**
- Modern, professional design
- Responsive layout
- Easy-to-use tabs
- Color-coded results

### 2. **Answer Verification Tab**
- **Input Fields:**
  - Gold/Expected Answer (supports LaTeX, plain math, numbers)
  - Model Prediction
  
- **Output Display:**
  - ✅/❌ Visual verification result
  - Step-by-step parsing details
  - Error classification (if incorrect)
  - Color-coded status indicators
  
- **Example Presets:**
  - Click examples to try them instantly
  - Pre-configured test cases

### 3. **Handwritten Math OCR Tab**
- **InkML File Upload:**
  - Upload `.inkml` files
  - Automatic model detection
  - Custom model path option
  - LaTeX output display
  
- **Image Upload:**
  - Upload images of handwritten math
  - Future: Direct image-to-LaTeX conversion
  - Status messages and error handling

### 4. **Batch Verification Tab**
- Upload two files:
  - Gold answers file (one per line)
  - Predictions file (one per line)
- Results displayed in table:
  - Gold answer
  - Prediction
  - Verification result (✓/✗)
  - Error classification
- Summary statistics

### 5. **About Tab**
- Project information
- Feature list
- Usage tips
- Documentation links

## 🎨 UI Features

### Visual Elements:
- **Color-coded Results:**
  - Green for correct answers
  - Red for incorrect answers
  - Yellow for warnings/info
  - Blue for parsing details

- **HTML Formatting:**
  - Rich text display
  - Code blocks for LaTeX
  - Structured sections
  - Professional styling

### Error Classification:
- Parse errors (can't parse input)
- Format mismatches (equivalent but different formats)
- Value errors (incorrect answers)
- Detailed error messages

## 📝 Usage Examples

### Example 1: Simple Verification
1. Go to "Answer Verification" tab
2. Enter Gold: `1/2`
3. Enter Prediction: `0.5`
4. Click "🔍 Verify Answer"
5. See detailed result with parsing info

### Example 2: LaTeX Verification
1. Enter Gold: `$\frac{1}{2}$`
2. Enter Prediction: `0.5`
3. See step-by-step parsing and verification

### Example 3: OCR Transcription
1. Go to "Handwritten Math OCR" tab
2. Upload an InkML file
3. Click "📄 Transcribe InkML"
4. View transcribed LaTeX output

### Example 4: Batch Processing
1. Go to "Batch Verification" tab
2. Upload gold_answers.txt (one per line)
3. Upload predictions.txt (one per line)
4. Click "📊 Process Batch"
5. View results table with all verifications

## 🔧 Configuration

### Server Settings:
```python
app.launch(
    server_name="0.0.0.0",  # Allow external access
    server_port=7860,       # Port number
    share=False,            # Set True for public link
    show_error=True         # Show detailed errors
)
```

### Customization:
- Modify `gradio_app.py` to customize:
  - Colors and styling
  - Layout and tabs
  - Additional features
  - Error messages

## 📊 Output Format

### Verification Result Structure:
```python
{
    'valid': True/False,
    'gold': "input gold answer",
    'prediction': "input prediction",
    'gold_parsed': "parsed representation",
    'pred_parsed': "parsed representation",
    'error_type': "error classification or None",
    'details': "detailed explanation"
}
```

### HTML Output:
- Structured sections
- Color-coded status
- Code blocks for LaTeX
- Error highlights
- Parsing details

## 🚨 Troubleshooting

### Issue: Gradio not installed
**Solution:**
```bash
pip install gradio
```

### Issue: Port already in use
**Solution:**
Change port in `demo_gradio.py`:
```python
app.launch(server_port=7861)  # Use different port
```

### Issue: Model not found (OCR)
**Solution:**
- Specify model path manually
- Or ensure default model exists at:
  `handwritten-math-transcription/model/model_best_0.pth`

## 🎯 Best Practices

1. **For Best Results:**
   - Use LaTeX format: `$\frac{1}{2}$`
   - Plain expressions work: `1/2`, `0.5`
   - Be consistent with formats

2. **For Presentations:**
   - Use `share=True` for public link
   - Test examples beforehand
   - Have backup examples ready

3. **For Batch Processing:**
   - Ensure files have same number of lines
   - One answer per line
   - No empty lines

## 📚 Related Files

- `main_interface/gradio_app.py` - Main Gradio interface code
- `demo_gradio.py` - Standalone demo script
- `main.py` - Main entry point (supports --mode gradio)
- `README.md` - Complete documentation

---

**Ready to use!** Launch with `python demo_gradio.py` or `python main.py --mode gradio`

