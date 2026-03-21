# 🚀 QUICK START GUIDE - MVM²

## ⚡ Fastest Way to Get Started

### Step 1: Open Terminal in Project Directory
```bash
cd c:\Users\Varshith Dharmaj\Downloads\major\math_verification_mvp
```

### Step 2: Run the Startup Script
```powershell
.\start.ps1
```

Choose option **2** for quick demo (Dashboard Only)

---

## 📋 What You'll See

1. **Dashboard opens at:** http://localhost:8501
2. **Two input modes:**
   - 📝 **Text Input** - Try the pre-filled example
   - 📷 **Image Upload** - Upload a handwritten math problem

3. **Click "Verify Solution"** to see results

---

## 🧪 Testing the System

### Quick Test (No Services Required)
The dashboard will work in demo mode even without microservices running.

### Full Test (All Services)
```powershell
.\start.ps1
```
Choose option **1** - This opens 4 windows:
- OCR Service (Port 8001)
- SymPy Service (Port 8002)
- LLM Service (Port 8003)
- Dashboard (Port 8501)

---

## 🎯 Try These Examples

### Example 1: Valid Solution ✅
**Problem:** "Janet has 3 apples. She buys 2 more. She gives 1 away."

**Steps:**
```
Janet starts with 3 apples
She buys 2 more: 3 + 2 = 5 apples
She gives 1 away: 5 - 1 = 4 apples
```

**Expected:** VALID with high confidence

---

### Example 2: Error Detection ❌
**Problem:** "There are 5 boxes with 8 apples each."

**Steps:**
```
Number of boxes = 5
Apples per box = 8
Total = 5 × 8 = 45
```

**Expected:** ERROR detected (5 × 8 = 40, not 45)

---

## 🔧 Prerequisites

### Required (Basic Demo)
- ✅ Python 3.10+
- ✅ Virtual environment (./start.ps1 creates this automatically)

### Optional (Full Features)
- Tesseract OCR (for image processing)
- Gemini API Key (for LLM reasoning)

---

## 📦 Installing Additional Components

### Tesseract OCR (for Image Mode)
1. Download: https://github.com/tesseract-ocr/tesseract
2. Install and add to PATH
3. Restart terminal

### Gemini API Key (for LLM Features)
1. Get free key: https://ai.google.dev/
2. Copy `.env.template` to `.env`
3. Add: `GEMINI_API_KEY=your_key_here`

---

## 🐛 Troubleshooting

### "Module not found"
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### "Port already in use"
Close any applications using ports 8001-8003, 8501

### Services not connecting
- Check if all service windows are still open
- Look for error messages in service windows
- Restart the startup script

---

## 📊 What to Expect

### Performance Metrics
- ⏱️ Processing time: 1-5 seconds per problem
- 🎯 Accuracy: 68%+ on valid test cases
- 🔍 Error detection: 78%+ when errors present

### Features Working
- ✅ Text input verification
- ✅ Multi-model consensus
- ✅ Error detection and reporting
- ✅ Confidence scoring
- ✅ Agreement analysis

### Image Input (Requires Tesseract)
- 📷 Handwritten math problems
- 📄 Printed worksheets
- 🖼️ Whiteboard photos

---

## 🎓 Research Features Demonstrated

1. **Multimodal Input** - Accept both text and images
2. **Weighted Consensus** - Symbolic (40%), LLM (35%), ML (25%)
3. **OCR-Aware Calibration** - Novel uncertainty propagation
4. **Real-time Processing** - <5 second response time

---

## 📞 Next Steps

1. ✅ **Test basic functionality** - Run the text examples
2. ⚡ **Try image upload** - If you have Tesseract installed
3. 🧪 **Run automated tests** - `python tests/test_system.py`
4. 📊 **Collect data** - Test with your own math problems
5. 🎨 **Customize** - Modify weights, add more patterns

---

## 🆘 Need Help?

Check the full README.md for:
- Detailed architecture
- API documentation
- Advanced configuration
- Deployment options

---

**MVM²** - Making Mathematical Verification Multimodal  
VNR VJIET Major Project 2025
