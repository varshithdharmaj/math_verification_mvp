# Enhanced CLI Interface - Features

## 🚀 Quick Start

### Basic Verification:
```bash
python main.py --mode cli verify --gold "1/2" --pred "0.5"
```

### Batch Verification:
```bash
python main.py --mode cli batch-verify --gold-file gold.txt --pred-file pred.txt
```

### Transcription:
```bash
python main.py --mode cli transcribe --inkml file.inkml --model model.pth
```

## ✨ Features

### 1. **Colored Output (Rich Library)**
- ✅ Green for correct answers
- ❌ Red for incorrect answers  
- 🟡 Yellow for warnings/details
- 🔵 Blue for information
- Professional panels and tables

### 2. **Detailed Verification Display**
- Status panel with colored border
- Input display (gold and prediction)
- Parsing details (how each was parsed)
- Error classification (if incorrect)
- Structured information table

### 3. **Progress Bars**
- Batch processing shows progress
- Uses tqdm or rich progress indicators
- Real-time status updates

### 4. **Error Classification**
- Parse errors (can't parse input)
- Format mismatches (equivalent but different formats)
- Value errors (incorrect answers)
- Detailed error messages

### 5. **Batch Results Table**
- Color-coded results table
- Summary statistics
- Error type for each incorrect answer
- Easy-to-read format

## 📊 Output Format

### Single Verification:
- Colored panel with status
- Information table with details
- Error classification if incorrect

### Batch Verification:
- Summary panel with statistics
- Results table with all verifications
- Color-coded correct/incorrect status

## 🔧 Installation

Install rich for colored output:
```bash
pip install rich
```

The CLI will work without rich, but with basic formatting.

## 📝 Examples

See the enhanced CLI in action:
```bash
python main.py --mode cli verify --gold "1/2" --pred "0.5"
```

