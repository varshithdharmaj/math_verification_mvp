# Troubleshooting: Can't Access Interface

## Quick Fixes

### 1. Check if Server is Running

When you run `python launch_with_debug.py`, you should see:
```
Running on local URL:  http://127.0.0.1:7860
```

**If you DON'T see this line**, the server didn't start. Check for errors above.

### 2. Try Different URLs

If `http://localhost:7860` doesn't work, try:
- `http://127.0.0.1:7860`
- `http://0.0.0.0:7860`

### 3. Check Port Availability

Port 7860 might be in use. Try a different port:

```python
# In launch_with_debug.py, change:
app.launch(server_port=7861)  # Use 7861 instead
```

Then access: `http://localhost:7861`

### 4. Check Firewall

Windows Firewall might be blocking the connection:
1. Open Windows Defender Firewall
2. Allow Python through firewall
3. Or temporarily disable firewall to test

### 5. Check Browser

- Try a different browser (Chrome, Firefox, Edge)
- Clear browser cache
- Try incognito/private mode

### 6. Manual Launch Steps

1. **Open terminal in project directory**
2. **Run:**
   ```bash
   python launch_with_debug.py
   ```
3. **Wait for:** "Running on local URL: http://127.0.0.1:7860"
4. **Copy the URL** from terminal
5. **Paste into browser** (don't just type localhost)

### 7. Check for Errors

Look for error messages in the terminal. Common issues:

**Error: "Address already in use"**
- Port 7860 is busy
- Solution: Use different port or close other programs

**Error: "Module not found"**
- Missing dependencies
- Solution: `pip install gradio`

**Error: Import errors**
- Python path issues
- Solution: Make sure you're in the project directory

### 8. Alternative: Use CLI Instead

If web interface won't work, use CLI:

```bash
python main.py --mode cli verify --gold "1/2" --pred "0.5"
```

### 9. Test with Simple Gradio App

Test if Gradio works at all:

```bash
python test_gradio_simple.py
```

If this simple test works, the issue is with our app, not Gradio.

### 10. Check Network Settings

- Make sure you're not using a VPN that blocks localhost
- Check proxy settings
- Try disabling antivirus temporarily

## Still Not Working?

1. **Share the terminal output** - Copy all text from when you run the launch command
2. **Check Python version** - Run `python --version`
3. **Check if port is listening** - Run `netstat -an | findstr 7860` (Windows)

## Quick Test Commands

```bash
# Test 1: Can you import Gradio?
python -c "import gradio; print('OK')"

# Test 2: Can you create our app?
python -c "from main_interface.gradio_app import create_gradio_app; print('OK')"

# Test 3: Launch with debug
python launch_with_debug.py
```

