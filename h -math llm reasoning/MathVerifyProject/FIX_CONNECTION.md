# Fix: "This Site Can't Be Reached"

## Quick Fixes

### Fix 1: Try Different URL

The server might be running but you're using the wrong URL. Try ALL of these:

1. `http://localhost:7860`
2. `http://127.0.0.1:7860`
3. `http://0.0.0.0:7860`

### Fix 2: Check if Server is Actually Running

Look at your terminal. Do you see:
```
Running on local URL:  http://127.0.0.1:7860
```

**If NO** - The server didn't start. Check for error messages.

**If YES** - The server is running, but browser can't connect.

### Fix 3: Windows Firewall

Windows Firewall might be blocking Python:

1. Open **Windows Defender Firewall**
2. Click **Allow an app through firewall**
3. Find **Python** and check both **Private** and **Public**
4. Or temporarily disable firewall to test

### Fix 4: Try Different Port

Port 7860 might be blocked. Try port 8080:

```python
# In simple_launch.py, change:
iface.launch(server_port=8080)
```

Then access: `http://localhost:8080`

### Fix 5: Check What's Running

Run this to see if port 7860 is in use:
```bash
netstat -an | findstr 7860
```

If you see `LISTENING`, the server is running.

### Fix 6: Use Auto-Open Browser

Run this version that tries to open browser automatically:

```bash
python launch_localhost.py
```

### Fix 7: Use CLI Instead

If web interface won't work, use command line (always works):

```bash
python main.py --mode cli verify --gold "1/2" --pred "0.5"
```

## Still Not Working?

1. **Check terminal output** - Is there an error message?
2. **Try different browser** - Chrome, Firefox, Edge
3. **Check antivirus** - Temporarily disable to test
4. **Try different port** - Use 8080 or 5000

## Alternative: Use CLI

The CLI interface works without any web server:

```bash
python main.py --mode cli verify --gold "1/2" --pred "0.5"
```

This will show results directly in terminal with colored output.

