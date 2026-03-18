r# Firewall Fix for Web Interface

## ⚠️ Important: Don't Disable Firewall Entirely!

Instead of turning off the firewall, **allow Python through it**. This is safer!

## ✅ Safe Method: Allow Python Through Firewall

### Step 1: Open Windows Firewall
1. Press `Win + R`
2. Type: `firewall.cpl`
3. Press Enter

### Step 2: Allow Python
1. Click **"Allow an app or feature through Windows Defender Firewall"** (on left)
2. Click **"Change settings"** (top right)
3. Click **"Allow another app..."** (bottom right)
4. Click **"Browse..."**
5. Navigate to: `C:\Users\Varshith Dharmaj\AppData\Local\Programs\Python\Python313\`
6. Select `python.exe`
7. Click **"Add"**
8. Check both **"Private"** and **"Public"** boxes
9. Click **"OK"**

## 🧪 Test Method: Temporarily Disable (For Testing Only)

**Only for testing!** Turn it back on after:

### Which Profile to Disable?

- **Private Network**: Use this if you're on a home/work network
- **Public Network**: Use this if you're on public WiFi
- **Domain Network**: Usually for work computers (don't disable this)

**Recommendation**: Disable **Private** only for testing.

### Steps:
1. Open Windows Firewall (`Win + R`, type `firewall.cpl`)
2. Click **"Turn Windows Defender Firewall on or off"** (left side)
3. For **Private network settings**, select **"Turn off Windows Defender Firewall"**
4. Click **OK**
5. Try: `python simple_launch.py`
6. **IMPORTANT**: Turn it back on after testing!

## 🎯 Better Solution: Use CLI Instead

The CLI works without any firewall changes:

```bash
python main.py --mode cli verify --gold "1/2" --pred "0.5"
```

This shows results in terminal - no browser needed!

## 📝 Quick Test

After allowing Python through firewall:

1. Run: `python simple_launch.py`
2. Look for: `Running on local URL: http://127.0.0.1:7860`
3. Open browser: `http://127.0.0.1:7860`

If it still doesn't work, the issue might be something else (browser, port, etc.).

