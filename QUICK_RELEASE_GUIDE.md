# ⚡ VeZ Quick Release Guide

**Fast reference for releasing VeZ**

---

## 🚀 Quick Setup (First Time Only)

### 1. Update Repository References (5 minutes)

Replace `YOUR_USERNAME` with your GitHub username:

```bash
# Automated replacement
sed -i 's/YOUR_USERNAME/your-github-username/g' install.sh
sed -i 's/YOUR_USERNAME/your-github-username/g' install.ps1
sed -i 's/YOUR_USERNAME/your-github-username/g' docs/index.html
sed -i 's/YOUR_USERNAME/your-github-username/g' INSTALLATION.md
```

### 2. Enable GitHub Actions Permissions

1. Go to: `Settings` → `Actions` → `General`
2. Select: **Read and write permissions**
3. Save

### 3. Enable GitHub Pages (Optional)

1. Go to: `Settings` → `Pages`
2. Source: `main` branch → `/docs` folder
3. Save

**Done! You're ready to release.**

---

## 📦 Creating a Release

### Standard Release

```bash
# 1. Update version in Cargo.toml
vim compiler/Cargo.toml  # Change version = "1.0.0"

# 2. Commit
git add .
git commit -m "Release v1.0.0"

# 3. Tag and push
git tag v1.0.0
git push origin main
git push origin v1.0.0

# 4. Wait ~10 minutes for builds to complete
# 5. Check: https://github.com/YOUR_USERNAME/VeZ/releases
```

### Test Release

```bash
git tag v0.1.0-test
git push origin v0.1.0-test
```

---

## ✅ Verification Checklist

After release completes:

- [ ] 5 binary files uploaded to release
- [ ] Test Linux install: `curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/VeZ/main/install.sh | bash`
- [ ] Test Windows install: `iwr -useb https://raw.githubusercontent.com/YOUR_USERNAME/VeZ/main/install.ps1 | iex`
- [ ] Test macOS install (if available)
- [ ] Verify `vezc --version` works
- [ ] Check website download buttons work

---

## 📥 Installation Commands

**Linux/macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/VeZ/main/install.sh | bash
```

**Windows:**
```powershell
iwr -useb https://raw.githubusercontent.com/YOUR_USERNAME/VeZ/main/install.ps1 | iex
```

---

## 🔧 Troubleshooting

**Build fails?**
- Check: `Actions` tab → View logs
- Common: Missing dependencies, compilation errors

**Release not created?**
- Verify tag pushed: `git ls-remote --tags origin`
- Check workflow permissions enabled

**Downloads don't work?**
- Verify release exists
- Check file names match pattern
- Update `YOUR_USERNAME` in scripts

---

## 📚 Full Documentation

- **User Guide:** `INSTALLATION.md`
- **Setup Guide:** `RELEASE_SETUP.md`
- **Complete Report:** `RELEASE_IMPLEMENTATION_COMPLETE.md`

---

## 🎯 That's It!

**Three commands to release:**
```bash
git tag v1.0.0
git push origin main
git push origin v1.0.0
```

**One command for users to install:**
```bash
curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/VeZ/main/install.sh | bash
```

**Simple. Automated. Professional.** 🚀
