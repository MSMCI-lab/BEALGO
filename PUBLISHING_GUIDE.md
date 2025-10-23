# Publishing BEALGO to GitHub - Step by Step Guide

This guide will walk you through publishing your BE algorithm to GitHub.

## Prerequisites

1. **GitHub Account**: Make sure you have a GitHub account at https://github.com
2. **Git Configured**: Your git is already initialized in this directory

## Step 1: Configure Git (if not already done)

Run these commands to set your identity:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

Or for this repository only:
```bash
cd "/Users/mao/Library/CloudStorage/OneDrive-ILStateUniversity/collaborations/Gaofeng Da/BEALGO"
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

## Step 2: Create a New Repository on GitHub

1. Go to https://github.com and log in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Fill in the details:
   - **Repository name**: `BEALGO` (or your preferred name)
   - **Description**: "Backward Elimination Algorithm for Joint PMF Computation in Network Propagation"
   - **Visibility**: Choose "Public" (so people can use it) or "Private"
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click "Create repository"

## Step 3: Connect Your Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these:

```bash
cd "/Users/mao/Library/CloudStorage/OneDrive-ILStateUniversity/collaborations/Gaofeng Da/BEALGO"

# Add the remote repository (replace 'yourusername' with your GitHub username)
git remote add origin https://github.com/yourusername/BEALGO.git

# Push your code to GitHub
git branch -M main
git push -u origin main
```

## Step 4: Update Package Files with Your GitHub URL

After pushing, update these files with your actual GitHub URL:

1. **setup.py**: Change line 13:
   ```python
   url="https://github.com/yourusername/BEALGO",
   ```

2. **setup.py**: Update author information (lines 8-9):
   ```python
   author="Your Name",
   author_email="your.email@example.com",
   ```

3. **LICENSE**: Update copyright line with your name/institution

Then commit and push the changes:
```bash
git add setup.py LICENSE
git commit -m "Update author information and GitHub URL"
git push
```

## Step 5: Verify Your Repository

1. Go to `https://github.com/yourusername/BEALGO`
2. You should see:
   - Your README.md displayed on the main page
   - All your files listed
   - The commit history

## Step 6: Test Installation

Test that others can install your package:

```bash
# In a different directory or virtual environment
pip install git+https://github.com/yourusername/BEALGO.git

# Test the import
python -c "from be_joint_pmf import joint_pmf_by_types; print('Success!')"
```

## Step 7: Share with Others

Now people can use your algorithm in three ways:

### Method 1: Direct Installation (Recommended)
```bash
pip install git+https://github.com/yourusername/BEALGO.git
```

### Method 2: Clone and Install
```bash
git clone https://github.com/yourusername/BEALGO.git
cd BEALGO
pip install -e .
```

### Method 3: Download Single File
```bash
curl -O https://raw.githubusercontent.com/yourusername/BEALGO/main/be_joint_pmf.py
```

## Optional: Create a Release

To make your code more official:

1. Go to your GitHub repository
2. Click "Releases" on the right side
3. Click "Create a new release"
4. Fill in:
   - **Tag version**: `v0.1.0`
   - **Release title**: `v0.1.0 - Initial Release`
   - **Description**: Brief description of features
5. Click "Publish release"

## Optional: Add a DOI with Zenodo

To make your code citable in academic papers:

1. Go to https://zenodo.org
2. Log in with your GitHub account
3. Go to GitHub settings in Zenodo
4. Enable the BEALGO repository
5. Create a new release on GitHub
6. Zenodo will automatically create a DOI
7. Add the DOI badge to your README.md

## File Structure Summary

Your repository now includes:

```
BEALGO/
├── .gitignore              # Files to ignore in git
├── LICENSE                 # MIT License
├── README.md              # Main documentation
├── USAGE.md               # Detailed usage guide
├── EXAMPLES.md            # Example usage patterns
├── MANIFEST.in            # Package manifest
├── requirements.txt       # Dependencies (none needed)
├── setup.py               # Installation configuration
├── be_joint_pmf.py        # Main algorithm
├── example3_4.py          # Example script
├── depth_compare.py       # Analysis script
├── table7.py              # Analysis script
├── table8.py              # Analysis script
├── insurace_pricing_example.py  # Application example
├── pricing_vs_L.csv       # Data file
└── table7_scada_by_depth.csv    # Data file
```

## Maintenance Tips

### Making Updates

When you make changes:
```bash
git add <changed-files>
git commit -m "Description of changes"
git push
```

### Checking Repository Status
```bash
git status
git log --oneline
```

### Creating Branches for Features
```bash
git checkout -b feature-name
# Make changes
git add .
git commit -m "Add feature"
git push -u origin feature-name
# Then create a pull request on GitHub
```

## Troubleshooting

### Authentication Issues

If you have issues pushing to GitHub:

**Option 1: Use Personal Access Token (Recommended)**
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate a new token with "repo" scope
3. Use the token as your password when pushing

**Option 2: Use SSH**
1. Generate SSH key: `ssh-keygen -t ed25519 -C "your.email@example.com"`
2. Add to SSH agent: `ssh-add ~/.ssh/id_ed25519`
3. Add public key to GitHub Settings → SSH keys
4. Change remote URL: `git remote set-url origin git@github.com:yourusername/BEALGO.git`

### OneDrive Sync Issues

Since your repository is in OneDrive, you might want to:
1. Pause OneDrive sync while working with git
2. Or clone a separate copy outside OneDrive for active development

## Next Steps

1. **Add Examples**: Include more example scripts
2. **Add Tests**: Create unit tests for your functions
3. **Documentation**: Consider using Sphinx for API docs
4. **CI/CD**: Set up GitHub Actions for automated testing
5. **PyPI**: Publish to PyPI for `pip install bealgo` (without git+)

## Questions?

If you encounter any issues:
1. Check GitHub's documentation: https://docs.github.com
2. Stack Overflow: https://stackoverflow.com/questions/tagged/git
3. GitHub Community: https://github.community/

---

**Current Status**: ✅ Repository initialized and ready to push to GitHub
**Next Action**: Create GitHub repository and push code (see Step 2 above)
