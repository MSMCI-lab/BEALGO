#!/usr/bin/env python3
"""
Pre-publishing verification script for BEALGO.
Run this before pushing to GitHub to ensure everything is ready.
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"  ✓ {description}")
        return True
    else:
        print(f"  ✗ {description} - MISSING")
        return False

def check_file_content(filepath, search_strings, description):
    """Check if file contains certain strings."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            for search_str in search_strings:
                if search_str in content:
                    print(f"  ⚠ {description}: '{search_str}' found - needs updating")
                    return False
        print(f"  ✓ {description}")
        return True
    except Exception as e:
        print(f"  ✗ {description}: Error - {e}")
        return False

def run_tests():
    """Run test suite."""
    print("\n🧪 Running Tests...")
    result = os.system("python3 test_bealgo.py > /dev/null 2>&1")
    if result == 0:
        print("  ✓ All tests passed")
        return True
    else:
        print("  ✗ Some tests failed - run 'python3 test_bealgo.py' for details")
        return False

def check_git_status():
    """Check git status."""
    print("\n📦 Checking Git Status...")
    
    # Check if git is initialized
    if not os.path.exists('.git'):
        print("  ✗ Git not initialized")
        return False
    print("  ✓ Git repository initialized")
    
    # Check for uncommitted changes
    result = os.popen('git status --porcelain').read().strip()
    if result:
        print("  ⚠ Uncommitted changes found:")
        print(result)
        return False
    else:
        print("  ✓ All changes committed")
        return True

def main():
    print("=" * 60)
    print("BEALGO Pre-Publishing Verification")
    print("=" * 60)
    
    all_checks_passed = True
    
    # Check required files
    print("\n📄 Checking Required Files...")
    required_files = [
        ('be_joint_pmf.py', 'Main algorithm file'),
        ('README.md', 'README documentation'),
        ('LICENSE', 'License file'),
        ('setup.py', 'Package setup file'),
        ('.gitignore', 'Git ignore file'),
        ('test_bealgo.py', 'Test suite'),
        ('example3_4.py', 'Example script'),
    ]
    
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_checks_passed = False
    
    # Check for placeholders that need updating
    print("\n🔍 Checking for Placeholders...")
    placeholders = [
        ('setup.py', ['yourusername', 'Your Name', 'your.email@example.com'], 
         'setup.py has placeholders'),
        ('LICENSE', ['[Your Name/Institution]'], 
         'LICENSE has placeholders'),
    ]
    
    for filepath, search_strs, description in placeholders:
        if os.path.exists(filepath):
            if not check_file_content(filepath, search_strs, description):
                print(f"     Note: Update {filepath} with your actual information")
    
    # Run tests
    if not run_tests():
        all_checks_passed = False
    
    # Check git status
    if not check_git_status():
        all_checks_passed = False
    
    # Check documentation files
    print("\n📚 Checking Documentation Files...")
    doc_files = [
        ('USAGE.md', 'Usage guide'),
        ('EXAMPLES.md', 'Examples documentation'),
        ('QUICK_REFERENCE.md', 'Quick reference'),
        ('PUBLISHING_GUIDE.md', 'Publishing guide'),
    ]
    
    for filepath, description in doc_files:
        check_file_exists(filepath, description)
    
    # Final summary
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("✅ All critical checks passed!")
        print("=" * 60)
        print("\n📋 Next Steps:")
        print("1. Update author information in setup.py and LICENSE")
        print("2. Create repository on GitHub")
        print("3. Run these commands:")
        print("   git remote add origin https://github.com/yourusername/BEALGO.git")
        print("   git push -u origin main")
        print("\nSee PUBLISHING_GUIDE.md for detailed instructions.")
    else:
        print("⚠ Some checks failed - please review above")
        print("=" * 60)
        return 1
    
    return 0

if __name__ == "__main__":
    # Change to script directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    sys.exit(main())
