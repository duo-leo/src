#!/usr/bin/env python3
"""
SemiAutoFixer Usage Script

This script demonstrates how to use the SemiAutoFixer tool for cleaning Chinese text
with Vietnamese phonetic transcriptions. It provides examples for different input formats
and configuration options.
"""

import os
import sys
from pathlib import Path

def print_banner():
    """Print the SemiAutoFixer banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                        SemiAutoFixer                         ║
    ║            Chinese Text Cleaning and Processing Tool         ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'pandas', 'tqdm', 'torch', 'transformers', 'underthesea',
        'selenium', 'google.generativeai', 'regex', 'tabulate'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install missing packages using:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed!")
    return True

def setup_working_directory():
    """Set up the working directory structure"""
    working_dir = Path("WORKING_FOLDER")
    
    # Create main working directory
    working_dir.mkdir(exist_ok=True)
    print(f"📁 Working directory created: {working_dir.absolute()}")
    
    # Create example subdirectories
    example_dirs = [
        "DiSanHanNomHue",
        "CauDoiWiki", 
        "DoiLien",
        "NewProject"
    ]
    
    for dir_name in example_dirs:
        (working_dir / dir_name).mkdir(exist_ok=True)
    
    print("📁 Example project directories created:")
    for dir_name in example_dirs:
        print(f"   - WORKING_FOLDER/{dir_name}/")

def create_sample_files():
    """Create sample input files for demonstration"""
    sample_dir = Path("WORKING_FOLDER/NewProject")
    
    # Sample Chinese text
    chinese_text = """劍有餘靈光若水
文從大塊壽如山
東西南北由斯道
公卿夫士出此途"""
    
    # Sample Vietnamese phonetic transcription
    phonetic_text = """Kiếm hữu dư linh quang nhược thủy
Văn tòng đại khối thọ như sơn
Đông tây nam bắc do tư đạo
Công khanh phu sĩ xuất thử đồ"""
    
    # Sample meanings
    meaning_text = """Kiếm sót khí thiêng ngời tựa nước
Văn cùng trời đất thọ như non
Đông, tây, nam, bắc đều theo đạo ấy
Công, khanh, phu, sĩ đều ra từ con đường này"""
    
    # Sample page indices
    page_index_text = """1
1
2
2"""
    
    # Write sample files
    with open(sample_dir / "original_chinese.txt", "w", encoding="utf-8") as f:
        f.write(chinese_text)
    
    with open(sample_dir / "original_sinoviet.txt", "w", encoding="utf-8") as f:
        f.write(phonetic_text)
    
    with open(sample_dir / "meaning.txt", "w", encoding="utf-8") as f:
        f.write(meaning_text)
    
    with open(sample_dir / "page-index.txt", "w", encoding="utf-8") as f:
        f.write(page_index_text)
    
    print("📝 Sample input files created in WORKING_FOLDER/NewProject/:")
    print("   - original_chinese.txt")
    print("   - original_sinoviet.txt") 
    print("   - meaning.txt")
    print("   - page-index.txt")

def show_usage_examples():
    """Show usage examples"""
    examples = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                         Usage Examples                       ║
    ╚══════════════════════════════════════════════════════════════╝
    
    1. Basic Usage with Text Files:
    ─────────────────────────────────────────────────────────────
    
    from utils import AutoFixer, Data
    from tqdm import trange
    
    # Set up the data directory
    DIR_PATH = 'WORKING_FOLDER/NewProject/'
    
    # Initialize the AutoFixer
    autofixer = AutoFixer()
    
    # Load data from text files
    with open(f'{DIR_PATH}original_chinese.txt', 'r', encoding='utf-8') as f:
        chinese = f.readlines()
    
    with open(f'{DIR_PATH}original_sinoviet.txt', 'r', encoding='utf-8') as f:
        phonetic = f.readlines()
        
    # Process each line
    data = []
    for i in range(len(chinese)):
        data.append(Data(chinese[i], phonetic[i], ''))
        data[i] = autofixer.fix(data[i])
    
    
    2. Using Excel Input:
    ─────────────────────────────────────────────────────────────
    
    import pandas as pd
    from utils import AutoFixer, Data
    
    # Load data from Excel file
    df = pd.read_excel('WORKING_FOLDER/your_data.xlsx')
    df.columns = ['Chinese', 'Sino', 'Modern']
    
    autofixer = AutoFixer()
    
    for i, row in df.iterrows():
        data_item = Data(row['Chinese'], row['Sino'], row['Modern'])
        cleaned_data = autofixer.fix(data_item)
        print(f"Original: {row['Chinese']}")
        print(f"Cleaned:  {''.join(cleaned_data.chinese)}")
    
    
    3. Using LLM Strategies:
    ─────────────────────────────────────────────────────────────
    
    from utils import AutoFixer
    from utils.llm_fixer import GeminiStrategy, GeminiSeleniumStrategy
    
    # Initialize with Gemini LLM strategy
    autofixer = AutoFixer()
    autofixer.llm_fixer.strategy = GeminiStrategy()
    
    # Or use Selenium-based Gemini strategy
    # autofixer.llm_fixer.strategy = GeminiSeleniumStrategy()
    
    
    4. Running the Complete Pipeline:
    ─────────────────────────────────────────────────────────────
    
    # Simply run the main script
    python runner.py
    
    # Make sure to set the correct DIR_PATH in runner.py:
    # DIR_PATH = 'WORKING_FOLDER/YourProjectName/'
    
    
    5. Initial Data Validation:
    ─────────────────────────────────────────────────────────────
    
    # Set INITIAL_CHECK = True in runner.py to validate data first
    # This will check for errors before processing
    
    
    6. Output Files:
    ─────────────────────────────────────────────────────────────
    
    After processing, you'll find these files in your working directory:
    - chinese.txt      : Cleaned Chinese text
    - phonetic.txt     : Cleaned phonetic transcription
    - log.txt          : Processing log with manual fix count
    - error.txt        : Any errors found during validation (if applicable)
    """
    
    print(examples)

def show_file_format_guide():
    """Show the expected file formats"""
    guide = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                      File Format Guide                       ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Input File Formats:
    ─────────────────────────────────────────────────────────────
    
    1. original_chinese.txt:
       - One line per Chinese text entry
       - UTF-8 encoding
       - Example:
         劍有餘靈光若水
         文從大塊壽如山
    
    2. original_sinoviet.txt:
       - One line per Vietnamese phonetic transcription
       - Space-separated words/syllables
       - UTF-8 encoding
       - Example:
         Kiếm hữu dư linh quang nhược thủy
         Văn tòng đại khối thọ như sơn
    
    3. meaning.txt (optional):
       - One line per meaning/translation
       - UTF-8 encoding
       - Example:
         Kiếm sót khí thiêng ngời tựa nước
         Văn cùng trời đất thọ như non
    
    4. page-index.txt (optional):
       - One line per page number
       - Integer values
       - Example:
         1
         1
         2
         2
    
    Excel Format:
    ─────────────────────────────────────────────────────────────
    
    Columns should be named:
    - Chinese  : Chinese text
    - Sino     : Vietnamese phonetic transcription  
    - Modern   : Modern Vietnamese meaning
    
    The tool will automatically detect and use the appropriate input format.
    """
    
    print(guide)

def show_configuration_options():
    """Show configuration options in runner.py"""
    config = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    Configuration Options                     ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Key Configuration Variables in runner.py:
    ─────────────────────────────────────────────────────────────
    
    1. DIR_PATH:
       - Set to your project directory path
       - Example: 'WORKING_FOLDER/MyProject/'
    
    2. EXCEL_PATH:
       - Set to Excel file path if using Excel input
       - Leave empty '' to use text files
       - Example: 'WORKING_FOLDER/data.xlsx'
    
    3. INITIAL_CHECK:
       - Set to True to validate data before processing
       - Set to False to skip validation and start processing
       - Recommended: True for first run
    
    4. LLM Strategy Selection:
       - Uncomment to use Gemini Selenium strategy:
         autofixer.llm_fixer.strategy = GeminiSeleniumStrategy()
       - Default uses manual fixing with some AI assistance
    
    Advanced Options:
    ─────────────────────────────────────────────────────────────
    
    - Manual fixing UI: Available for interactive correction
    - Batch processing: Handles large datasets automatically
    - Resume capability: Can continue from where it left off
    - Error logging: Comprehensive error tracking and reporting
    """
    
    print(config)

def main():
    """Main function to run the usage guide"""
    print_banner()
    
    print("🔍 Checking requirements...")
    if not check_requirements():
        return
    
    print("\n📁 Setting up directories...")
    setup_working_directory()
    
    print("\n📝 Creating sample files...")
    create_sample_files()
    
    # Interactive menu
    while True:
        menu = """
        ╔══════════════════════════════════════════════════════════════╗
        ║                         Main Menu                            ║
        ╚══════════════════════════════════════════════════════════════╝
        
        1. Show Usage Examples
        2. Show File Format Guide  
        3. Show Configuration Options
        4. Run Sample Processing
        5. Exit
        
        Choose an option (1-5): """
        
        choice = input(menu).strip()
        
        if choice == '1':
            show_usage_examples()
        elif choice == '2':
            show_file_format_guide()
        elif choice == '3':
            show_configuration_options()
        elif choice == '4':
            print("\n🚀 To run sample processing:")
            print("1. Edit runner.py:")
            print("   DIR_PATH = 'WORKING_FOLDER/NewProject/'")
            print("   INITIAL_CHECK = True  # for first run")
            print("2. Run: python runner.py")
            print("3. Check output files in WORKING_FOLDER/NewProject/")
        elif choice == '5':
            print("\n👋 Thank you for using SemiAutoFixer!")
            break
        else:
            print("❌ Invalid choice. Please select 1-5.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
