# SemiAutoFixer

An intelligent tool for cleaning and processing Chinese text with Vietnamese phonetic transcriptions. This tool combines manual text processing with AI-powered language models to automatically correct and align Chinese characters with their Vietnamese phonetic equivalents.

## Features

- **Automatic Text Cleaning**: Normalizes Chinese text and Vietnamese phonetic transcriptions
- **AI-Powered Correction**: Integrates with multiple LLM strategies (Gemini, GPT, TongGu)
- **Manual Correction Interface**: GUI for interactive text correction
- **Batch Processing**: Handles large datasets efficiently
- **Multiple Input Formats**: Supports both text files and Excel spreadsheets
- **Resume Capability**: Can continue processing from where it left off
- **Comprehensive Logging**: Tracks all corrections and manual interventions

## Quick Start

### 1. Automatic Setup (Recommended)

```bash
# Clone or download the SemiAutoFixer tool
cd SemiAutoFixer

# Run the automated setup
python setup.py

# Follow the interactive usage guide
python how_to_use.py
```

### 2. Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create working directories
mkdir -p WORKING_FOLDER/MyProject

# Place your input files in WORKING_FOLDER/MyProject/
# Run the processor
python runner.py
```

## Requirements

- **Python**: 3.8 or higher
- **Dependencies**: Listed in `requirements.txt`
  - pandas, torch, transformers, underthesea
  - selenium, google-generativeai (for LLM features)
  - tqdm, tabulate, openpyxl

## Input Formats

### Option 1: Separate Text Files

Place these files in your project directory (e.g., `WORKING_FOLDER/MyProject/`):

- `original_chinese.txt`: Original Chinese text (one line per entry)
- `original_sinoviet.txt`: Vietnamese phonetic transcription (space-separated)
- `meaning.txt`: Modern Vietnamese meanings (optional)
- `page-index.txt`: Page numbers (optional)

**Example:**
```
# original_chinese.txt
劍有餘靈光若水
文從大塊壽如山

# original_sinoviet.txt  
Kiếm hữu dư linh quang nhược thủy
Văn tòng đại khối thọ như sơn
```

### Option 2: Excel File

Create an Excel file with columns:
- `Chinese`: Chinese text
- `Sino`: Vietnamese phonetic transcription  
- `Modern`: Modern Vietnamese meaning

## Configuration

Edit `runner.py` to configure your processing:

```python
# Set your project directory
DIR_PATH = 'WORKING_FOLDER/MyProject/'

# Use Excel input (optional)
EXCEL_PATH = 'WORKING_FOLDER/data.xlsx'  # or leave empty for text files

# Enable initial validation
INITIAL_CHECK = True  # recommended for first run

# Configure LLM strategy (optional)
autofixer.llm_fixer.strategy = GeminiSeleniumStrategy()
```

## Usage Examples

### Basic Processing
```python
from utils import AutoFixer, Data

# Initialize the fixer
autofixer = AutoFixer()

# Process a single entry
data = Data("劍有餘靈", "Kiếm hữu dư linh", "Kiếm sót khí thiêng")
cleaned_data = autofixer.fix(data)

print(f"Cleaned: {''.join(cleaned_data.chinese)}")
```

### Batch Processing
```python
# Run the complete pipeline
python runner.py
```

### With AI Enhancement
```python
from utils.llm_fixer import GeminiStrategy

autofixer = AutoFixer()
autofixer.llm_fixer.strategy = GeminiStrategy()
# API key configuration required
```

## Output Files

After processing, the following files are generated:

- `chinese.txt`: Cleaned Chinese text
- `phonetic.txt`: Cleaned Vietnamese phonetic transcription  
- `log.txt`: Processing statistics and manual fix count
- `error.txt`: Validation errors (if any)

## Advanced Features

### LLM Integration
- **Gemini AI**: Google's generative AI for text correction
- **GPT Integration**: OpenAI GPT for intelligent text processing
- **TongGu Model**: Specialized classical Chinese language model

### Manual Correction Interface
- Interactive GUI for manual text correction
- Batch editing capabilities
- Real-time preview of changes

### Data Validation
- Pre-processing validation to identify potential issues
- Alignment checking between Chinese and Vietnamese text
- Comprehensive error reporting

## Directory Structure

```
SemiAutoFixer/
├── requirements.txt          # Package dependencies
├── setup.py                 # Automated setup script  
├── how_to_use.py            # Interactive usage guide
├── runner.py                # Main processing script
├── utils/                   # Core processing modules
│   ├── autofixer.py        # Main AutoFixer class
│   ├── llm_fixer.py        # LLM integration strategies
│   ├── manual_fixer.py     # Manual correction GUI
│   ├── normalize_text.py   # Text normalization utilities
│   └── data/               # Reference data files
├── WORKING_FOLDER/          # Project workspace
│   ├── MyProject/          # Your project files
│   │   ├── original_chinese.txt
│   │   ├── original_sinoviet.txt
│   │   └── ...
│   └── ...
```

## Getting Help

1. **Interactive Guide**: Run `python how_to_use.py` for step-by-step instructions
2. **Setup Issues**: Run `python setup.py` to verify your installation
3. **Configuration**: Check the configuration examples in `how_to_use.py`
4. **File Formats**: See the file format guide in the usage script

## Post-Processing

After processing a dataset:

1. **Backup Results**: Copy output files (`chinese.txt`, `phonetic.txt`, `log.txt`) to a safe location
2. **Review Changes**: Check the log file for processing statistics
3. **Quality Check**: Manually review a sample of the cleaned text
4. **Prepare Next Dataset**: Replace input files for the next project

## Troubleshooting

- **Import Errors**: Run `python setup.py` to verify installation
- **File Not Found**: Ensure input files are in the correct directory
- **Encoding Issues**: All files should be UTF-8 encoded
- **LLM Errors**: Check API key configuration and internet connection