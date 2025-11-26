# Lease & Invoice Analyzer

A Streamlit application that analyzes lease documents and invoices to extract key terms, identify violations, and generate comprehensive PDF reports.

## Features

- **Exact Lease Wording Extraction**: Extracts verbatim text from lease documents with clause references
- **Invoice Analysis**: Parses invoice line items and categorizes charges
- **Violation Detection**: Identifies charges that violate lease terms with specific clause references
- **Clean PDF Reports**: Generates professional PDF reports with detailed analysis
- **Clause References**: Links violations to specific lease clauses and sections

## Requirements

- Python 3.8 or higher
- OpenAI API key (for AI-powered extraction)

## Installation

1. **Clone or download this repository**

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get an OpenAI API key:**
   - Visit https://platform.openai.com/api-keys
   - Create a new API key
   - Copy the key (you'll need it when running the app)

## Running the Application

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser:**
   - The app will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, navigate to that URL in your browser

3. **Configure your API key:**
   - Enter your OpenAI API key in the sidebar
   - Click "✓" to confirm it's configured

## Usage

### Step 1: Upload Documents

1. Go to the **"📤 Upload Documents"** tab
2. **Upload Lease Document**: Click "Browse files" and select your lease PDF
   - Click "Extract Lease Data" to process the document
   - Review the extracted CAM rules, taxes, utilities, escalation caps, allowed/disallowed fees
3. **Upload Invoice Document**: Click "Browse files" and select your invoice PDF
   - Click "Extract Invoice Data" to process the document
   - Review the extracted line items

### Step 2: Review Analysis Report

1. Go to the **"📄 Report"** tab
2. View the **Executive Summary** with key metrics
3. Review **Flagged Violations** with detailed explanations and clause references
4. Check **Relevant Lease Terms** showing exact wording from your lease
5. Review **Invoice Line Items** and **Allowed Charges**

### Step 3: Download PDF Report

1. Click the **"Generate PDF Report"** button
2. Once generated, click **"📄 Download PDF Report"** to save the report

## Project Structure

```
Gino/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── Sample_Lease (1).pdf  # Sample lease document (for testing)
└── Sample_Invoice (1).pdf # Sample invoice document (for testing)
```

## Dependencies

- `streamlit` - Web app framework
- `PyPDF2` - PDF processing
- `pdfplumber` - Advanced PDF text extraction
- `reportlab` - PDF report generation
- `pandas` - Data manipulation and display
- `openai` - AI-powered text extraction

## Notes

- The app uses OpenAI's GPT-4 model for extracting structured data from documents
- All extracted text is cleaned and smoothed for better readability
- Clause references are automatically identified when available
- The app maintains exact wording from lease documents without paraphrasing

## Troubleshooting

**Issue: "OpenAI extraction failed"**
- Check that your API key is correct and has available credits
- The app will fall back to regex-based extraction if OpenAI fails

**Issue: "Could not extract text from PDF"**
- Ensure the PDF is not password-protected
- Try a different PDF file to verify the PDF is readable

**Issue: Module not found errors**
- Run `pip install -r requirements.txt` again
- Make sure you're using the correct Python environment

## Support

For issues or questions, please check that:
1. All dependencies are installed correctly
2. Your OpenAI API key is valid and has credits
3. Your PDF files are not corrupted or password-protected
