import streamlit as st
import PyPDF2
import pdfplumber
import pandas as pd
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
from openai import OpenAI

# Page configuration
st.set_page_config(
    page_title="Lease & Invoice Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Lease & Invoice Analyzer")
st.markdown("Upload lease documents and invoices to extract key fields and identify mismatches.")

# Sidebar for API configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key. You can get one from https://platform.openai.com/api-keys",
        value=st.session_state.get('openai_api_key', '')
    )
    if api_key:
        st.session_state.openai_api_key = api_key
        st.success("✓ API key configured")
    else:
        st.warning("⚠️ Please enter your OpenAI API key to use AI-powered extraction")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This app uses OpenAI's API to extract structured data from lease documents and invoices.")

# Initialize session state
if 'lease_data' not in st.session_state:
    st.session_state.lease_data = None
if 'invoice_data' not in st.session_state:
    st.session_state.invoice_data = None
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None


def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file using pdfplumber for better accuracy."""
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return ""


def smooth_extracted_text(text: str) -> str:
    """Clean and smooth extracted text to make it more natural."""
    if not text:
        return text
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common PDF extraction issues
    # Join words broken across lines (e.g., "mainte- nance" -> "maintenance")
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'([.,;:!?])\s*([A-Z])', r'\1 \2', text)
    
    # Fix spacing around quotes
    text = re.sub(r'\s+"', '"', text)
    text = re.sub(r'"\s+', '" ', text)
    
    # Remove orphaned single characters at line starts (common PDF artifact)
    text = re.sub(r'^\s*[a-zA-Z]\s+', '', text, flags=re.MULTILINE)
    
    # Fix multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    
    # Trim and clean
    text = text.strip()
    
    return text


def extract_lease_fields(text: str, api_key: str = None) -> Dict:
    """Extract key fields from lease document using OpenAI API."""
    if not api_key:
        # Fallback to regex if no API key
        return extract_lease_fields_regex(text)
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Truncate text if too long (OpenAI has token limits)
        max_chars = 12000  # Leave room for prompt
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[... text truncated ...]"
        
        lease_schema = {
            "type": "object",
            "properties": {
                "cam_rules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "exact_wording": {"type": "string", "description": "The exact quoted text from the lease document"},
                            "clause_reference": {"type": "string", "description": "Clause number, section, or paragraph reference (e.g., 'Section 5.2', 'Clause 12', 'Paragraph 3.1')"}
                        },
                        "required": ["exact_wording", "clause_reference"],
                        "additionalProperties": False
                    },
                    "description": "CAM rules with exact wording and clause references"
                },
                "taxes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "exact_wording": {"type": "string", "description": "The exact quoted text from the lease document"},
                            "clause_reference": {"type": "string", "description": "Clause number, section, or paragraph reference"}
                        },
                        "required": ["type", "exact_wording", "clause_reference"],
                        "additionalProperties": False
                    },
                    "description": "Tax-related information with exact wording and clause references"
                },
                "utilities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "exact_wording": {"type": "string", "description": "The exact quoted text from the lease document"},
                            "clause_reference": {"type": "string", "description": "Clause number, section, or paragraph reference"}
                        },
                        "required": ["type", "exact_wording", "clause_reference"],
                        "additionalProperties": False
                    },
                    "description": "Utility-related information with exact wording and clause references"
                },
                "escalation_caps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "exact_wording": {"type": "string", "description": "The exact quoted text from the lease document"},
                            "clause_reference": {"type": "string", "description": "Clause number, section, or paragraph reference"}
                        },
                        "required": ["type", "exact_wording", "clause_reference"],
                        "additionalProperties": False
                    },
                    "description": "Escalation caps with exact wording and clause references"
                },
                "allowed_fees": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "exact_wording": {"type": "string", "description": "The exact quoted text from the lease document"},
                            "clause_reference": {"type": "string", "description": "Clause number, section, or paragraph reference"}
                        },
                        "required": ["exact_wording", "clause_reference"],
                        "additionalProperties": False
                    },
                    "description": "Allowed fees with exact wording and clause references"
                },
                "disallowed_fees": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "exact_wording": {"type": "string", "description": "The exact quoted text from the lease document"},
                            "clause_reference": {"type": "string", "description": "Clause number, section, or paragraph reference"}
                        },
                        "required": ["exact_wording", "clause_reference"],
                        "additionalProperties": False
                    },
                    "description": "Disallowed fees with exact wording and clause references"
                }
            },
            "required": ["cam_rules", "taxes", "utilities", "escalation_caps", "allowed_fees", "disallowed_fees"],
            "additionalProperties": False
        }
        
        prompt = f"""Extract the following key information from this lease document. CRITICAL: Extract the EXACT wording from the lease document - do not paraphrase or summarize. Quote the text verbatim.

For each item, provide:
1. The EXACT quoted text from the lease (word-for-word, preserving original language, but cleaned up for readability - remove awkward line breaks, fix spacing)
2. The clause reference (section number, clause number, paragraph number, article number, etc. - look for patterns like "Section 5.2", "Clause 12", "Paragraph 3.1", "Article IV", "§ 5.2", "5.2", etc.)

IMPORTANT FOR CLAUSE REFERENCES:
- Look for section numbers, clause numbers, paragraph numbers, article numbers, or subsection identifiers near the relevant text
- Common patterns: "Section X", "Section X.Y", "Clause X", "Paragraph X", "Article X", "§ X", "Subsection X", "Part X"
- If you see a number followed by text (e.g., "5.2 Common Area Maintenance"), use "Section 5.2" or "Clause 5.2"
- If multiple references exist, use the most specific one (e.g., "Section 5.2.3" is better than "Section 5")
- If no explicit reference is found, look for nearby headings or numbered items that might indicate the section

Categories to extract:
1. CAM Rules: Extract all Common Area Maintenance (CAM) rules, charges, allocation methods, and provisions with exact wording
2. Taxes: Extract all tax-related information with exact wording from the lease
3. Utilities: Extract utility-related information with exact wording from the lease
4. Escalation Caps: Extract any escalation caps, annual increase limits, maximum increase percentages, or rent increase restrictions with exact wording
5. Allowed Fees: Extract all fees and charges that are explicitly allowed, permitted, or authorized with exact wording
6. Disallowed Fees: Extract all fees and charges that are explicitly disallowed, prohibited, or not allowed with exact wording

Lease Document Text:
{text}

IMPORTANT: 
- Copy the exact text from the lease document - do not rephrase or summarize, but clean up awkward line breaks and spacing
- ALWAYS try to find a clause/section/paragraph reference - look carefully for numbered sections, clauses, or paragraphs near the relevant text
- If you cannot find a specific reference, use "See lease document" as a fallback
- If a category has no information, return an empty array"""

        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an expert at extracting structured information from lease documents. Extract EXACT wording verbatim - do not paraphrase, summarize, or reword. Always look carefully for clause/section/paragraph references near the relevant text. Clean up awkward line breaks and spacing in the extracted text for readability."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_schema", "json_schema": {"name": "lease_extraction", "strict": True, "schema": lease_schema}}
        )
        
        lease_data = json.loads(response.choices[0].message.content)
        lease_data['raw_text'] = text  # Keep original text
        
        # Smooth extracted wording for all fields
        def smooth_lease_item(item):
            if isinstance(item, dict):
                if 'exact_wording' in item:
                    item['exact_wording'] = smooth_extracted_text(item['exact_wording'])
                # Ensure clause_reference is not empty - use fallback if needed
                if not item.get('clause_reference') or item.get('clause_reference', '').strip() == '':
                    item['clause_reference'] = 'See lease document'
            return item
        
        # Apply smoothing to all extracted items
        if isinstance(lease_data.get('cam_rules'), list):
            lease_data['cam_rules'] = [smooth_lease_item(item) for item in lease_data['cam_rules']]
        if isinstance(lease_data.get('taxes'), list):
            taxes_list = [smooth_lease_item(item) for item in lease_data.get('taxes', [])]
            taxes_dict = {}
            for item in taxes_list:
                taxes_dict[item['type']] = item['exact_wording']
            lease_data['taxes'] = taxes_dict
            lease_data['taxes_details'] = taxes_list
        if isinstance(lease_data.get('utilities'), list):
            utilities_list = [smooth_lease_item(item) for item in lease_data.get('utilities', [])]
            utilities_dict = {}
            for item in utilities_list:
                utilities_dict[item['type']] = item['exact_wording']
            lease_data['utilities'] = utilities_dict
            lease_data['utilities_details'] = utilities_list
        if isinstance(lease_data.get('escalation_caps'), list):
            escalation_list = [smooth_lease_item(item) for item in lease_data.get('escalation_caps', [])]
            escalation_dict = {}
            for item in escalation_list:
                escalation_dict[item['type']] = item['exact_wording']
            lease_data['escalation_caps'] = escalation_dict
            lease_data['escalation_caps_details'] = escalation_list
        if isinstance(lease_data.get('allowed_fees'), list):
            lease_data['allowed_fees'] = [smooth_lease_item(item) for item in lease_data['allowed_fees']]
        if isinstance(lease_data.get('disallowed_fees'), list):
            lease_data['disallowed_fees'] = [smooth_lease_item(item) for item in lease_data['disallowed_fees']]
        
        return lease_data
        
    except Exception as e:
        st.warning(f"OpenAI extraction failed: {str(e)}. Falling back to regex extraction.")
        return extract_lease_fields_regex(text)


def extract_lease_fields_regex(text: str) -> Dict:
    """Fallback regex-based extraction."""
    lease_data = {
        'cam_rules': [],
        'taxes': {},
        'utilities': {},
        'escalation_caps': {},
        'allowed_fees': [],
        'disallowed_fees': [],
        'raw_text': text
    }
    
    text_lower = text.lower()
    
    # Extract CAM (Common Area Maintenance) rules
    cam_patterns = [
        r'cam[:\s]+([^\.]+)',
        r'common\s+area\s+maintenance[:\s]+([^\.]+)',
        r'common\s+area\s+charges[:\s]+([^\.]+)',
    ]
    for pattern in cam_patterns:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            cam_text = match.group(1).strip()
            if len(cam_text) > 10:
                lease_data['cam_rules'].append(cam_text[:200])
    
    # Extract tax information
    tax_patterns = [
        r'(property\s+tax[es]?|real\s+estate\s+tax[es]?)[:\s]+([0-9,]+\.?\d*%?)',
        r'tax[es]?\s+([0-9,]+\.?\d*%?)',
    ]
    for pattern in tax_patterns:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            tax_type = match.group(1).strip()
            tax_value = match.group(2) if len(match.groups()) > 1 else match.group(1)
            lease_data['taxes'][tax_type] = tax_value
    
    # Extract utilities information
    utility_patterns = [
        r'(electric[ity]?|water|gas|sewer|trash)[:\s]+([^\.\n]+)',
    ]
    for pattern in utility_patterns:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            utility_type = match.group(1).strip()
            utility_info = match.group(2).strip()[:100]
            lease_data['utilities'][utility_type] = utility_info
    
    # Extract escalation caps
    escalation_patterns = [
        r'escalation\s+cap[:\s]+([0-9,]+\.?\d*%?)',
        r'annual\s+increase\s+cap[:\s]+([0-9,]+\.?\d*%?)',
        r'maximum\s+increase[:\s]+([0-9,]+\.?\d*%?)',
    ]
    for pattern in escalation_patterns:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            cap_value = match.group(1).strip()
            lease_data['escalation_caps']['annual'] = cap_value
    
    # Extract allowed fees
    allowed_fee_patterns = [
        r'allowed\s+fee[s]?[:\s]+([^\.\n]+)',
        r'permitted\s+charge[s]?[:\s]+([^\.\n]+)',
    ]
    for pattern in allowed_fee_patterns:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            fees = match.group(1).strip().split(',')
            lease_data['allowed_fees'].extend([f.strip() for f in fees])
    
    # Extract disallowed fees
    disallowed_fee_patterns = [
        r'disallowed\s+fee[s]?[:\s]+([^\.\n]+)',
        r'prohibited\s+charge[s]?[:\s]+([^\.\n]+)',
        r'not\s+allowed[:\s]+([^\.\n]+)',
    ]
    for pattern in disallowed_fee_patterns:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            fees = match.group(1).strip().split(',')
            lease_data['disallowed_fees'].extend([f.strip() for f in fees])
    
    return lease_data


def extract_invoice_line_items(text: str, api_key: str = None) -> List[Dict]:
    """Extract line items from invoice using OpenAI API."""
    if not api_key:
        # Fallback to regex if no API key
        return extract_invoice_line_items_regex(text)
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Truncate text if too long
        max_chars = 12000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[... text truncated ...]"
        
        invoice_schema = {
            "type": "object",
            "properties": {
                "line_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": "Description of the line item or charge"
                            },
                            "amount": {
                                "type": "number",
                                "description": "The monetary amount for this line item"
                            },
                            "category": {
                                "type": "string",
                                "enum": ["CAM", "Tax", "Utilities", "Rent", "Insurance", "Other"],
                                "description": "Category of the charge"
                            }
                        },
                        "required": ["description", "amount", "category"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["line_items"],
            "additionalProperties": False
        }
        
        prompt = f"""Extract all line items from this invoice document. For each line item, extract:
1. Description: The full description of the charge or service
2. Amount: The monetary amount (as a number, not including currency symbols)
3. Category: Categorize each item as one of: CAM (Common Area Maintenance), Tax, Utilities, Rent, Insurance, or Other

Invoice Document Text:
{text}

Extract all line items with their amounts and categorize them appropriately."""

        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an expert at extracting structured line items from invoices. Extract all charges accurately with their amounts and categories."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_schema", "json_schema": {"name": "invoice_extraction", "strict": True, "schema": invoice_schema}}
        )
        
        result = json.loads(response.choices[0].message.content)
        line_items = result.get('line_items', [])
        
        # Add line numbers
        for i, item in enumerate(line_items, 1):
            item['line_number'] = i
        
        return line_items
        
    except Exception as e:
        st.warning(f"OpenAI extraction failed: {str(e)}. Falling back to regex extraction.")
        return extract_invoice_line_items_regex(text)


def extract_invoice_line_items_regex(text: str) -> List[Dict]:
    """Fallback regex-based extraction."""
    line_items = []
    lines = text.split('\n')
    
    # Pattern to match line items: description followed by amount
    amount_pattern = r'\$?([0-9,]+\.?\d{0,2})'
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Look for lines with amounts (likely line items)
        amounts = re.findall(amount_pattern, line)
        if amounts:
            # Try to extract description and amount
            parts = re.split(amount_pattern, line)
            if len(parts) >= 2:
                description = parts[0].strip()
                amount_str = amounts[-1]  # Usually the last amount is the total
                try:
                    amount = float(amount_str.replace(',', ''))
                    if amount > 0 and len(description) > 2:
                        # Try to categorize
                        category = categorize_charge(description)
                        line_items.append({
                            'description': description,
                            'amount': amount,
                            'category': category,
                            'line_number': i + 1
                        })
                except ValueError:
                    pass
    
    # If no line items found with pattern matching, try table extraction
    if not line_items:
        # Look for table-like structures
        for i, line in enumerate(lines):
            if re.search(r'\d+\.\d{2}', line):  # Has decimal amount
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        # Try to find amount at the end
                        for part in reversed(parts):
                            amount_str = re.sub(r'[^\d.]', '', part)
                            if amount_str and '.' in amount_str:
                                amount = float(amount_str)
                                description = ' '.join(parts[:-1]) if len(parts) > 1 else line
                                category = categorize_charge(description)
                                line_items.append({
                                    'description': description,
                                    'amount': amount,
                                    'category': category,
                                    'line_number': i + 1
                                })
                                break
                    except (ValueError, IndexError):
                        pass
    
    return line_items


def categorize_charge(description: str) -> str:
    """Categorize a charge based on its description."""
    desc_lower = description.lower()
    
    categories = {
        'CAM': ['cam', 'common area', 'maintenance'],
        'Tax': ['tax', 'property tax', 'real estate'],
        'Utilities': ['utility', 'electric', 'water', 'gas', 'sewer', 'trash'],
        'Rent': ['rent', 'base rent', 'monthly rent'],
        'Insurance': ['insurance', 'liability'],
        'Other': []
    }
    
    for category, keywords in categories.items():
        if category == 'Other':
            continue
        if any(keyword in desc_lower for keyword in keywords):
            return category
    
    return 'Other'


def find_relevant_clause(lease_data: Dict, search_term: str, category: str = None) -> str:
    """Find the most relevant clause reference for a violation."""
    clause_refs = []
    
    # Check disallowed fees first (most specific)
    disallowed_fees = lease_data.get('disallowed_fees', [])
    if isinstance(disallowed_fees, list):
        for fee_item in disallowed_fees:
            if isinstance(fee_item, dict) and 'exact_wording' in fee_item:
                exact_wording = fee_item.get('exact_wording', '').lower()
                if search_term.lower() in exact_wording:
                    clause_ref = fee_item.get('clause_reference', '').strip()
                    if clause_ref and clause_ref.lower() != 'see lease document':
                        clause_refs.append(clause_ref)
    
    # Check allowed fees (might have relevant restrictions)
    allowed_fees = lease_data.get('allowed_fees', [])
    if isinstance(allowed_fees, list):
        for fee_item in allowed_fees:
            if isinstance(fee_item, dict) and 'exact_wording' in fee_item:
                exact_wording = fee_item.get('exact_wording', '').lower()
                if search_term.lower() in exact_wording:
                    clause_ref = fee_item.get('clause_reference', '').strip()
                    if clause_ref and clause_ref.lower() != 'see lease document':
                        clause_refs.append(clause_ref)
    
    # Check CAM rules if category is CAM or if search term relates
    if category == 'CAM' or 'cam' in search_term.lower() or 'common area' in search_term.lower():
        cam_rules = lease_data.get('cam_rules', [])
        if isinstance(cam_rules, list):
            for cam_item in cam_rules:
                if isinstance(cam_item, dict):
                    clause_ref = cam_item.get('clause_reference', '').strip()
                    if clause_ref and clause_ref.lower() != 'see lease document':
                        clause_refs.append(clause_ref)
    
    # Check utilities if relevant
    if 'utility' in search_term.lower() or category == 'Utilities':
        utilities_details = lease_data.get('utilities_details', [])
        if isinstance(utilities_details, list):
            for util_item in utilities_details:
                if isinstance(util_item, dict):
                    clause_ref = util_item.get('clause_reference', '').strip()
                    if clause_ref and clause_ref.lower() != 'see lease document':
                        clause_refs.append(clause_ref)
    
    # Check taxes if relevant
    if 'tax' in search_term.lower() or category == 'Tax':
        taxes_details = lease_data.get('taxes_details', [])
        if isinstance(taxes_details, list):
            for tax_item in taxes_details:
                if isinstance(tax_item, dict):
                    clause_ref = tax_item.get('clause_reference', '').strip()
                    if clause_ref and clause_ref.lower() != 'see lease document':
                        clause_refs.append(clause_ref)
    
    # Check escalation caps if relevant
    if 'escalation' in search_term.lower() or 'cap' in search_term.lower():
        escalation_details = lease_data.get('escalation_caps_details', [])
        if isinstance(escalation_details, list):
            for esc_item in escalation_details:
                if isinstance(esc_item, dict):
                    clause_ref = esc_item.get('clause_reference', '').strip()
                    if clause_ref and clause_ref.lower() != 'see lease document':
                        clause_refs.append(clause_ref)
    
    # Return first found clause reference, or try to find any clause reference as fallback
    if clause_refs:
        return clause_refs[0]
    
    # Fallback: search all lease sections for any clause reference
    all_sections = [
        lease_data.get('cam_rules', []),
        lease_data.get('disallowed_fees', []),
        lease_data.get('allowed_fees', []),
        lease_data.get('utilities_details', []),
        lease_data.get('taxes_details', []),
        lease_data.get('escalation_caps_details', [])
    ]
    
    for section in all_sections:
        if isinstance(section, list):
            for item in section:
                if isinstance(item, dict):
                    clause_ref = item.get('clause_reference', '').strip()
                    if clause_ref and clause_ref.lower() != 'see lease document':
                        return clause_ref
    
    return ""


def check_violation(description: str, amount: float, category: str, lease_data: Dict, total_invoice: float) -> tuple:
    """
    Check if an invoice line item violates lease terms.
    Returns (is_violation, reason, explanation, clause_reference)
    """
    desc_lower = description.lower()
    
    # 1. Check for Utility Admin Fee / Markups (not allowed)
    utility_markup_keywords = ['utility admin', 'admin fee', 'utility markup', 'utility surcharge', 
                              'utility processing', 'utility handling', 'utility service fee']
    if any(keyword in desc_lower for keyword in utility_markup_keywords):
        clause_ref = find_relevant_clause(lease_data, 'utility', 'Utilities')
        if not clause_ref:
            clause_ref = find_relevant_clause(lease_data, 'disallowed', None)
        clause_text = f" (See {clause_ref})" if clause_ref else ""
        return (True, "Utility Admin Fee / Markup", 
                f"Utility markups and administrative fees are typically not allowed unless explicitly stated in the lease{clause_text}. These are pass-through charges and landlords cannot add markups.",
                clause_ref)
    
    # 2. Check for Legal Fees unrelated to Tenant
    legal_keywords = ['legal fee', 'attorney', 'lawyer', 'legal cost', 'litigation']
    if any(keyword in desc_lower for keyword in legal_keywords):
        # Check if it's tenant-related (eviction, collection, etc.)
        tenant_related = any(term in desc_lower for term in ['eviction', 'collection', 'tenant', 'default', 'breach'])
        if not tenant_related:
            clause_ref = find_relevant_clause(lease_data, 'legal', None)
            clause_text = f" (See {clause_ref})" if clause_ref else ""
            return (True, "Legal Fees Unrelated to Tenant", 
                    f"Legal fees that are not related to tenant actions (like eviction or collection) are typically not chargeable to tenants{clause_text}. General legal fees for landlord operations should not be passed through.",
                    clause_ref)
    
    # 3. Check for Capital Improvements (like Roof Replacement)
    capital_improvement_keywords = ['roof', 'capital improvement', 'capital expenditure', 'renovation', 
                                   'remodeling', 'structural', 'building improvement', 'facility upgrade',
                                   'hvac replacement', 'plumbing replacement', 'electrical upgrade']
    if any(keyword in desc_lower for keyword in capital_improvement_keywords):
        clause_ref = find_relevant_clause(lease_data, 'capital', 'CAM')
        clause_text = f" (See {clause_ref})" if clause_ref else ""
        return (True, "Capital Improvement Charge", 
                f"Capital improvements and major repairs (like roof replacement, HVAC systems, structural work) are typically the landlord's responsibility and should not be charged to tenants{clause_text}. These are long-term investments that benefit the property owner.",
                clause_ref)
    
    # 4. Check for Management Fee over 5%
    management_keywords = ['management fee', 'property management', 'management charge', 'mgmt fee']
    if any(keyword in desc_lower for keyword in management_keywords):
        if total_invoice > 0:
            management_percentage = (amount / total_invoice) * 100
            if management_percentage > 5:
                clause_ref = find_relevant_clause(lease_data, 'management', None)
                clause_text = f" (See {clause_ref})" if clause_ref else ""
                return (True, f"Management Fee Over 5% ({management_percentage:.1f}%)", 
                        f"Management fees exceeding 5% of total charges are typically excessive{clause_text}. This charge represents {management_percentage:.1f}% of the total invoice, which exceeds the standard 5% cap.",
                        clause_ref)
            # Also check if management fee is explicitly disallowed
            disallowed_fees = lease_data.get('disallowed_fees', [])
            if isinstance(disallowed_fees, list):
                for fee_item in disallowed_fees:
                    if isinstance(fee_item, dict):
                        fee_text = fee_item.get('exact_wording', '')
                    else:
                        fee_text = str(fee_item)
                    if 'management' in fee_text.lower():
                        clause_ref = fee_item.get('clause_reference', '') if isinstance(fee_item, dict) else ''
                        clause_text = f" (See {clause_ref})" if clause_ref else ""
                        return (True, "Management Fee Not Allowed", 
                                f"Management fees are explicitly disallowed per the lease terms{clause_text}.",
                                clause_ref)
    
    # 5. Check against explicit disallowed fees from lease
    disallowed_fees = lease_data.get('disallowed_fees', [])
    for disallowed in disallowed_fees:
        if isinstance(disallowed, dict):
            disallowed_text = disallowed.get('exact_wording', '')
            clause_ref = disallowed.get('clause_reference', '').strip()
            # Use fallback if clause_ref is empty or generic
            if not clause_ref or clause_ref.lower() == 'see lease document':
                clause_ref = find_relevant_clause(lease_data, 'disallowed', None)
        else:
            disallowed_text = str(disallowed)
            clause_ref = find_relevant_clause(lease_data, 'disallowed', None)
        
        disallowed_lower = disallowed_text.lower()
        # Check if disallowed fee appears in description
        if disallowed_lower in desc_lower or any(word in desc_lower for word in disallowed_lower.split() if len(word) > 3):
            clause_text = f" (See {clause_ref})" if clause_ref else ""
            return (True, f"Disallowed Fee: {disallowed_text[:50]}", 
                    f"This fee type is explicitly disallowed per the lease terms{clause_text}: '{disallowed_text}'",
                    clause_ref)
    
    # 6. Check for CAM charges that might violate CAM rules
    if category == 'CAM':
        cam_rules = lease_data.get('cam_rules', [])
        # Check if CAM charges seem excessive or include non-CAM items
        cam_exclusions = ['capital', 'improvement', 'roof', 'structural', 'renovation', 'upgrade']
        if any(exclusion in desc_lower for exclusion in cam_exclusions):
            clause_ref = find_relevant_clause(lease_data, 'cam', 'CAM')
            clause_text = f" (See {clause_ref})" if clause_ref else ""
            return (True, "Non-CAM Item in CAM Charges", 
                    f"This charge appears to be a capital improvement or non-CAM item incorrectly categorized as CAM{clause_text}. CAM should only include common area maintenance, not capital improvements.",
                    clause_ref)
    
    # 7. Check for duplicate charges
    # This will be handled in the main comparison function
    
    return (False, None, None, None)


def compare_lease_invoice(lease_data: Dict, invoice_data: List[Dict]) -> Dict:
    """Compare lease terms with invoice charges and flag mismatches."""
    comparison = {
        'mismatches': [],
        'overcharges': [],
        'total_overcharge': 0.0,
        'allowed_charges': [],
        'disallowed_charges': [],
        'summary': {}
    }
    
    if not lease_data or not invoice_data:
        return comparison
    
    # Calculate total invoice first (needed for percentage calculations)
    total_invoice = sum(item['amount'] for item in invoice_data)
    
    # Track seen descriptions to detect duplicates
    seen_descriptions = {}
    
    # Check each invoice line item
    for item in invoice_data:
        description = item['description']
        amount = item['amount']
        category = item['category']
        
        # Check for duplicate charges
        desc_normalized = description.lower().strip()
        if desc_normalized in seen_descriptions:
            comparison['mismatches'].append({
                'item': item,
                'reason': 'Duplicate Charge',
                'explanation': f"Duplicate charge detected. This item appears multiple times in the invoice.",
                'suggested_action': 'Review with landlord - remove duplicate'
            })
            comparison['overcharges'].append({
                'description': description,
                'amount': amount,
                'reason': 'Duplicate Charge'
            })
            comparison['total_overcharge'] += amount
            continue
        
        seen_descriptions[desc_normalized] = True
        
        # Check for violations
        is_violation, violation_reason, explanation, clause_reference = check_violation(
            description, amount, category, lease_data, total_invoice
        )
        
        if is_violation:
            comparison['mismatches'].append({
                'item': item,
                'reason': violation_reason,
                'explanation': explanation,
                'clause_reference': clause_reference,
                'suggested_action': 'Review with landlord - request removal or justification'
            })
            comparison['overcharges'].append({
                'description': description,
                'amount': amount,
                'reason': violation_reason
            })
            comparison['total_overcharge'] += amount
            comparison['disallowed_charges'].append({
                'item': item,
                'reason': violation_reason
            })
        else:
            comparison['allowed_charges'].append(item)
    
    # Summary statistics
    comparison['summary'] = {
        'total_invoice_amount': total_invoice,
        'total_overcharge': comparison['total_overcharge'],
        'total_allowed': total_invoice - comparison['total_overcharge'],
        'number_of_items': len(invoice_data),
        'number_of_mismatches': len(comparison['mismatches']),
        'overcharge_percentage': (comparison['total_overcharge'] / total_invoice * 100) if total_invoice > 0 else 0
    }
    
    return comparison


def generate_pdf_report(lease_data: Dict, invoice_data: List[Dict], comparison: Dict) -> bytes:
    """Generate a PDF report with the analysis results."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, 
                           leftMargin=0.75*inch, rightMargin=0.75*inch,
                           topMargin=0.75*inch, bottomMargin=0.75*inch)
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=10,
        spaceBefore=16,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=11,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        spaceAfter=6
    )
    
    # Title
    story.append(Paragraph("Lease & Invoice Analysis Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", normal_style))
    story.append(Spacer(1, 0.4*inch))
    
    # Summary Section
    story.append(Paragraph("Executive Summary", heading_style))
    if comparison and comparison.get('summary'):
        summary = comparison['summary']
        summary_data = [
            ['Metric', 'Value'],
            ['Total Invoice Amount', f"${summary.get('total_invoice_amount', 0):,.2f}"],
            ['Total Overcharge', f"${summary.get('total_overcharge', 0):,.2f}"],
            ['Total Allowed Charges', f"${summary.get('total_allowed', 0):,.2f}"],
            ['Number of Items', str(summary.get('number_of_items', 0))],
            ['Number of Violations', str(summary.get('number_of_mismatches', 0))],
            ['Overcharge Percentage', f"{summary.get('overcharge_percentage', 0):.2f}%"],
        ]
        summary_table = Table(summary_data, colWidths=[3.5*inch, 2.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Lease Terms Section
    if lease_data:
        story.append(Paragraph("Relevant Lease Terms", heading_style))
        
        if lease_data.get('cam_rules'):
            story.append(Paragraph("<b>CAM Rules:</b>", subheading_style))
            cam_rules = lease_data['cam_rules']
            if isinstance(cam_rules, list):
                for cam_item in cam_rules[:3]:  # Limit to first 3
                    if isinstance(cam_item, dict):
                        exact_wording = cam_item.get('exact_wording', '')
                        clause_ref = cam_item.get('clause_reference', '')
                        clause_text = f" <i>({clause_ref})</i>" if clause_ref else ""
                        story.append(Paragraph(f"• \"{exact_wording[:200]}{'...' if len(exact_wording) > 200 else ''}\"{clause_text}", normal_style))
                    else:
                        story.append(Paragraph(f"• {str(cam_item)[:200]}...", normal_style))
            story.append(Spacer(1, 0.15*inch))
        
        if lease_data.get('disallowed_fees'):
            story.append(Paragraph("<b>Disallowed Fees:</b>", subheading_style))
            disallowed_fees = lease_data['disallowed_fees']
            if isinstance(disallowed_fees, list):
                for fee_item in disallowed_fees[:5]:
                    if isinstance(fee_item, dict):
                        exact_wording = fee_item.get('exact_wording', '')
                        clause_ref = fee_item.get('clause_reference', '')
                        clause_text = f" <i>({clause_ref})</i>" if clause_ref else ""
                        story.append(Paragraph(f"• \"{exact_wording[:150]}{'...' if len(exact_wording) > 150 else ''}\"{clause_text}", normal_style))
                    else:
                        story.append(Paragraph(f"• {str(fee_item)}", normal_style))
            story.append(Spacer(1, 0.15*inch))
        
        if lease_data.get('allowed_fees'):
            story.append(Paragraph("<b>Allowed Fees:</b>", subheading_style))
            allowed_fees = lease_data['allowed_fees']
            if isinstance(allowed_fees, list):
                for fee_item in allowed_fees[:5]:
                    if isinstance(fee_item, dict):
                        exact_wording = fee_item.get('exact_wording', '')
                        clause_ref = fee_item.get('clause_reference', '')
                        clause_text = f" <i>({clause_ref})</i>" if clause_ref else ""
                        story.append(Paragraph(f"• \"{exact_wording[:150]}{'...' if len(exact_wording) > 150 else ''}\"{clause_text}", normal_style))
                    else:
                        story.append(Paragraph(f"• {str(fee_item)}", normal_style))
            story.append(Spacer(1, 0.3*inch))
    
    # Invoice Line Items Section
    if invoice_data:
        story.append(Paragraph("Invoice Line Items", heading_style))
        invoice_table_data = [['Description', 'Amount', 'Category']]
        for item in invoice_data:
            invoice_table_data.append([
                item['description'][:60],  # Truncate long descriptions
                f"${item['amount']:,.2f}",
                item['category']
            ])
        
        invoice_table = Table(invoice_table_data, colWidths=[3.8*inch, 1*inch, 1.2*inch])
        invoice_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
            ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 1), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
        ]))
        story.append(invoice_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Mismatches Section
    if comparison and comparison.get('mismatches'):
        story.append(Paragraph("Flagged Violations & Overcharges", heading_style))
        
        # Create a summary table with violations
        mismatch_table_data = [['Description', 'Amount', 'Violation Type', 'Clause']]
        for mismatch in comparison['mismatches']:
            item = mismatch['item']
            violation_type = mismatch.get('reason', 'Unknown Violation')
            clause_ref = mismatch.get('clause_reference', '')
            clause_display = clause_ref[:20] if clause_ref else 'N/A'
            mismatch_table_data.append([
                item['description'][:40],
                f"${item['amount']:,.2f}",
                violation_type[:30],
                clause_display
            ])
        
        mismatch_table = Table(mismatch_table_data, colWidths=[2.5*inch, 0.9*inch, 2*inch, 1.1*inch])
        mismatch_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#c0392b')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
            ('ALIGN', (2, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (3, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ffebee')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#b71c1c')),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ffebee')]),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 1), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
        ]))
        story.append(mismatch_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Add detailed explanations for each violation
        story.append(Paragraph("Detailed Violation Analysis", subheading_style))
        for i, mismatch in enumerate(comparison['mismatches'], 1):
            item = mismatch['item']
            violation_type = mismatch.get('reason', 'Unknown Violation')
            explanation = mismatch.get('explanation', 'No explanation provided.')
            clause_ref = mismatch.get('clause_reference', '')
            suggested_action = mismatch.get('suggested_action', 'Review with landlord')
            
            # Violation header with clause reference
            clause_text = f" <i>(Lease Reference: {clause_ref})</i>" if clause_ref else ""
            story.append(Paragraph(f"<b>Violation #{i}: {violation_type}</b>{clause_text}", normal_style))
            story.append(Paragraph(f"<b>Charge:</b> {item['description']} - <b>${item['amount']:,.2f}</b>", normal_style))
            story.append(Paragraph(f"<b>Explanation:</b> {explanation}", normal_style))
            story.append(Paragraph(f"<b>Recommended Action:</b> {suggested_action}", normal_style))
            story.append(Spacer(1, 0.2*inch))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# Main UI
tab1, tab2 = st.tabs(["📤 Upload Documents", "📄 Report"])

with tab1:
    st.header("Upload Documents")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Lease Document")
        lease_file = st.file_uploader(
            "Upload Lease PDF",
            type=['pdf'],
            key='lease_upload'
        )
        
        if lease_file is not None:
            if st.button("Extract Lease Data", key='extract_lease'):
                with st.spinner("Extracting lease information..."):
                    text = extract_text_from_pdf(lease_file)
                    if text:
                        api_key = st.session_state.get('openai_api_key', '')
                        lease_data = extract_lease_fields(text, api_key=api_key)
                        st.session_state.lease_data = lease_data
                        st.success("Lease data extracted successfully!")
                        
                        # Display extracted fields in tables with exact wording
                        if lease_data.get('cam_rules'):
                            st.subheader("CAM Rules")
                            cam_rules = lease_data['cam_rules']
                            if isinstance(cam_rules, list) and len(cam_rules) > 0 and isinstance(cam_rules[0], dict):
                                cam_data = []
                                for cam_item in cam_rules:
                                    cam_data.append({
                                        'Exact Wording': cam_item.get('exact_wording', ''),
                                        'Clause Reference': cam_item.get('clause_reference', '').strip() or 'See lease document'
                                    })
                                cam_df = pd.DataFrame(cam_data)
                            else:
                                cam_df = pd.DataFrame({'Rule': cam_rules if isinstance(cam_rules, list) else [cam_rules]})
                            st.dataframe(cam_df, use_container_width=True, hide_index=True)
                        
                        if lease_data.get('taxes'):
                            st.subheader("Taxes")
                            taxes_details = lease_data.get('taxes_details', [])
                            if taxes_details and isinstance(taxes_details[0], dict):
                                tax_data = []
                                for tax_item in taxes_details:
                                    tax_data.append({
                                        'Tax Type': tax_item.get('type', ''),
                                        'Exact Wording': tax_item.get('exact_wording', ''),
                                        'Clause Reference': tax_item.get('clause_reference', '').strip() or 'See lease document'
                                    })
                                tax_df = pd.DataFrame(tax_data)
                            else:
                                tax_df = pd.DataFrame({
                                    'Tax Type': list(lease_data['taxes'].keys()),
                                    'Value': list(lease_data['taxes'].values())
                                })
                            st.dataframe(tax_df, use_container_width=True, hide_index=True)
                        
                        if lease_data.get('utilities'):
                            st.subheader("Utilities")
                            utilities_details = lease_data.get('utilities_details', [])
                            if utilities_details and isinstance(utilities_details[0], dict):
                                util_data = []
                                for util_item in utilities_details:
                                    util_data.append({
                                        'Utility Type': util_item.get('type', ''),
                                        'Exact Wording': util_item.get('exact_wording', ''),
                                        'Clause Reference': util_item.get('clause_reference', '').strip() or 'See lease document'
                                    })
                                util_df = pd.DataFrame(util_data)
                            else:
                                util_df = pd.DataFrame({
                                    'Utility Type': list(lease_data['utilities'].keys()),
                                    'Details': list(lease_data['utilities'].values())
                                })
                            st.dataframe(util_df, use_container_width=True, hide_index=True)
                        
                        if lease_data.get('escalation_caps'):
                            st.subheader("Escalation Caps")
                            escalation_details = lease_data.get('escalation_caps_details', [])
                            if escalation_details and isinstance(escalation_details[0], dict):
                                esc_data = []
                                for esc_item in escalation_details:
                                    esc_data.append({
                                        'Type': esc_item.get('type', ''),
                                        'Exact Wording': esc_item.get('exact_wording', ''),
                                        'Clause Reference': esc_item.get('clause_reference', '').strip() or 'See lease document'
                                    })
                                esc_df = pd.DataFrame(esc_data)
                            else:
                                esc_df = pd.DataFrame({
                                    'Type': list(lease_data['escalation_caps'].keys()),
                                    'Cap Value': list(lease_data['escalation_caps'].values())
                                })
                            st.dataframe(esc_df, use_container_width=True, hide_index=True)
                        
                        if lease_data.get('allowed_fees'):
                            st.subheader("Allowed Fees")
                            allowed_fees = lease_data['allowed_fees']
                            if isinstance(allowed_fees, list) and len(allowed_fees) > 0 and isinstance(allowed_fees[0], dict):
                                allowed_data = []
                                for fee_item in allowed_fees:
                                    allowed_data.append({
                                        'Exact Wording': fee_item.get('exact_wording', ''),
                                        'Clause Reference': fee_item.get('clause_reference', '').strip() or 'See lease document'
                                    })
                                allowed_df = pd.DataFrame(allowed_data)
                            else:
                                allowed_df = pd.DataFrame({'Fee': allowed_fees if isinstance(allowed_fees, list) else [allowed_fees]})
                            st.dataframe(allowed_df, use_container_width=True, hide_index=True)
                        
                        if lease_data.get('disallowed_fees'):
                            st.subheader("Disallowed Fees")
                            disallowed_fees = lease_data['disallowed_fees']
                            if isinstance(disallowed_fees, list) and len(disallowed_fees) > 0 and isinstance(disallowed_fees[0], dict):
                                disallowed_data = []
                                for fee_item in disallowed_fees:
                                    disallowed_data.append({
                                        'Exact Wording': fee_item.get('exact_wording', ''),
                                        'Clause Reference': fee_item.get('clause_reference', '').strip() or 'See lease document'
                                    })
                                disallowed_df = pd.DataFrame(disallowed_data)
                            else:
                                disallowed_df = pd.DataFrame({'Fee': disallowed_fees if isinstance(disallowed_fees, list) else [disallowed_fees]})
                            st.dataframe(disallowed_df, use_container_width=True, hide_index=True)
                        
                        if not any([lease_data.get('cam_rules'), lease_data.get('taxes'), 
                                   lease_data.get('utilities'), lease_data.get('escalation_caps'),
                                   lease_data.get('allowed_fees'), lease_data.get('disallowed_fees')]):
                            st.info("No structured fields were extracted. The PDF may need manual review.")
                    else:
                        st.error("Could not extract text from PDF. Please ensure it's a valid PDF.")
    
    with col2:
        st.subheader("Invoice Document")
        invoice_file = st.file_uploader(
            "Upload Invoice PDF",
            type=['pdf'],
            key='invoice_upload'
        )
        
        if invoice_file is not None:
            if st.button("Extract Invoice Data", key='extract_invoice'):
                with st.spinner("Extracting invoice information..."):
                    text = extract_text_from_pdf(invoice_file)
                    if text:
                        api_key = st.session_state.get('openai_api_key', '')
                        invoice_data = extract_invoice_line_items(text, api_key=api_key)
                        st.session_state.invoice_data = invoice_data
                        st.success(f"Invoice data extracted successfully! Found {len(invoice_data)} line items.")
                        if invoice_data:
                            st.dataframe(invoice_data)
                    else:
                        st.error("Could not extract text from PDF. Please ensure it's a valid PDF.")

with tab2:
    st.header("Analysis Report")
    
    if st.session_state.lease_data and st.session_state.invoice_data:
        # Run comparison if not already done
        if not st.session_state.comparison_results:
            with st.spinner("Running comparison analysis..."):
                comparison = compare_lease_invoice(
                    st.session_state.lease_data,
                    st.session_state.invoice_data
                )
                st.session_state.comparison_results = comparison
        
        comparison = st.session_state.comparison_results
        
        # Display Summary
        if comparison and comparison.get('summary'):
            st.subheader("📊 Executive Summary")
            summary = comparison['summary']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Invoice", f"${summary.get('total_invoice_amount', 0):,.2f}")
            with col2:
                st.metric("Total Overcharge", f"${summary.get('total_overcharge', 0):,.2f}", 
                         delta=f"-${summary.get('total_overcharge', 0):,.2f}")
            with col3:
                st.metric("Allowed Charges", f"${summary.get('total_allowed', 0):,.2f}")
            with col4:
                st.metric("Violations Found", summary.get('number_of_mismatches', 0))
            
            st.progress(summary.get('overcharge_percentage', 0) / 100 if summary.get('overcharge_percentage', 0) <= 100 else 1.0)
            st.caption(f"Overcharge Percentage: {summary.get('overcharge_percentage', 0):.2f}%")
            st.divider()
        
        # Display Violations with Exact Wording and Clause References
        if comparison and comparison.get('mismatches'):
            st.subheader("🚨 Flagged Violations & Overcharges")
            
            for i, mismatch in enumerate(comparison['mismatches'], 1):
                item = mismatch['item']
                violation_type = mismatch.get('reason', 'Unknown Violation')
                explanation = mismatch.get('explanation', 'No explanation provided.')
                clause_ref = mismatch.get('clause_reference', '')
                suggested_action = mismatch.get('suggested_action', 'Review with landlord')
                
                with st.expander(f"Violation #{i}: {violation_type} - ${item['amount']:,.2f}", expanded=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**Charge:** {item['description']}")
                        st.write(f"**Amount:** ${item['amount']:,.2f}")
                        st.write(f"**Category:** {item.get('category', 'N/A')}")
                    with col2:
                        if clause_ref and clause_ref.lower() != 'see lease document':
                            st.info(f"📋 **Clause Reference:**\n{clause_ref}")
                        elif clause_ref:
                            st.caption(f"📋 {clause_ref}")
                        else:
                            st.caption("📋 Clause reference not available")
                    
                    st.write("**Explanation:**")
                    st.write(explanation)
                    
                    st.write("**Recommended Action:**")
                    st.info(suggested_action)
            
            st.divider()
        
        # Display Relevant Lease Terms (Exact Wording)
        if st.session_state.lease_data:
            st.subheader("📄 Relevant Lease Terms (Exact Wording)")
            
            lease_data = st.session_state.lease_data
            
            # Disallowed Fees
            if lease_data.get('disallowed_fees'):
                st.write("**Disallowed Fees:**")
                disallowed_fees = lease_data['disallowed_fees']
                if isinstance(disallowed_fees, list) and len(disallowed_fees) > 0 and isinstance(disallowed_fees[0], dict):
                    for fee_item in disallowed_fees:
                        exact_wording = fee_item.get('exact_wording', '')
                        clause_ref = fee_item.get('clause_reference', '')
                        if clause_ref:
                            st.write(f"• \"{exact_wording}\" *({clause_ref})*")
                        else:
                            st.write(f"• \"{exact_wording}\"")
                else:
                    for fee in disallowed_fees if isinstance(disallowed_fees, list) else [disallowed_fees]:
                        st.write(f"• {fee}")
                st.write("")
            
            # CAM Rules
            if lease_data.get('cam_rules'):
                st.write("**CAM Rules:**")
                cam_rules = lease_data['cam_rules']
                if isinstance(cam_rules, list) and len(cam_rules) > 0 and isinstance(cam_rules[0], dict):
                    for cam_item in cam_rules[:3]:  # Show first 3
                        exact_wording = cam_item.get('exact_wording', '')
                        clause_ref = cam_item.get('clause_reference', '')
                        if clause_ref:
                            st.write(f"• \"{exact_wording[:300]}{'...' if len(exact_wording) > 300 else ''}\" *({clause_ref})*")
                        else:
                            st.write(f"• \"{exact_wording[:300]}{'...' if len(exact_wording) > 300 else ''}\"")
                else:
                    for cam in (cam_rules[:3] if isinstance(cam_rules, list) else [cam_rules]):
                        st.write(f"• {str(cam)[:300]}...")
                st.write("")
            
            st.divider()
        
        # Display Invoice Line Items
        if st.session_state.invoice_data:
            st.subheader("📋 Invoice Line Items")
            invoice_df = pd.DataFrame(st.session_state.invoice_data)
            st.dataframe(invoice_df[['description', 'amount', 'category']], use_container_width=True, hide_index=True)
            st.divider()
        
        # Display Allowed Charges
        if comparison and comparison.get('allowed_charges'):
            st.subheader("✅ Allowed Charges")
            allowed_df = pd.DataFrame(comparison['allowed_charges'])
            st.dataframe(allowed_df[['description', 'amount', 'category']], use_container_width=True, hide_index=True)
            st.divider()
        
        # PDF Generation Button
        st.subheader("📥 Download Report")
        if st.button("Generate PDF Report", key='generate_report', type="primary"):
            with st.spinner("Generating PDF report..."):
                pdf_bytes = generate_pdf_report(
                    st.session_state.lease_data,
                    st.session_state.invoice_data,
                    st.session_state.comparison_results
                )
                
                st.success("PDF report generated successfully!")
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"lease_invoice_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    type="primary"
                )
    else:
        st.info("Please upload and extract data from both lease and invoice documents in the Upload tab to generate a report.")

