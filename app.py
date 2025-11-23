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
                    "items": {"type": "string"},
                    "description": "List of CAM (Common Area Maintenance) rules, charges, and provisions"
                },
                "taxes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "value": {"type": "string"}
                        },
                        "required": ["type", "value"],
                        "additionalProperties": False
                    },
                    "description": "Tax-related information as array of {type, value} objects"
                },
                "utilities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "details": {"type": "string"}
                        },
                        "required": ["type", "details"],
                        "additionalProperties": False
                    },
                    "description": "Utility-related information as array of {type, details} objects"
                },
                "escalation_caps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "value": {"type": "string"}
                        },
                        "required": ["type", "value"],
                        "additionalProperties": False
                    },
                    "description": "Escalation caps as array of {type, value} objects"
                },
                "allowed_fees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of fees and charges that are explicitly allowed or permitted"
                },
                "disallowed_fees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of fees and charges that are explicitly disallowed or prohibited"
                }
            },
            "required": ["cam_rules", "taxes", "utilities", "escalation_caps", "allowed_fees", "disallowed_fees"],
            "additionalProperties": False
        }
        
        prompt = f"""Extract the following key information from this lease document:

1. CAM Rules: Extract all Common Area Maintenance (CAM) rules, charges, allocation methods, and provisions
2. Taxes: Extract all tax-related information including property taxes, real estate taxes, tax rates, and tax responsibilities
3. Utilities: Extract utility-related information including electricity, water, gas, sewer, trash, and any utility charges or responsibilities
4. Escalation Caps: Extract any escalation caps, annual increase limits, maximum increase percentages, or rent increase restrictions
5. Allowed Fees: Extract all fees and charges that are explicitly allowed, permitted, or authorized in the lease
6. Disallowed Fees: Extract all fees and charges that are explicitly disallowed, prohibited, or not allowed in the lease

Lease Document Text:
{text}

Extract all relevant information. If a category has no information, return an empty array or object as appropriate."""

        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an expert at extracting structured information from lease documents. Extract all relevant fields accurately."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_schema", "json_schema": {"name": "lease_extraction", "strict": True, "schema": lease_schema}}
        )
        
        lease_data = json.loads(response.choices[0].message.content)
        lease_data['raw_text'] = text  # Keep original text
        
        # Convert arrays back to dictionaries for compatibility
        if isinstance(lease_data.get('taxes'), list):
            lease_data['taxes'] = {item['type']: item['value'] for item in lease_data.get('taxes', [])}
        if isinstance(lease_data.get('utilities'), list):
            lease_data['utilities'] = {item['type']: item['details'] for item in lease_data.get('utilities', [])}
        if isinstance(lease_data.get('escalation_caps'), list):
            lease_data['escalation_caps'] = {item['type']: item['value'] for item in lease_data.get('escalation_caps', [])}
        
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


def check_violation(description: str, amount: float, category: str, lease_data: Dict, total_invoice: float) -> tuple:
    """
    Check if an invoice line item violates lease terms.
    Returns (is_violation, reason, explanation)
    """
    desc_lower = description.lower()
    
    # 1. Check for Utility Admin Fee / Markups (not allowed)
    utility_markup_keywords = ['utility admin', 'admin fee', 'utility markup', 'utility surcharge', 
                              'utility processing', 'utility handling', 'utility service fee']
    if any(keyword in desc_lower for keyword in utility_markup_keywords):
        return (True, "Utility Admin Fee / Markup", 
                "Utility markups and administrative fees are typically not allowed unless explicitly stated in the lease. These are pass-through charges and landlords cannot add markups.")
    
    # 2. Check for Legal Fees unrelated to Tenant
    legal_keywords = ['legal fee', 'attorney', 'lawyer', 'legal cost', 'litigation']
    if any(keyword in desc_lower for keyword in legal_keywords):
        # Check if it's tenant-related (eviction, collection, etc.)
        tenant_related = any(term in desc_lower for term in ['eviction', 'collection', 'tenant', 'default', 'breach'])
        if not tenant_related:
            return (True, "Legal Fees Unrelated to Tenant", 
                    "Legal fees that are not related to tenant actions (like eviction or collection) are typically not chargeable to tenants. General legal fees for landlord operations should not be passed through.")
    
    # 3. Check for Capital Improvements (like Roof Replacement)
    capital_improvement_keywords = ['roof', 'capital improvement', 'capital expenditure', 'renovation', 
                                   'remodeling', 'structural', 'building improvement', 'facility upgrade',
                                   'hvac replacement', 'plumbing replacement', 'electrical upgrade']
    if any(keyword in desc_lower for keyword in capital_improvement_keywords):
        return (True, "Capital Improvement Charge", 
                "Capital improvements and major repairs (like roof replacement, HVAC systems, structural work) are typically the landlord's responsibility and should not be charged to tenants. These are long-term investments that benefit the property owner.")
    
    # 4. Check for Management Fee over 5%
    management_keywords = ['management fee', 'property management', 'management charge', 'mgmt fee']
    if any(keyword in desc_lower for keyword in management_keywords):
        if total_invoice > 0:
            management_percentage = (amount / total_invoice) * 100
            if management_percentage > 5:
                return (True, f"Management Fee Over 5% ({management_percentage:.1f}%)", 
                        f"Management fees exceeding 5% of total charges are typically excessive. This charge represents {management_percentage:.1f}% of the total invoice, which exceeds the standard 5% cap.")
            # Also check if management fee is explicitly disallowed
            disallowed_fees = lease_data.get('disallowed_fees', [])
            if any('management' in fee.lower() for fee in disallowed_fees):
                return (True, "Management Fee Not Allowed", 
                        "Management fees are explicitly disallowed per the lease terms.")
    
    # 5. Check against explicit disallowed fees from lease
    disallowed_fees = lease_data.get('disallowed_fees', [])
    for disallowed in disallowed_fees:
        disallowed_lower = disallowed.lower()
        # Check if disallowed fee appears in description
        if disallowed_lower in desc_lower or any(word in desc_lower for word in disallowed_lower.split() if len(word) > 3):
            return (True, f"Disallowed Fee: {disallowed}", 
                    f"This fee type is explicitly disallowed per the lease terms: '{disallowed}'")
    
    # 6. Check for CAM charges that might violate CAM rules
    if category == 'CAM':
        cam_rules = lease_data.get('cam_rules', [])
        # Check if CAM charges seem excessive or include non-CAM items
        cam_exclusions = ['capital', 'improvement', 'roof', 'structural', 'renovation', 'upgrade']
        if any(exclusion in desc_lower for exclusion in cam_exclusions):
            return (True, "Non-CAM Item in CAM Charges", 
                    "This charge appears to be a capital improvement or non-CAM item incorrectly categorized as CAM. CAM should only include common area maintenance, not capital improvements.")
    
    # 7. Check for duplicate charges
    # This will be handled in the main comparison function
    
    return (False, None, None)


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
        is_violation, violation_reason, explanation = check_violation(
            description, amount, category, lease_data, total_invoice
        )
        
        if is_violation:
            comparison['mismatches'].append({
                'item': item,
                'reason': violation_reason,
                'explanation': explanation,
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
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph("Lease & Invoice Analysis Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
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
            ['Number of Mismatches', str(summary.get('number_of_mismatches', 0))],
            ['Overcharge Percentage', f"{summary.get('overcharge_percentage', 0):.2f}%"],
        ]
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.2*inch))
    
    # Lease Terms Section
    if lease_data:
        story.append(Paragraph("Extracted Lease Terms", heading_style))
        
        if lease_data.get('cam_rules'):
            story.append(Paragraph("<b>CAM Rules:</b>", styles['Normal']))
            for cam in lease_data['cam_rules'][:3]:  # Limit to first 3
                story.append(Paragraph(f"• {cam[:150]}...", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
        
        if lease_data.get('escalation_caps'):
            story.append(Paragraph("<b>Escalation Caps:</b>", styles['Normal']))
            for key, value in lease_data['escalation_caps'].items():
                story.append(Paragraph(f"• {key.title()}: {value}", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
        
        if lease_data.get('allowed_fees'):
            story.append(Paragraph("<b>Allowed Fees:</b>", styles['Normal']))
            for fee in lease_data['allowed_fees'][:5]:
                story.append(Paragraph(f"• {fee}", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
        
        if lease_data.get('disallowed_fees'):
            story.append(Paragraph("<b>Disallowed Fees:</b>", styles['Normal']))
            for fee in lease_data['disallowed_fees'][:5]:
                story.append(Paragraph(f"• {fee}", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
    
    # Invoice Line Items Section
    if invoice_data:
        story.append(Paragraph("Invoice Line Items", heading_style))
        invoice_table_data = [['Description', 'Amount', 'Category']]
        for item in invoice_data:
            invoice_table_data.append([
                item['description'][:50],  # Truncate long descriptions
                f"${item['amount']:,.2f}",
                item['category']
            ])
        
        invoice_table = Table(invoice_table_data, colWidths=[3.5*inch, 1*inch, 1*inch])
        invoice_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        story.append(invoice_table)
        story.append(Spacer(1, 0.2*inch))
    
    # Mismatches Section
    if comparison and comparison.get('mismatches'):
        story.append(Paragraph("🚨 Flagged Violations & Overcharges", heading_style))
        
        # Create a detailed table with explanations
        mismatch_table_data = [['Description', 'Amount', 'Violation Type']]
        for mismatch in comparison['mismatches']:
            item = mismatch['item']
            violation_type = mismatch.get('reason', 'Unknown Violation')
            mismatch_table_data.append([
                item['description'][:45],
                f"${item['amount']:,.2f}",
                violation_type[:35]
            ])
        
        mismatch_table = Table(mismatch_table_data, colWidths=[3*inch, 1*inch, 2.5*inch])
        mismatch_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightcoral),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        story.append(mismatch_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Add detailed explanations for each violation
        story.append(Paragraph("<b>Violation Details:</b>", styles['Heading3']))
        for i, mismatch in enumerate(comparison['mismatches'], 1):
            item = mismatch['item']
            violation_type = mismatch.get('reason', 'Unknown Violation')
            explanation = mismatch.get('explanation', 'No explanation provided.')
            suggested_action = mismatch.get('suggested_action', 'Review with landlord')
            
            story.append(Paragraph(f"<b>Violation #{i}: {violation_type}</b>", styles['Normal']))
            story.append(Paragraph(f"<b>Charge:</b> {item['description']} - ${item['amount']:,.2f}", styles['Normal']))
            story.append(Paragraph(f"<b>Explanation:</b> {explanation}", styles['Normal']))
            story.append(Paragraph(f"<b>Recommended Action:</b> {suggested_action}", styles['Normal']))
            story.append(Spacer(1, 0.15*inch))
    
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
                        
                        # Display extracted fields in tables
                        if lease_data.get('cam_rules'):
                            st.subheader("CAM Rules")
                            cam_df = pd.DataFrame({
                                'Rule': lease_data['cam_rules']
                            })
                            st.dataframe(cam_df, use_container_width=True, hide_index=True)
                        
                        if lease_data.get('taxes'):
                            st.subheader("Taxes")
                            tax_df = pd.DataFrame({
                                'Tax Type': list(lease_data['taxes'].keys()),
                                'Value': list(lease_data['taxes'].values())
                            })
                            st.dataframe(tax_df, use_container_width=True, hide_index=True)
                        
                        if lease_data.get('utilities'):
                            st.subheader("Utilities")
                            util_df = pd.DataFrame({
                                'Utility Type': list(lease_data['utilities'].keys()),
                                'Details': list(lease_data['utilities'].values())
                            })
                            st.dataframe(util_df, use_container_width=True, hide_index=True)
                        
                        if lease_data.get('escalation_caps'):
                            st.subheader("Escalation Caps")
                            esc_df = pd.DataFrame({
                                'Type': list(lease_data['escalation_caps'].keys()),
                                'Cap Value': list(lease_data['escalation_caps'].values())
                            })
                            st.dataframe(esc_df, use_container_width=True, hide_index=True)
                        
                        if lease_data.get('allowed_fees'):
                            st.subheader("Allowed Fees")
                            allowed_df = pd.DataFrame({
                                'Fee': lease_data['allowed_fees']
                            })
                            st.dataframe(allowed_df, use_container_width=True, hide_index=True)
                        
                        if lease_data.get('disallowed_fees'):
                            st.subheader("Disallowed Fees")
                            disallowed_df = pd.DataFrame({
                                'Fee': lease_data['disallowed_fees']
                            })
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
    st.header("Generate Report")
    
    if st.session_state.lease_data and st.session_state.invoice_data:
        # Run comparison if not already done
        if not st.session_state.comparison_results:
            with st.spinner("Running comparison analysis..."):
                comparison = compare_lease_invoice(
                    st.session_state.lease_data,
                    st.session_state.invoice_data
                )
                st.session_state.comparison_results = comparison
        
        if st.button("Generate PDF Report", key='generate_report'):
            with st.spinner("Generating PDF report..."):
                pdf_bytes = generate_pdf_report(
                    st.session_state.lease_data,
                    st.session_state.invoice_data,
                    st.session_state.comparison_results
                )
                
                st.success("PDF report generated successfully!")
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"lease_invoice_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
    else:
        st.info("Please upload and extract data from both lease and invoice documents in the Upload tab to generate a report.")

