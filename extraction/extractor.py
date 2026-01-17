"""
Information extraction from documents.

Extracts structured data using rule-based methods, regex patterns,
and light NLP. Extraction is type-specific based on document classification.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional


def normalize_date(date_str: str) -> Optional[str]:
    """
    Normalize date string to ISO format (YYYY-MM-DD).
    
    Handles various date formats:
    - MM/DD/YYYY, DD/MM/YYYY
    - YYYY-MM-DD
    - Month DD, YYYY
    - DD Month YYYY
    
    Args:
        date_str: Date string to normalize
        
    Returns:
        Normalized date in ISO format (YYYY-MM-DD) or None if parsing fails
    """
    if not date_str or not date_str.strip():
        return None
    
    date_str = date_str.strip()
    
    # Common date patterns
    patterns = [
        # ISO format: YYYY-MM-DD
        (r'(\d{4})-(\d{2})-(\d{2})', lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
        # MM/DD/YYYY or DD/MM/YYYY (try both)
        (r'(\d{1,2})/(\d{1,2})/(\d{4})', lambda m: _try_date_parse(m.group(1), m.group(2), m.group(3))),
        # Month DD, YYYY or DD Month YYYY
        (r'([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})', lambda m: _parse_month_date(m.group(1), m.group(2), m.group(3))),
        (r'(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})', lambda m: _parse_month_date(m.group(2), m.group(1), m.group(3))),
        # DD-MM-YYYY
        (r'(\d{1,2})-(\d{1,2})-(\d{4})', lambda m: _try_date_parse(m.group(1), m.group(2), m.group(3))),
    ]
    
    for pattern, converter in patterns:
        match = re.search(pattern, date_str)
        if match:
            try:
                normalized = converter(match)
                if normalized:
                    # Validate the date
                    datetime.strptime(normalized, '%Y-%m-%d')
                    return normalized
            except (ValueError, AttributeError):
                continue
    
    # Try direct parsing with common formats
    for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%d-%m-%Y', '%B %d, %Y', '%d %B %Y']:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            continue
    
    return None


def _try_date_parse(part1: str, part2: str, year: str) -> Optional[str]:
    """Try to parse date assuming MM/DD/YYYY first, then DD/MM/YYYY."""
    for fmt in ['%m/%d/%Y', '%d/%m/%Y']:
        try:
            dt = datetime.strptime(f"{part1}/{part2}/{year}", fmt)
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            continue
    return None


def _parse_month_date(month_str: str, day: str, year: str) -> Optional[str]:
    """Parse date with month name."""
    month_names = {
        'january': '01', 'february': '02', 'march': '03', 'april': '04',
        'may': '05', 'june': '06', 'july': '07', 'august': '08',
        'september': '09', 'october': '10', 'november': '11', 'december': '12',
        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
        'jun': '06', 'jul': '07', 'aug': '08', 'sep': '09',
        'oct': '10', 'nov': '11', 'dec': '12'
    }
    
    month_lower = month_str.lower()
    month_num = month_names.get(month_lower)
    if month_num:
        try:
            dt = datetime.strptime(f"{year}-{month_num}-{day.zfill(2)}", '%Y-%m-%d')
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            return None
    return None


def extract_currency_amount(text: str) -> Optional[float]:
    """
    Extract currency amount from text.
    
    Args:
        text: Text containing currency amount
        
    Returns:
        Amount as float or None if not found
    """
    # Pattern for currency: $123.45, $1,234.56, 123.45 USD, etc.
    patterns = [
        r'\$[\s]*([\d,]+\.?\d*)',  # $123.45 or $1,234.56
        r'([\d,]+\.?\d*)[\s]*(?:USD|usd|dollars?)',  # 123.45 USD
        r'(?:total|amount|due|balance)[\s]*:?[\s]*\$?[\s]*([\d,]+\.?\d*)',  # Total: $123.45
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Get the last match (usually the total)
            amount_str = matches[-1].replace(',', '')
            try:
                return float(amount_str)
            except ValueError:
                continue
    
    return None


def extract_invoice_number(text: str) -> Optional[str]:
    """
    Extract invoice number from text.
    
    Args:
        text: Document text
        
    Returns:
        Invoice number as string or None
    """
    # Patterns for invoice numbers
    patterns = [
        r'invoice\s*(?:number|#|no\.?)[\s:]*([A-Z0-9\-]+)',
        r'inv\s*(?:number|#|no\.?)[\s:]*([A-Z0-9\-]+)',
        r'invoice[\s:]+([A-Z0-9\-]{4,})',
        r'#[\s]*([A-Z0-9\-]{4,})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            inv_num = match.group(1).strip()
            if len(inv_num) >= 3:  # Reasonable minimum length
                return inv_num
    
    return None


def extract_company_name(text: str) -> Optional[str]:
    """
    Extract company/vendor name from invoice text.
    
    Args:
        text: Document text
        
    Returns:
        Company name or None
    """
    # Look for common invoice header patterns
    patterns = [
        r'(?:from|vendor|company|bill\s+from)[\s:]+([A-Z][A-Za-z\s&\.,]+?)(?:\n|$)',
        r'^([A-Z][A-Za-z\s&\.,]{3,40})(?:\n|$)',
    ]
    
    # Get first few lines (usually contains company name)
    lines = text.split('\n')[:10]
    header_text = '\n'.join(lines)
    
    for pattern in patterns:
        match = re.search(pattern, header_text, re.IGNORECASE | re.MULTILINE)
        if match:
            company = match.group(1).strip()
            # Filter out common non-company words
            if not any(word in company.lower() for word in ['invoice', 'bill', 'date', 'number']):
                if len(company) >= 3 and len(company) <= 100:
                    return company
    
    return None


def extract_email(text: str) -> Optional[str]:
    """
    Extract email address from text.
    
    Args:
        text: Document text
        
    Returns:
        Email address or None
    """
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    match = re.search(pattern, text)
    if match:
        return match.group(0).lower()
    return None


def extract_phone(text: str) -> Optional[str]:
    """
    Extract phone number from text.
    
    Args:
        text: Document text
        
    Returns:
        Phone number as string or None
    """
    # Various phone number patterns
    patterns = [
        r'\+?1?[\s\-]?\(?(\d{3})\)?[\s\-]?(\d{3})[\s\-]?(\d{4})',  # US format
        r'\+?[\d\s\-]{10,15}',  # General international
        r'\(\d{3}\)\s?\d{3}[\s\-]?\d{4}',  # (123) 456-7890
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            if isinstance(matches[0], tuple):
                # Format US phone number
                return f"({matches[0][0]}) {matches[0][1]}-{matches[0][2]}"
            else:
                return matches[0].strip()
    
    return None


def extract_name_from_resume(text: str) -> Optional[str]:
    """
    Extract person's name from resume (usually first line or header).
    
    Args:
        text: Resume text
        
    Returns:
        Name or None
    """
    # Name is usually at the top, before email/phone
    lines = text.split('\n')[:5]
    
    for line in lines:
        line = line.strip()
        # Skip empty lines and common headers
        if not line or any(word in line.lower() for word in ['resume', 'cv', 'curriculum', 'vitae', 'phone', 'email']):
            continue
        
        # Check if line looks like a name (2-4 words, capitalized)
        words = line.split()
        if 2 <= len(words) <= 4:
            # Check if all words start with capital letter
            if all(word[0].isupper() for word in words if word):
                # Exclude common non-name words
                if not any(word.lower() in ['llc', 'inc', 'corp', 'ltd', 'company'] for word in words):
                    return line
    
    return None


def extract_experience_years(text: str) -> Optional[int]:
    """
    Extract years of experience from resume.
    
    Args:
        text: Resume text
        
    Returns:
        Years of experience as integer or None
    """
    # Patterns for experience
    patterns = [
        r'(\d+)[\s]*(?:years?|yrs?)[\s]*(?:of\s+)?(?:experience|exp)',
        r'(?:experience|exp)[\s:]+(\d+)[\s]*(?:years?|yrs?)',
        r'(\d+)\+[\s]*(?:years?|yrs?)[\s]*(?:of\s+)?(?:experience|exp)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                years = int(match.group(1))
                if 0 <= years <= 50:  # Reasonable range
                    return years
            except ValueError:
                continue
    
    return None


def extract_account_number(text: str) -> Optional[str]:
    """
    Extract account number from utility bill.
    
    Args:
        text: Document text
        
    Returns:
        Account number as string or None
    """
    patterns = [
        r'account\s*(?:number|#|no\.?)[\s:]*([A-Z0-9\-]+)',
        r'acct\s*(?:number|#|no\.?)[\s:]*([A-Z0-9\-]+)',
        r'account[\s:]+([A-Z0-9\-]{6,})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            acct_num = match.group(1).strip()
            if len(acct_num) >= 6:  # Reasonable minimum
                return acct_num
    
    return None


def extract_usage_kwh(text: str) -> Optional[float]:
    """
    Extract usage in kWh from utility bill.
    
    Args:
        text: Document text
        
    Returns:
        Usage in kWh as float or None
    """
    patterns = [
        r'(\d+(?:,\d+)*(?:\.\d+)?)[\s]*(?:kwh|k\.w\.h\.?)',
        r'usage[\s:]+(\d+(?:,\d+)*(?:\.\d+)?)[\s]*(?:kwh|k\.w\.h\.?)',
        r'(\d+(?:,\d+)*(?:\.\d+)?)[\s]*(?:kilowatt|kw)[\s]*(?:hours?|hrs?)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            usage_str = match.group(1).replace(',', '')
            try:
                return float(usage_str)
            except ValueError:
                continue
    
    return None


def extract_invoice_data(text: str) -> Dict[str, Any]:
    """
    Extract structured data from Invoice document.
    
    Schema:
    - invoice_number: str
    - date: str (ISO format)
    - company: str
    - total_amount: float
    
    Args:
        text: Invoice document text
        
    Returns:
        Dictionary with extracted fields (None for missing fields)
    """
    result: Dict[str, Any] = {
        "invoice_number": None,
        "date": None,
        "company": None,
        "total_amount": None
    }
    
    # Extract invoice number
    result["invoice_number"] = extract_invoice_number(text)
    
    # Extract date
    date_patterns = [
        r'invoice\s+date[\s:]+([^\n]+)',
        r'date[\s:]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
        r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',  # General date pattern
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            date_str = match.group(1).strip()
            normalized = normalize_date(date_str)
            if normalized:
                result["date"] = normalized
                break
    
    # Extract company name
    result["company"] = extract_company_name(text)
    
    # Extract total amount
    # Look for "total" patterns
    total_patterns = [
        r'total[\s]*(?:amount|due|balance)?[\s:]+[\$]?[\s]*([\d,]+\.?\d*)',
        r'amount\s+due[\s:]+[\$]?[\s]*([\d,]+\.?\d*)',
        r'grand\s+total[\s:]+[\$]?[\s]*([\d,]+\.?\d*)',
    ]
    
    for pattern in total_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount_str = match.group(1).replace(',', '')
            try:
                result["total_amount"] = float(amount_str)
                break
            except ValueError:
                continue
    
    # If no total found, try general currency extraction
    if result["total_amount"] is None:
        result["total_amount"] = extract_currency_amount(text)
    
    return result


def extract_resume_data(text: str) -> Dict[str, Any]:
    """
    Extract structured data from Resume document.
    
    Schema:
    - name: str
    - email: str
    - phone: str
    - experience_years: int
    
    Args:
        text: Resume document text
        
    Returns:
        Dictionary with extracted fields (None for missing fields)
    """
    result: Dict[str, Any] = {
        "name": None,
        "email": None,
        "phone": None,
        "experience_years": None
    }
    
    # Extract name
    result["name"] = extract_name_from_resume(text)
    
    # Extract email
    result["email"] = extract_email(text)
    
    # Extract phone
    result["phone"] = extract_phone(text)
    
    # Extract experience years
    result["experience_years"] = extract_experience_years(text)
    
    return result


def extract_utility_bill_data(text: str) -> Dict[str, Any]:
    """
    Extract structured data from Utility Bill document.
    
    Schema:
    - account_number: str
    - date: str (ISO format)
    - usage_kwh: float
    - amount_due: float
    
    Args:
        text: Utility bill document text
        
    Returns:
        Dictionary with extracted fields (None for missing fields)
    """
    result: Dict[str, Any] = {
        "account_number": None,
        "date": None,
        "usage_kwh": None,
        "amount_due": None
    }
    
    # Extract account number
    result["account_number"] = extract_account_number(text)
    
    # Extract date (bill date or service period)
    date_patterns = [
        r'bill\s+date[\s:]+([^\n]+)',
        r'service\s+period[\s:]+([^\n]+)',
        r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',  # General date pattern
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            date_str = match.group(1).strip()
            normalized = normalize_date(date_str)
            if normalized:
                result["date"] = normalized
                break
    
    # Extract usage in kWh
    result["usage_kwh"] = extract_usage_kwh(text)
    
    # Extract amount due
    amount_patterns = [
        r'amount\s+due[\s:]+[\$]?[\s]*([\d,]+\.?\d*)',
        r'total\s+due[\s:]+[\$]?[\s]*([\d,]+\.?\d*)',
        r'balance\s+due[\s:]+[\$]?[\s]*([\d,]+\.?\d*)',
        r'current\s+charges[\s:]+[\$]?[\s]*([\d,]+\.?\d*)',
    ]
    
    for pattern in amount_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount_str = match.group(1).replace(',', '')
            try:
                result["amount_due"] = float(amount_str)
                break
            except ValueError:
                continue
    
    # If no amount found, try general currency extraction
    if result["amount_due"] is None:
        result["amount_due"] = extract_currency_amount(text)
    
    return result


class InformationExtractor:
    """
    Extracts structured information from documents based on document type.
    
    Uses rule-based methods, regex patterns, and light NLP.
    Extraction is type-specific and only works for classified document types.
    """
    
    def __init__(self) -> None:
        """Initialize the extractor."""
        pass
    
    def extract(
        self,
        text: str,
        document_class: str
    ) -> Dict[str, Any]:
        """
        Extract structured data from document based on its class.
        
        Args:
            text: Document text
            document_class: Classified document type (Invoice, Resume, Utility Bill, etc.)
            
        Returns:
            Dictionary with extracted fields. Returns empty dict for
            Other/Unclassifiable document types.
        """
        # Only extract for classified document types
        if document_class in ["Other", "Unclassifiable"]:
            return {}
        
        # Route to appropriate extractor
        if document_class == "Invoice":
            return extract_invoice_data(text)
        elif document_class == "Resume":
            return extract_resume_data(text)
        elif document_class == "Utility Bill":
            return extract_utility_bill_data(text)
        else:
            # Unknown class, return empty
            return {}
    
    def extract_structured_data(
        self,
        text: str,
        document_class: str
    ) -> Dict[str, Any]:
        """
        Alias for extract() method.
        
        Args:
            text: Document text
            document_class: Classified document type
            
        Returns:
            Dictionary with extracted fields
        """
        return self.extract(text, document_class)
