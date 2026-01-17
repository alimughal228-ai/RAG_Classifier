"""
Format converter for assessment output specification.

Converts the full pipeline output format to the assessment-required format.
"""

import json
from pathlib import Path
from typing import Any, Dict, List


def convert_to_assessment_format(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert pipeline results to assessment-required format.
    
    Assessment format:
    {
      "invoice_1.pdf": {
        "class": "Invoice",
        "invoice_number": "INV-1234",
        "date": "2025-01-01",
        "company": "ACME Ltd.",
        "total_amount": 350.5
      },
      "resume_1.pdf": {
        "class": "Resume",
        "name": "John Doe",
        "email": "john@example.com",
        "phone": "123-456-7890",
        "experience_years": 5
      }
    }
    
    Args:
        results: List of pipeline results
        
    Returns:
        Dictionary with filename as key and extracted data as value
    """
    assessment_output: Dict[str, Any] = {}
    
    for result in results:
        file_name = result.get("file_name", "unknown")
        classification = result.get("classification", {})
        extracted_data = result.get("extracted_data", {})
        
        # Get document class
        doc_class = classification.get("class", "Unclassifiable") if classification else "Unclassifiable"
        
        # Build output entry
        entry: Dict[str, Any] = {
            "class": doc_class
        }
        
        # Add extracted fields based on document type
        if doc_class in ["Invoice", "Resume", "Utility Bill"]:
            # Flatten extracted_data into entry
            for key, value in extracted_data.items():
                if value is not None:  # Only include non-null values
                    entry[key] = value
        
        # Add to output
        assessment_output[file_name] = entry
    
    return assessment_output


def save_assessment_format(
    results: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """
    Save results in assessment-required format.
    
    Args:
        results: List of pipeline results
        output_path: Path to output JSON file
    """
    assessment_output = convert_to_assessment_format(results)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(assessment_output, f, indent=2, ensure_ascii=False)
    
    print(f"\nAssessment format saved to: {output_path}")
    print(f"Total documents: {len(assessment_output)}")

