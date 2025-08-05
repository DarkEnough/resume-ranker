import pytest
from core_pipeline.jd_cleaner import clean_job_description

def test_simple_jd_cleaning():
    """Simple test that JD cleaner works."""
    jd = """
    We need a Python developer.
    
    Required Skills:
    - Python programming
    - Django experience
    
    About the Company:
    We are a great company with amazing benefits.
    
    Compensation:
    Great salary and benefits package.
    """
    
    cleaned = clean_job_description(jd)
    
    # Should keep job requirements
    assert "Python developer" in cleaned
    assert "Required Skills:" in cleaned
    assert "Python programming" in cleaned
    
    # Should remove company info and compensation
    assert "About the Company:" not in cleaned
    assert "Compensation:" not in cleaned
    assert "benefits package" not in cleaned
