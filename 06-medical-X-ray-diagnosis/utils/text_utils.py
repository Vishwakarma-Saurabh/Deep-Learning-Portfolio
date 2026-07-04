import re

def clean_report(text):
    """Clean and format report text"""
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove extra periods
    text = re.sub(r'\.\.+', '.', text)
    
    # Capitalize sentences
    sentences = text.split('.')
    sentences = [s.strip().capitalize() for s in sentences if s.strip()]
    text = '. '.join(sentences)
    
    if text and text[-1] != '.':
        text += '.'
    
    return text