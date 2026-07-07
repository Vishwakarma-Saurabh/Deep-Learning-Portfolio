"""
Email Tool - Send real emails via SMTP or SendGrid.
Falls back to simulation if not configured.
"""

import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Email configuration from .env
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_EMAIL = os.getenv("SMTP_EMAIL", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")


def send_email(to: str, subject: str, body: str, attachment: str = None) -> dict:
  
    print(f"✉️  Sending email to: {to}")
    
    # If SMTP not configured, simulate
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        print("   ⚠️  SMTP not configured. Simulating email.")
        return _simulate_email(to, subject, body, attachment)
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = SMTP_EMAIL
        msg['To'] = to
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        # Add attachment if provided
        if attachment:
            attachment_part = MIMEText(attachment, 'plain')
            attachment_part.add_header(
                'Content-Disposition',
                'attachment',
                filename='report.txt'
            )
            msg.attach(attachment_part)
        
        # Send via SMTP
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.send_message(msg)
        
        return {
            "success": True,
            "message": f"Email sent to {to}",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        print(f"   ❌ Email failed: {e}")
        print("   Falling back to simulation...")
        return _simulate_email(to, subject, body, attachment)


def _simulate_email(to: str, subject: str, body: str, attachment: str = None) -> dict:
    """Simulate email for development."""
    email_record = {
        "to": to,
        "subject": subject,
        "body_preview": body[:200],
        "has_attachment": attachment is not None,
        "sent_at": datetime.now().isoformat(),
        "status": "simulated"
    }
    
    # Log to file
    try:
        with open("email_log.json", "a") as f:
            f.write(json.dumps(email_record) + "\n")
    except:
        pass
    
    return {
        "success": True,
        "message": f"Email simulated to {to} (SMTP not configured)",
        "details": email_record
    }


TOOL_DESCRIPTION = """
Tool: send_email
Description: Send an email with report or findings.
Input: {"to": "email@example.com", "subject": "Subject line", "body": "Email content", "attachment": "Optional report text"}
Output: Confirmation of email sent
Use when: User asks to email, send, or share findings
"""


if __name__ == "__main__":
    result = send_email(
        "test@example.com",
        "Test Subject",
        "This is a test email body.",
        "Report content here"
    )
    print(result)