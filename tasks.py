from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from models import EmailCategory, EmailPriority

class EmailTask(BaseModel):
    id: str
    sender: str
    subject: str
    body: str
    timestamp: str
    difficulty: str
    ground_truth_category: EmailCategory
    ground_truth_priority: EmailPriority
    ground_truth_should_archive: bool
    keywords_in_response: List[str] = Field(default_factory=list)
    expected_action_reason: str = ""

TASKS: List[EmailTask] = [
    # ── EASY ──────────────────────────────────────────────────────────────────
    EmailTask(
        id="E001",
        sender="noreply@scam-offers.biz",
        subject="Urgently Win 1 MILLION DOLLARS!!!",
        body="Dear customer, you have been selected to win a huge prize! Click here: http://malicious-link.com",
        timestamp="2026-04-08 09:00:00",
        difficulty="easy",
        ground_truth_category=EmailCategory.SPAM,
        ground_truth_priority=EmailPriority.LOW,
        ground_truth_should_archive=True,
        expected_action_reason="Contains common spam indicators: suspicious sender, unrealistic prize, malicious link."
    ),
    EmailTask(
        id="E002",
        sender="marketing@junkmail.net",
        subject="Final Call: 90% Off All Socks",
        body="Don't miss out on this incredible sale on our new sock collection. Limited time only!",
        timestamp="2026-04-08 09:15:00",
        difficulty="easy",
        ground_truth_category=EmailCategory.SPAM,
        ground_truth_priority=EmailPriority.LOW,
        ground_truth_should_archive=True,
        expected_action_reason="Promotional marketing email, clearly low priority and likely unwanted."
    ),
    EmailTask(
        id="E003",
        sender="no-reply@newsletter.randomsite.com",
        subject="Your Weekly Digest is Ready",
        body="Hi there! Here's your weekly digest of top stories. Click to read more. Unsubscribe at any time.",
        timestamp="2026-04-08 09:30:00",
        difficulty="easy",
        ground_truth_category=EmailCategory.SPAM,
        ground_truth_priority=EmailPriority.LOW,
        ground_truth_should_archive=True,
        expected_action_reason="Unsolicited newsletter from unknown sender — spam."
    ),

    # ── MEDIUM ────────────────────────────────────────────────────────────────
    EmailTask(
        id="M001",
        sender="hr@company.com",
        subject="Updated Health Benefits for 2026",
        body="Please review the attached document for the updated health benefits starting next month. Action is required by Friday.",
        timestamp="2026-04-08 10:00:00",
        difficulty="medium",
        ground_truth_category=EmailCategory.NORMAL,
        ground_truth_priority=EmailPriority.NORMAL,
        ground_truth_should_archive=False,
        keywords_in_response=["acknowledge", "review", "Friday"],
        expected_action_reason="Standard company communication requiring action, but not a critical emergency."
    ),
    EmailTask(
        id="M002",
        sender="system@cloudmonitor.io",
        subject="[ALERT] Database Latency High",
        body="Warning: DB-Primary is experiencing higher than normal latency (450ms). Investigation recommended.",
        timestamp="2026-04-08 10:30:00",
        difficulty="medium",
        ground_truth_category=EmailCategory.IMPORTANT,
        ground_truth_priority=EmailPriority.HIGH,
        ground_truth_should_archive=False,
        keywords_in_response=["investigating", "incident", "latency"],
        expected_action_reason="Technical alert requiring immediate attention to prevent potential service degradation."
    ),
    EmailTask(
        id="M003",
        sender="mom@email.com",
        subject="Dinner this Sunday?",
        body="Hi honey, are you free for dinner this Sunday? Let me know!",
        timestamp="2026-04-08 11:00:00",
        difficulty="medium",
        ground_truth_category=EmailCategory.NORMAL,
        ground_truth_priority=EmailPriority.LOW,
        ground_truth_should_archive=False,
        keywords_in_response=["Sunday", "dinner", "family"],
        expected_action_reason="Personal email, should be handled but is not work-critical."
    ),
    EmailTask(
        id="M004",
        sender="it-support@company.com",
        subject="Scheduled Maintenance: Saturday 2–4 AM",
        body="Please be aware that our systems will undergo scheduled maintenance this Saturday from 2–4 AM. No action required.",
        timestamp="2026-04-08 11:30:00",
        difficulty="medium",
        ground_truth_category=EmailCategory.NORMAL,
        ground_truth_priority=EmailPriority.LOW,
        ground_truth_should_archive=True,
        keywords_in_response=["noted", "maintenance", "Saturday"],
        expected_action_reason="Informational notice — low priority, archive after reading."
    ),

    # ── HARD ──────────────────────────────────────────────────────────────────
    EmailTask(
        id="H001",
        sender="ceo@startup.io",
        subject="URGENT: Investor Presentation Deck",
        body="I need the final slides for the investor deck by 2 PM today. Please send them ASAP.",
        timestamp="2026-04-08 12:00:00",
        difficulty="hard",
        ground_truth_category=EmailCategory.IMPORTANT,
        ground_truth_priority=EmailPriority.HIGH,
        ground_truth_should_archive=True,
        keywords_in_response=["CEO", "slides", "investor", "ASAP"],
        expected_action_reason="High-priority request from an executive with a tight deadline."
    ),
    EmailTask(
        id="H002",
        sender="billing@softwarevendor.com",
        subject="Overdue Invoice #4492",
        body="Your account is 30 days past due. Please settle the balance of $1,200 to avoid service interruption.",
        timestamp="2026-04-08 13:00:00",
        difficulty="hard",
        ground_truth_category=EmailCategory.IMPORTANT,
        ground_truth_priority=EmailPriority.HIGH,
        ground_truth_should_archive=False,
        keywords_in_response=["invoice", "payment", "overdue", "billing"],
        expected_action_reason="Critical billing issue that could lead to service disruption."
    ),
    EmailTask(
        id="H003",
        sender="support@client.com",
        subject="Trouble logging in",
        body="I've tried resetting my password but haven't received the email. Can you help?",
        timestamp="2026-04-08 14:00:00",
        difficulty="hard",
        ground_truth_category=EmailCategory.NORMAL,
        ground_truth_priority=EmailPriority.NORMAL,
        ground_truth_should_archive=False,
        keywords_in_response=["support", "password", "login", "reset"],
        expected_action_reason="Customer support request requiring a helpful response and tracking."
    ),

    # ── HARD: Tricky / Adversarial ────────────────────────────────────────────
    EmailTask(
        id="H004",
        sender="it-security@company-helpdesk.net",
        subject="Action Required: Verify Your Account Immediately",
        body="We detected unusual activity on your account. Click here to verify your identity within 24 hours or your account will be suspended: http://verify-account-now.xyz",
        timestamp="2026-04-08 14:30:00",
        difficulty="hard",
        ground_truth_category=EmailCategory.SPAM,
        ground_truth_priority=EmailPriority.LOW,
        ground_truth_should_archive=True,
        expected_action_reason="Phishing email disguised as an IT security alert. Suspicious domain and urgency tactics are classic phishing indicators."
    ),
    EmailTask(
        id="H005",
        sender="john.smith@company.com",
        subject="Quick favour",
        body="Hey, I'm in a meeting and can't talk. I need you to urgently purchase 5x $200 Amazon gift cards for a client and email me the codes. I'll reimburse you later. Thanks, John (CEO)",
        timestamp="2026-04-08 15:00:00",
        difficulty="hard",
        ground_truth_category=EmailCategory.SPAM,
        ground_truth_priority=EmailPriority.LOW,
        ground_truth_should_archive=True,
        expected_action_reason="CEO gift card scam — a known social engineering attack. Despite appearing internal, the request pattern is a textbook scam."
    ),
    EmailTask(
        id="H006",
        sender="legal@external-lawfirm.com",
        subject="Notice of Pending Litigation — Confidential",
        body="This notice is to inform you that our client intends to file a civil claim against your organisation regarding breach of contract dated January 2026. Please forward to your legal counsel immediately.",
        timestamp="2026-04-08 15:30:00",
        difficulty="hard",
        ground_truth_category=EmailCategory.IMPORTANT,
        ground_truth_priority=EmailPriority.HIGH,
        ground_truth_should_archive=False,
        keywords_in_response=["legal", "counsel", "litigation", "contract"],
        expected_action_reason="Legal notice from an external law firm — must be escalated to legal counsel immediately, do not archive."
    ),
    EmailTask(
        id="H007",
        sender="devops@company.com",
        subject="Re: Re: Re: Deployment pipeline",
        body="As discussed, the pipeline has been updated. Let me know if the staging deployment looks good to you before we push to prod tonight.",
        timestamp="2026-04-08 16:00:00",
        difficulty="hard",
        ground_truth_category=EmailCategory.IMPORTANT,
        ground_truth_priority=EmailPriority.HIGH,
        ground_truth_should_archive=False,
        keywords_in_response=["staging", "deployment", "pipeline", "production"],
        expected_action_reason="Production deployment approval request — requires urgent review and response before it goes to prod."
    ),
    EmailTask(
        id="H008",
        sender="notifications@github.com",
        subject="[company/core-api] Security vulnerability found in dependency",
        body="Dependabot has found a critical security vulnerability (CVE-2024-1234, CVSS 9.8) in package requests==2.28.0 used in company/core-api. An update is available.",
        timestamp="2026-04-08 16:30:00",
        difficulty="hard",
        ground_truth_category=EmailCategory.IMPORTANT,
        ground_truth_priority=EmailPriority.HIGH,
        ground_truth_should_archive=False,
        keywords_in_response=["vulnerability", "CVE", "security", "update", "dependency"],
        expected_action_reason="Critical security vulnerability alert (CVSS 9.8) in production code — must be addressed immediately."
    ),
]

def get_tasks() -> List[EmailTask]:
    """Return the list of pre-defined tasks."""
    return TASKS
