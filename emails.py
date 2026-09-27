"""
Raw "emails" (100% fake, for teaching purposes only).

These look like real emails but are entirely made up:
  - No real companies, people, or working links
  - All links point at *.example.com, which is a domain reserved by
    RFC 2606 specifically for documentation/examples (it never
    resolves to a real site)

This is the RAW data, before any processing. Nothing here is a
number yet -- it's just text, exactly like a real inbox would give
you. preprocess.py turns this into the numeric dataset the model
actually trains on.
"""

emails = [
    # ---------------- PHISHING (label = 1) ----------------
    {
        "id": "p1",
        "sender": "security-alerts@nova-bank-verify.example.com",
        "subject": "URGENT: Your account has been suspended",
        "body": """Dear Customer,

We detected unusual activity on your account. Your account has been
suspended and requires urgent verification. Click here to verify your
identity and restore access immediately:

http://nova-bank-verify.example.com/login
http://nova-bank-verify.example.com/verify-account
http://nova-bank-verify.example.com/support

If you do not verify within 24 hours, your account will remain
suspended. Act now to avoid permanent loss of access.

Nova Bank Security Team""",
        "label": 1,
    },
    {
        "id": "p2",
        "sender": "no-reply@quickpay-rewards.example.com",
        "subject": "Congratulations! You are our lucky winner",
        "body": """Congratulations!

You have been selected as the winner of our monthly cash prize.
To claim your reward, please verify your bank account details using
the secure link below within 48 hours:

http://quickpay-rewards.example.com/claim
http://quickpay-rewards.example.com/verify

This is a limited time offer, act now before it expires!

QuickPay Rewards Team""",
        "label": 1,
    },
    {
        "id": "p3",
        "sender": "it-support@corp-mailsystem.example.com",
        "subject": "Security Alert: Password reset required",
        "body": """Dear Employee,

Our system detected a security alert on your account. Your password
has expired and must be reset urgently to avoid suspension of your
mailbox. Click the link below to verify your credentials and reset
your password:

http://corp-mailsystem.example.com/reset-password
http://corp-mailsystem.example.com/verify-account
http://corp-mailsystem.example.com/help
http://corp-mailsystem.example.com/policy

Failure to verify will result in your account being suspended.

IT Support""",
        "label": 1,
    },
    {
        "id": "p4",
        "sender": "billing@fasttrack-delivery.example.com",
        "subject": "Action required: Confirm your delivery payment",
        "body": """Hello,

Your package delivery is on hold. We could not process payment on
your account. Please click here to confirm your payment details and
verify your address urgently:

http://fasttrack-delivery.example.com/confirm-payment
http://fasttrack-delivery.example.com/verify

Act now, your package will be returned if not confirmed within 24
hours.

Fast Track Delivery""",
        "label": 1,
    },
    {
        "id": "p5",
        "sender": "accounts@securepay-alert.example.com",
        "subject": "Your account will be suspended - verify now",
        "body": """Dear Customer,

This is an urgent security alert. We noticed a login attempt from an
unrecognized device on your bank account. For your protection, your
account access has been temporarily suspended.

Click here immediately to verify your identity:
http://securepay-alert.example.com/verify-now

SecurePay Team""",
        "label": 1,
    },
    {
        "id": "p6",
        "sender": "rewards@luckydraw-winners.example.com",
        "subject": "FINAL NOTICE: Claim your prize before it expires",
        "body": """Congratulations Winner!

This is your final notice. Your prize is about to expire! Click
below urgently to confirm your details and claim your reward before
the limited time offer ends:

http://luckydraw-winners.example.com/claim-prize
http://luckydraw-winners.example.com/verify-details
http://luckydraw-winners.example.com/terms
http://luckydraw-winners.example.com/contact
http://luckydraw-winners.example.com/unsubscribe

Act now! Winner Support Team""",
        "label": 1,
    },

    # ---------------- SAFE (label = 0) ----------------
    {
        "id": "s1",
        "sender": "hr@acmecorp.example.com",
        "subject": "Reminder: Team meeting moved to 3 PM",
        "body": """Hi team,

Just a quick reminder that tomorrow's team meeting has been moved
from 2 PM to 3 PM. Same conference room. Let me know if that time
doesn't work for anyone.

Thanks,
Priya""",
        "label": 0,
    },
    {
        "id": "s2",
        "sender": "no-reply@devtools.example.com",
        "subject": "Your weekly build summary",
        "body": """Hi Kaushik,

Here is your weekly build summary for the project repository. 3
builds passed, 0 failed. Full details are available on the
dashboard: http://devtools.example.com/dashboard

Have a great week!
DevTools Bot""",
        "label": 0,
    },
    {
        "id": "s3",
        "sender": "raj.mentor@acmecorp.example.com",
        "subject": "Notes from today's code review",
        "body": """Hey,

Good progress on the login module today. A couple of small things
to clean up before merging, otherwise looks solid. Let's sync again
on Thursday.

Cheers,
Raj""",
        "label": 0,
    },
    {
        "id": "s4",
        "sender": "newsletter@techweekly.example.com",
        "subject": "This week in web development",
        "body": """Hello subscriber,

Here's this week's roundup of web development news, including a
new JavaScript framework release and a deep dive into CSS grid
layouts. Read the full newsletter here:

http://techweekly.example.com/issue-42

See you next week!""",
        "label": 0,
    },
    {
        "id": "s5",
        "sender": "events@localdevmeetup.example.com",
        "subject": "You're registered for Saturday's meetup",
        "body": """Hi Kaushik,

Thanks for registering for Saturday's local developer meetup. Doors
open at 10 AM. Feel free to bring a friend.

See you there!
Meetup Organizers""",
        "label": 0,
    },
    {
        "id": "s6",
        "sender": "billing@cloudhost.example.com",
        "subject": "Your monthly invoice is ready",
        "body": """Hi Kaushik,

Your invoice for this billing cycle is now available in your
account dashboard. No action is needed if you're on auto-pay.

Thanks for being a customer,
CloudHost Billing""",
        "label": 0,
    },
]
