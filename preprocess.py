"""
Data Preprocessing: Raw Emails -> Numeric Feature Dataset
============================================================

This is the missing step between "real looking emails" and the
dataset the model actually trains on. It's done with plain string
operations only -- no regex, no NLP libraries -- so every step is
visible and explainable.

PIPELINE
--------
raw email text  --->  lowercase + clean  --->  count features  --->  [x1, x2, label]

FEATURE 1: num_suspicious_words
    How many times any word/phrase from SUSPICIOUS_PHRASES appears
    in the email (subject + body combined), case-insensitive.

FEATURE 2: num_links
    How many times "http://" or "https://" appears in the email.
    (A stand-in for a real link-extractor -- keeps things dependency-free.)
"""

from emails import emails

# A hand-picked list of words/phrases that commonly show up in
# phishing emails. This is the kind of "domain knowledge" a human
# would encode before any ML happens.
SUSPICIOUS_PHRASES = [
    "urgent",
    "verify",
    "click here",
    "act now",
    "suspended",
    "winner",
    "congratulations",
    "limited time",
    "security alert",
    "confirm your",
    "bank account",
    "final notice",
    "claim your",
]


def clean_text(subject, body):
    """Lowercase and merge subject + body into one string to scan."""
    return (subject + " " + body).lower()


def count_suspicious_words(text):
    """Counts total occurrences of any suspicious phrase in the text."""
    count = 0
    for phrase in SUSPICIOUS_PHRASES:
        count += text.count(phrase)
    return count


def count_links(text):
    """Counts how many links appear, by counting 'http' occurrences."""
    return text.count("http://") + text.count("https://")


def email_to_features(email):
    """Converts one raw email dict into a numeric [x1, x2, label] row."""
    text = clean_text(email["subject"], email["body"])
    x1 = count_suspicious_words(text)
    x2 = count_links(text)
    return [x1, x2, email["label"]]


# The final dataset, built by running every raw email through the
# pipeline above. THIS is what phishing_classifier_scratch.py trains on.
dataset = [email_to_features(e) for e in emails]


if __name__ == "__main__":
    print(f"{'ID':4s} {'Label':7s} {'Suspicious words':18s} {'Links':6s} Subject")
    print("-" * 80)
    for email, row in zip(emails, dataset):
        x1, x2, y = row
        label_str = "PHISHING" if y == 1 else "SAFE"
        print(f"{email['id']:4s} {label_str:7s} {x1:<18d} {x2:<6d} {email['subject'][:45]}")

    print("\nResulting numeric dataset (this feeds directly into the model):\n")
    for row in dataset:
        print(row)
