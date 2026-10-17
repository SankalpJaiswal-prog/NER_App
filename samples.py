"""Demo texts for the app. These are illustrative sentences, not real news."""

SAMPLES = {
    "News": (
        "Sundar Pichai met officials from Google India in Hyderabad on Monday. "
        "Later in the week he travelled to Mumbai to meet representatives of the "
        "Reserve Bank of India and Tata Consultancy Services."
    ),
    "Corporate filing": (
        "The Board of Directors of Reliance Industries Limited appointed Anil Sharma as an "
        "independent director. The registered office of the company is in Navi Mumbai, and "
        "its statutory auditors are Deloitte Haskins & Sells."
    ),
    "Tricky (ambiguous names)": (
        "Washington signed the order in Washington after meeting Amazon executives. "
        "Jordan visited Jordan to open a new Apple store."
    ),
}


def _add_team_paragraphs():
    """Add the team's 18 news paragraphs (data/samples.csv) to the sample picker as 'News P1'... 'News P18'."""
    import csv
    import os

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "samples.csv")
    if not os.path.exists(path):
        return
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            SAMPLES[f"News {row['sample_id']}"] = row["text"]


_add_team_paragraphs()
