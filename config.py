"""App settings. Samrudhi: send your final colours as hex values and replace COLORS below."""

APP_TITLE = "Named Entity Recognition App"
APP_SUBTITLE = "Find people, organizations and places in any text"

# One highlight colour per entity type (placeholder values until the design is final).
COLORS = {
    "PERSON": "#FFD166",
    "ORG": "#8EC9F5",
    "LOC": "#9AE6B4",
}

LABEL_NAMES = {
    "PERSON": "Person",
    "ORG": "Organization",
    "LOC": "Location",
}
