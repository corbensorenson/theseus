LABELS = {
    "info": "information",
    "error": "error",
}


def classify(level):
    return LABELS.get(level, "unknown")
