HEADLINE_STORE = "current_headlines.txt"

def load_headlines():
    try:
        with open(HEADLINE_STORE, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines
    except FileNotFoundError:
        return []
