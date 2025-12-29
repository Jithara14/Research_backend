import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HEADLINE_STORE = "current_headlines.txt"

# -------------------------------
# Load models (same as your code)
# -------------------------------
embed_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indicbart")
headline_model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indicbart")

THRESHOLD = 0.85


# -------------------------------
# Helper functions (UNCHANGED)
# -------------------------------
def is_headline(text):
    text = text.strip()
    if len(set(text)) <= 3:
        return False
    return len(text.split()) <= 8 and "." not in text and "。" not in text


def get_first_clean_para(group):
    for p in group:
        p = p.strip()
        if len(p) > 15 and len(set(p)) > 5:
            return p
    return group[0]


def is_mostly_text(text, min_ratio=0.6):
    tamil_chars = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')
    return tamil_chars / max(len(text), 1) >= min_ratio


def generate_headline_safe(group):
    first_para = group[0].strip()
    if is_headline(first_para):
        return first_para

    text = " ".join(group)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    inputs.pop("token_type_ids", None)

    outputs = headline_model.generate(
        **inputs,
        max_new_tokens=15,
        do_sample=False
    )

    headline = tokenizer.decode(
        outputs[0], skip_special_tokens=True
    ).strip()

    if (
        not headline or
        "<extra_id" in headline or
        len(set(headline)) <= 2 or
        not is_mostly_text(headline)
    ):
        return get_first_clean_para(group)

    words = headline.split()
    if len(words) > 8:
        headline = " ".join(words[:8])
    if len(words) < 4:
        headline = get_first_clean_para(group).split("।")[0]

    return headline


# -------------------------------
# MAIN FUNCTION
# -------------------------------
def extract_headings_and_groups(text: str):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    embeddings = embed_model.encode(
        paragraphs, convert_to_tensor=True
    )

    groups = []
    current_group = [paragraphs[0]]
    current_emb = embeddings[0]

    for i in range(1, len(paragraphs)):
        sim = util.cos_sim(embeddings[i], current_emb).item()

        if sim >= THRESHOLD:
            current_group.append(paragraphs[i])
            current_emb = torch.mean(
                torch.stack([current_emb, embeddings[i]]), dim=0
            )
        else:
            groups.append(current_group)
            current_group = [paragraphs[i]]
            current_emb = embeddings[i]

    groups.append(current_group)

    results = []
    headlines = []

    for group in groups:
        title = generate_headline_safe(group)
        headlines.append(title)
        results.append({
            "headline": title,
            "content": group
        })

    # 🔹 SAVE HEADLINES IMMEDIATELY
    with open(HEADLINE_STORE, "w", encoding="utf-8") as f:
        for h in headlines:
            f.write(h + "\n")

    return results
