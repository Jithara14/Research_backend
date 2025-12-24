import streamlit as st
import faiss
import json
import re
import requests
import os
import pickle
import networkx as nx
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==========================================================
# 1. CONFIGURATION
# ==========================================================
st.set_page_config(page_title="Trilingual Legal AI", layout="wide", page_icon="⚖️")

# ⚠️ UPDATE PATHS (Use raw strings r"..." for Windows paths)
BASE_DIR = r"C:\Ajanthan\llm" 
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "index.faiss")
METADATA_PATH = os.path.join(BASE_DIR, "legal_data.jsonl")
GRAPH_PATH = "legal_graph.gpickle"

# LLM Config
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# 2. LOAD MODELS
# ==========================================================
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

@st.cache_resource
def load_translator():
    # Load NLLB Model (Supports Tamil, Sinhala, English)
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
    return tokenizer, model

embed_model = load_embed_model()

# ==========================================================
# 3. LOAD DATABASES (FAISS + GRAPH)
# ==========================================================
@st.cache_resource
def load_resources():
    # A. FAISS
    index = None
    if os.path.exists(FAISS_INDEX_PATH):
        try: 
            index = faiss.read_index(FAISS_INDEX_PATH)
        except Exception as e:
            st.error(f"❌ FAISS Load Error: {e}")

    # B. Metadata 
    metadata = []
    metadata_lookup = {}
    
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try: 
                        obj = json.loads(line)
                        metadata.append(obj)
                        
                        sid = str(obj.get('section_id', obj.get('sectionid', '')))
                        heading = obj.get('heading', obj.get('title', '')) 
                        content = obj.get('content', obj.get('text', ''))
                        
                        if sid:
                            full_text = f"*{heading}*\n{content}" if heading else content
                            metadata_lookup[sid] = full_text     
                    except: 
                        pass
    
    # C. Graph
    graph = None
    if os.path.exists(GRAPH_PATH):
        with open(GRAPH_PATH, "rb") as f:
            try: 
                graph = pickle.load(f)
            except Exception as e:
                st.warning(f"⚠️ Graph Load Error: {e}")
            
    return index, metadata, metadata_lookup, graph

index, metadata, metadata_lookup, G = load_resources()

# ==========================================================
# 4. TRANSLATION FUNCTION
# ==========================================================
def is_tamil(text):
    return bool(re.search("[\u0B80-\u0BFF]", text))

def is_sinhala(text):
    return bool(re.search("[\u0D80-\u0DFF]", text))

def translate_text(text, target_lang):
    if not text or not isinstance(text, str): return str(text)

    tokenizer, model = load_translator()
    
    src_lang = "eng_Latn"
    tgt_token = "eng_Latn"

    if target_lang == "en":
        if is_sinhala(text):
            src_lang = "sin_Sinh"
            tgt_token = "eng_Latn"
        elif is_tamil(text):
            src_lang = "tam_Taml"
            tgt_token = "eng_Latn"
        else:
            return text

    elif target_lang == "ta":
        src_lang = "eng_Latn"
        tgt_token = "tam_Taml"

    elif target_lang == "si":
        src_lang = "eng_Latn"
        tgt_token = "sin_Sinh"

    try:
        tokenizer.src_lang = src_lang
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
        forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_token)
        
        with torch.no_grad():
            out = model.generate(**inputs, forced_bos_token_id=forced_bos_id, max_length=1024)
        
        return tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        return f"[Translation Error: {e}]"

# ==========================================================
# 5. RETRIEVAL (FAISS + GRAPH)
# ==========================================================
def get_graph_neighbors(section_id):
    if G is None or not section_id: return []
    related = []
    sid_str = str(section_id)
    if sid_str in G:
        try:
            neighbors = list(G.neighbors(sid_str))
            for n in neighbors:
                n_str = str(n)
                if n_str in metadata_lookup:
                    content = metadata_lookup[n_str]
                    related.append(f"[Related Law] Section {n_str}: {content}")
        except Exception:
            pass
    return related

def retrieve_hybrid_data(query, k=6):
    if index is None or not metadata: return []
    
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k)
    
    docs = []
    found_ids = set()

    if len(indices) > 0:
        for idx in indices[0]:
            if 0 <= idx < len(metadata):
                d = metadata[idx]
                sid = str(d.get('section_id', d.get('sectionid', 'N/A')))
                heading = d.get('heading', d.get('title', ''))
                content = d.get('content', d.get('text', ''))
                
                if heading:
                    docs.append(f"Section {sid} - {heading}: {content}")
                else:
                    docs.append(f"Section {sid}: {content}")
                found_ids.add(sid)

    graph_context = []
    if G:
        for sid in found_ids:
            rel = get_graph_neighbors(sid)
            graph_context.extend(rel)
    
    return list(set(docs + graph_context))

# ==========================================================
# 6. QWEN GENERATION (FIXED PROMPT FOR BETTER RECALL)
# ==========================================================
def clean_json_response(text):
    try:
        text = re.sub(r"json", "", text, flags=re.IGNORECASE)
        text = re.sub(r"", "", text)
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            json_str = text[start:end+1]
            return json.loads(json_str)
    except Exception:
        pass
    return {"Section": "Error", "Simple_Explanation": text[:500], "Example": "-", "Punishment": "-", "Next_Steps": "-"}

def generate_answer_qwen(query, context):
    # 1. IMMEDIATE FAIL IF NO CONTEXT (Keep this safety)
    if not context:
        return json.dumps({
            "Section": "Not Found",
            "Simple_Explanation": "The requested information is not mentioned in the legal database.",
            "Example": "N/A",
            "Punishment": "N/A",
            "Next_Steps": "Please consult a lawyer."
        })

    # Format context with numbered sources to help the LLM "see" them better
    formatted_context = ""
    for i, doc in enumerate(context[:15]):
        formatted_context += f"SOURCE {i+1}: {doc}\n\n"
    
    # --- UPDATED PROMPT: ENCOURAGES FINDING ANSWERS ---
    prompt = f"""
    [INST] You are a helpful Sri Lankan Legal Advisor.
    
    I have retrieved the following legal documents (SOURCES) for you. Your job is to answer the User Question using ONLY these sources.
    
    SOURCES:
    {formatted_context}
    
    USER QUESTION: 
    {query}
    
    INSTRUCTIONS:
    1. *Check Thoroughly:* Look at every SOURCE above. If any source relates to the question, use it.
    2. *Partial Matches:* If the exact answer isn't there, but a related law is, explain that related law.
    3. *Not Found:* ONLY if the sources are completely irrelevant (e.g., asking about murder but sources are about tax), then output "Not Mentioned".
    
    OUTPUT FORMAT (JSON ONLY):
    
    *If Info Found (Even Partially):*
    {{
        "Section": "Act Name / Section Number found in Source",
        "Simple_Explanation": "Explain the law simply in 4-6 sentences.",
        "Example": "A realistic example situation.",
        "Punishment": "Any fines or jail time mentioned (if none, say 'Not specified in this section').",
        "Next_Steps": "3 actionable steps."
    }}

    *If TRULY Not Found:*
    {{
        "Section": "Not Mentioned in Database",
        "Simple_Explanation": "This specific topic is not covered in the provided legal documents.",
        "Example": "N/A",
        "Punishment": "N/A",
        "Next_Steps": "Consult a lawyer."
    }}
    [/INST]
    """
    
    try:
        # Increased temperature slightly to 0.3 to prevent it from being "too rigid"
        resp = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL, 
            "prompt": prompt, 
            "stream": False,
            "format": "json", 
            "options": {"temperature": 0.3, "num_ctx": 4096}
        })
        if resp.status_code == 200:
            return resp.json().get("response", "")
        return "{}"
    except Exception:
        return "{}"

# ==========================================================
# 7. CHAT UI
# ==========================================================
st.title("⚖️ Trilingual Legal AI (Sinhala/Tamil/English)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            data = msg["content"]
            def display_response(d, lang_label):
                st.success(f"{lang_label}")
                
                # Check for "Not Found" scenario
                if "Not Mentioned" in d.get('Section', '') or "Not Found" in d.get('Section', ''):
                    st.error(f"❌ {d.get('Simple_Explanation')}")
                    st.write(f"*Advice:* {d.get('Next_Steps')}")
                else:
                    st.write(f"*📜 Section:* {d.get('Section')}")
                    st.markdown(f"*💡 Explanation:*\n{d.get('Simple_Explanation')}")
                    st.info(f"*📝 Example:*\n{d.get('Example')}")
                    st.warning(f"*⚖️ Punishment:* {d.get('Punishment')}")
                    st.write(f"*🚀 Next Steps:* {d.get('Next_Steps')}")

            if "sinhala_data" in data and data["sinhala_data"]:
                display_response(data["sinhala_data"], "🇱🇰 පිළිතුර (Sinhala)")
            elif "tamil_data" in data and data["tamil_data"]:
                display_response(data["tamil_data"], "🇱🇰 பதில் (Tamil)")
            else:
                display_response(data["english_data"], "🇬🇧 Answer (English)")

user_input = st.chat_input("Enter question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
        
    with st.chat_message("assistant"):
        status = st.status("🧠 Processing...", expanded=True)
        
        # 1. Translate Input
        is_si = is_sinhala(user_input)
        is_tam = is_tamil(user_input)
        
        if is_si or is_tam:
            status.write("🔄 Translating to English...")
            query_en = translate_text(user_input, "en")
        else:
            query_en = user_input
            
        # 2. Retrieve & Generate
        status.write("🕸️ searching Laws...")
        context_docs = retrieve_hybrid_data(query_en, k=6)
        
        status.write("🤖 Drafting Explanation...")
        raw_json = generate_answer_qwen(query_en, context_docs)
        english_data = clean_json_response(raw_json)
        
        final_obj = {"english_data": english_data, "tamil_data": None, "sinhala_data": None}
        
        # 3. Translate Output
        if is_si:
            status.write("🔄 Translating to Sinhala...")
            final_obj["sinhala_data"] = {k: translate_text(v, "si") for k, v in english_data.items()}
        elif is_tam:
            status.write("🔄 Translating to Tamil...")
            final_obj["tamil_data"] = {k: translate_text(v, "ta") for k, v in english_data.items()}
            
        status.update(label="Done!", state="complete", expanded=False)
        
        # Display logic
        def show_result(d, title):
            st.success(title)
            if "Not Mentioned" in d.get('Section', '') or "Not Found" in d.get('Section', ''):
                st.error(f"❌ {d.get('Simple_Explanation')}")
                st.write(f"*Advice:* {d.get('Next_Steps')}")
            else:
                st.write(f"*Section:* {d.get('Section')}")
                st.markdown(f"*Explanation:*\n{d.get('Simple_Explanation')}")
                st.info(f"*Example:*\n{d.get('Example')}")
                st.warning(f"*Punishment:* {d.get('Punishment')}")
                st.write(f"*Next Steps:* {d.get('Next_Steps')}")

        if is_si:
            show_result(final_obj["sinhala_data"], "🇱🇰 *පිළිතුර (Sinhala)*")
        elif is_tam:
            show_result(final_obj["tamil_data"], "🇱🇰 *பதில் (Tamil)*")
        else:
            show_result(final_obj["english_data"], "🇬🇧 *Answer (English)*")
            
        st.session_state.messages.append({"role": "assistant", "content": final_obj})