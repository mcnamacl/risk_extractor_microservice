import uuid
import hashlib
import json
import time
import pymupdf
import os
import sys
from pathlib import Path
import torch
import re
import numpy as np

from sklearn.cluster import KMeans

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from sentence_transformers import SentenceTransformer, util

import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab')

import spacy

from typing import Optional

os.environ["TRANSFORMERS_CACHE"] = "./cache"  # Set cache directory

nlp = spacy.load("en_core_web_sm")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit loading
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  
    bnb_4bit_use_double_quant=True 
)

# Base model name
base_model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"  

def load_base():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    return AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

def load_trained(adapter_path: str):
    base = load_base()
    return PeftModel.from_pretrained(
        base,
        adapter_path,
        inference_mode=False,
    )

tokenizer = AutoTokenizer.from_pretrained(base_model_name, model_max_length=1848)
tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def decompose_query(query: str) -> Optional[str]:
    doc = nlp(query.lower())
    for ent in doc.ents:
        if ent.label_ == "DATE":
            match = re.search(r'\b(20\d{2})\b', ent.text)
            if match:
                return int(match.group(1))
    match = re.search(r'(fiscal year|end(ing)?|in|for|year)?\s*(20\d{2})', query.lower())
    if match:
        return str(int(match.group(3)))
    raise ValueError(f"No valid year found in query: {query}")

START_PATTERN = re.compile(
    r"(item\s+1a\.?\s+risk\s+factors)",
    re.IGNORECASE
)

END_PATTERN_10Q = re.compile(
    r"(item\s+1b\.?\s+unresolved\s+staff\s+comments)",
    re.IGNORECASE
)

def normalise(text):
    text = text.replace("\u00a0", " ")   # non-breaking spaces
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text

def extract_risk_factors_10q(text):
    text = normalise(text)

    text = text[10000:] # Ignore table of contents.

    # matches = list(START_PATTERN.finditer(text))
    start = START_PATTERN.search(text)
    if not start:
        raise ValueError("Item 1A not found")
    
    end = END_PATTERN_10Q.search(text, start.end())
    if not end:
        raise ValueError("Item 2 not found")

    return text[start.end():end.start()]

def extract_risk_section(document_path: str) -> str:
    with pymupdf.open(document_path) as doc:
        text = chr(12).join([page.get_text() for page in doc])
    print(f"Extracting risk factors from document at: {document_path}")
    risk_factors_section = extract_risk_factors_10q(text)
    return risk_factors_section

def chunk_text(text: str, max_tokens: int = 1500, top_k_clusters: int = 4) -> list:
    """
    Uses unsupervised sentence clustering + weak supervision
    to identify and chunk risk-dense content.
    """

    # 1. Sentence split
    sentences = nltk.sent_tokenize(text)
    print(f"Number of sentences created: {len(sentences)}")
    if len(sentences) < 10:
        return [text]

    # 2. Precompute embeddings and token counts
    sentence_embeddings = sbert_model.encode(sentences)
    sentence_token_counts = [len(tokenizer.encode(s)) for s in sentences]

    # 3. Cluster sentences (stable bounds)
    num_clusters = max(4, min(6, len(sentences) // 4))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(sentence_embeddings)

    # 4. Risk anchor (semantic prior, still unsupervised)
    risk_anchors = [
        sbert_model.encode("risk factor threat vulnerability exposure"),
        sbert_model.encode("uncertainty may might could potentially"),
        sbert_model.encode("negative adverse impact damage loss"),
        sbert_model.encode("regulatory legal compliance violation")
    ]
    cluster_scores = []

    for cluster_id in range(num_clusters):
        idxs = [i for i, c in enumerate(cluster_labels) if c == cluster_id]
        if not idxs:
            cluster_scores.append(0.0)
            continue

        cluster_text = " ".join(sentences[i] for i in idxs).lower()
        cluster_emb = np.mean(sentence_embeddings[idxs], axis=0)

        # Lexical heuristics
        negation = sum(cluster_text.count(w) for w in ['not', 'no', 'never', 'without', 'lack', 'fail'])
        uncertainty = sum(cluster_text.count(w) for w in ['may', 'might', 'could', 'would', 'should', 'possibly'])
        consequence = sum(cluster_text.count(w) for w in ['lead to', 'result in', 'cause', 'impact', 'effect',
                                                          'consequence', 'outcome', 'damage', 'loss'])
        modal = sum(cluster_text.count(w) for w in ['must', 'shall', 'will', 'should', 'would', 'could', 'might', 'may'])

        lexical_score = (
            min(negation / len(idxs), 2) * 0.25 +
            min(uncertainty / len(idxs), 2) * 0.25 +
            min(consequence / len(idxs), 2) * 0.25 +
            min(modal / len(idxs), 3) * 0.15
        )

        # Semantic similarity to risk anchor
        semantic_score = max(
            util.cos_sim(cluster_emb, anchor).item() 
            for anchor in risk_anchors
        )

        cluster_scores.append(lexical_score + 0.35 * semantic_score)

    # 5. Select top-k clusters
    top_clusters = np.argsort(cluster_scores)[-top_k_clusters:]

    # 6. Collect risky sentences
    risky_indices = [i for i, c in enumerate(cluster_labels) if c in top_clusters]
    risky_indices.sort()
    risky_sentences = [(sentences[i], sentence_token_counts[i]) for i in risky_indices]

    # 7. Chunk with token constraints

    chunks = []
    current_chunk = ""

    for sent in risky_sentences:
        candidate = current_chunk + (" " if current_chunk else "") + sent[0]
        candidate_tokens = len(tokenizer.encode(candidate))

        if candidate_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sent[0]
        else:
            current_chunk = candidate

    if current_chunk:
        chunks.append(current_chunk)
    
    print(f"Number of chunks created: {len(chunks)}")
    return chunks

def extract_risks_from_chunks(chunks: list, adapter_path="") -> list:
    BATCH_SIZE = 4
    all_risks = []

    dir_path = os.path.dirname(os.path.realpath(__file__))

    doc_path = f"{dir_path}/{adapter_path}" 
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE}: {len(batch)} chunks")
        
        # Process batch through GPU model
        decoded_outputs = batch_generate(batch, adapter_path=doc_path)
        
        # Parse outputs for risks
        for decoded in decoded_outputs:
            batch_risks = parse_output(decoded)
            all_risks.extend(batch_risks)
        
        print(f"Extracted {len(batch_risks)} risks from current batch")
    
    print(f"Total risks extracted: {len(all_risks)}")
    return all_risks

def parse_output(decoded: str) -> list:
    risks = []
    try:
        # Find the output part after "output: "
        output_part = decoded.split("output: ", 1)[-1].strip()
        
        # Split on "|" for categories and summary
        parts = output_part.split("|", 1)
        if len(parts) != 2:
            raise ValueError("Invalid format: No '|' separator found.")
        
        categories_str = parts[0].strip()
        summary = parts[1].strip()

        if summary.count('[') > 1:
          summary = summary.split("[", 2)[0]
        
        # Parse categories as list (safe eval or json.loads)
        categories = json.loads(categories_str.replace("'", '"'))  # Convert single quotes to double for JSON
        
        risks.append({
            "risk_categories": categories,
            "risk_summary": summary,
            "output": output_part
        })
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Parsing error: {e} for output: {decoded}")
        return []  
    
    return risks

def batch_generate(chunks: list, adapter_path="", max_new_tokens=250,) -> list:
    model = load_trained(adapter_path)
    prompts = [f"Extract financial risks. Output as [CATEGORIES] | SUMMARY input: {chunk} output: " for chunk in chunks]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad(), torch.inference_mode():  # Double for safety
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=1.0  # Greedy decoding
        )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded
    
def merge_risks(all_risks: list) -> list:
    merged = []
    for risk in all_risks:
        summary_emb = sbert_model.encode(risk['risk_summary'])
        similar = next((m for m in merged if util.cos_sim(summary_emb, sbert_model.encode(m['risk_summary'])) > 0.85), None)
        if similar:
            # Merge categories (union)
            similar['risk_categories'] = list(set(similar['risk_categories'] + risk['risk_categories']))
        else:
            merged.append(risk)
    return merged

def validate_and_add_provenance(merged_risks: list, source_doc: str, query_ts: int, user_id: str, model: str) -> dict:
    print(f"Validating merged risks: {merged_risks}")
    # Validate JSON (simple schema check)
    if not all(isinstance(r, dict) and 'risk_categories' in r and 'risk_summary' in r and "output" in r for r in merged_risks):
        raise ValueError("Invalid risk format")
    
    generation_ts = int(time.time())
    audit_id = str(uuid.uuid4())
    lineage = json.dumps([{"step": "query_decompose", "ts": query_ts}, {"step": "risk_extract", "ts": generation_ts}])
    output_json = {
        "risks": merged_risks,
        "source_document": source_doc,
        "query_timestamp": query_ts,
        "generation_timestamp": generation_ts,
        "user_id": user_id,
        "model_used": model,
        "audit_id": audit_id,
        "data_lineage": lineage,
        "compliance_tags": ["GDPR-compliant", "Basel-III-aligned"]  # Banking-specific
    }
    integrity_hash = hashlib.sha256(json.dumps(output_json).encode()).hexdigest()
    output_json["integrity_hash"] = integrity_hash
    return output_json

if __name__ == "__main__":
    dir_path = "/content/drive/MyDrive/risk_extractor_microservice/documents"

    adapter_path = Path("/content/drive/MyDrive/risk_extractor_microservice/LLM_Model_Test_v8_gretel")


    doc_path = f"/corp-10k-2020.pdf"  
    fname = dir_path + doc_path

    print(extract_risk_section(fname))
    text = extract_risk_section(fname)
    print(text[:100])
    chunks = chunk_text(text)
    print(chunks)
    print(len(chunks))
    risks = extract_risks_from_chunks(chunks)
    print(risks)