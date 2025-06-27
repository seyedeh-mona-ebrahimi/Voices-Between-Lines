import numpy as np
import re
import torch
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from scipy.spatial.distance import jensenshannon
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#from Evaluation.OurAlgorithm import *
from nltk import ngrams
from Evaluation.cocnmf import full_training_loop
from collections import Counter, defaultdict
import libvoikko  # for Finnish lemmatizatio
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import libvoikko
from scipy.spatial.distance import jensenshannon
from collections import defaultdict
from typing import Dict, List, Tuple
from wordcloud import WordCloud


# --- Configuration ---

seed_words = [
    "terapeutti",
    "negatiivisuus",
    "depressio",
    "paniikkikohtaus",
    "terapeutti",
    "tunne",
    "positiivisuus",
    "onnellisuus",
    "motivaatio",
    "tasapaino",
    "selkeys",
    "mielenrauha",
    "itsetunto",
    "itseluottamus",
    "ahdistus",
    "masennus",
    "yksinäisyys",
    "epävarmuus",
    "pelko",
    "stressitaso",
    "stressihäiriö",
    "stressitön",
    "stressihormo",
    "mielenterveysongelma",
    "skitsofreenikko",
    "terapia",
    "psykoterapia",
    "lääkäri",
    "neuvonta",
    "tuki",
    "keskustelu",
    "ystävät",
    "perhe",
    "epätoivo",
    "viha",
    "pelko",
    "häpeä",
    "turhautuminen",
    "ahdistuskohtaus",
    "itku",
    "toipuminen",
    "paraneminen",
    "edistyminen",
    "itsehoito",
    "rentoutuminen",
    "hengitellä",
    "chillaa",
    "voimaantuminen",
    "neuvo",
    "jutella",
    "kipu",
    "hullu",
    "hoito",
    "ilo",
    "ärsyttävä",
    "pelastua",
    "heikkous",
    "huume",
    "lepo",
    "odotus",
    "kriisitila",
    "epäsosiaalinen",
    "yhteisöllisyys",
    "psyko",
    "huolestua",
    "väsymys",
    "neuropsykologia",
    "arvostelukyky",
    "ihmissuhde",
    "terveydellinen",
    "Asperger",
    "kuunnella",
    "itsemurha",
    "käyttäytymishäiriö",
    "autismi",
    "ahdistuneisuushäiriö",
    "paniikki",
    "eristäytyminen",
    "yksinäisyys",
    "adhd",
    "anoreksia",
    "bulimia",
    "mielenterveyshäiriö",
    "hallusinaatio",
    "harhaluulo",
    "luottaa",
    "väärinkäyttö",
    "kannabis",
    "väkivalta",
    "lihavuus",
    "kärsimys",
    "itsetuhoisuus",
    "sairaus",
    "paniikkihäiriö",
    "ahdistuneisuushäiriö",
    "ocd",
    "depressio",
    "tarkkaavaisuushäiriö",
    "syömishäiriöt",
    "skitsofrenia",
    "hallusinaatioita",
    "ptsd",
    "toivoton",
    "huolestunut",
    "surullinen",
    "tukea",
    "apua",
    "uupumus",
    "trauma",
    "paniikkikohtaus",
    "psykoterapia",
    "jooga",
    "meditaatio",
    "psykiatria",
    "sairaalahoito",
    "itsehoito",
    "kriisiapu",
    "kriisipuhelin",
    "tukiryhmä"
    "serotoniinivajaus",
    'addiktio',
    'alakulo',
    'burnout',
    'elämänhallinta',
    'erot',
    'häirintä',
    'häpeä',
    'ihmissuhteet',
    'itseinho',
    'itsemyötätunto',
    'itsetuhoisuus',
    'julkisuus',
    'jännittäminen',
    'kateus',
    'kehonkuva',
    'kiitollisuus',
    'kiusaaminen',
    'lapsettomuus',
    'maskaaminen',
    'mielenterveys',
    'multimodaalisuus',
    'päihteet',
    'stigma',
    'stressi',
    'syrjäytyminen',
    'syömishäiriö',
    'terapia',
    'ulkonäkö',
    'uupumus',
    'vaikuttaminen',
    'vertaistuki',
    'vihapuhe',
    'vuorovaikutus',
    'yksinäisyys',
    'ylikontrolli',
    'ylisuorittaminen'
    "tunteet",
    "mindstorm"
    "rakkaus",
    "surumielisyys",
    "onnellisuus",
    "ilonaihe",
    "pettymys",
    "empatia",
    "psykoosi",
    "epäluuloisuus",
    "psyykkinen trauma",
    "ajatushäiriö",
    "persoonallisuushäiriö",
    "psykopaattisuus",
    "kehotietoisuus",
    "itsetutkiskelu",
    "hyvinvointivalmennus",
    "terapiasuositus",
    "elämäntapamuutos",
    "resilienssi",
    "rentoutumistekniikat",
    "itsesäätely",
    "stressinhallinta",
    "tunnereaktio",
    "perhesuhteet",
    "ryhmäterapia",
    "suhdeongelmat",
    "aivokemia",
    "unihäiriöt",
    "aivotoiminta",
    "fysiologinen",
    "ensiapu",
    "hätäapu",
    "voimaannuttaminen",
    "skitsoaffektiivisesta",
    "kriisi",
    "ahdistunut",
    "depressiivinen",
    "rinkirunkkaukset",
    "Ahdistus",
    "masennus",
    "paniikkihäiriö",
    "syömishäiriö",
    "ulkonäkö",
    "itsetunto"
]
SEED_WORDS= set(seed_words)
STOPWORDS_PATH = './stopwords-fi.txt'
#TOPIC_FILE = "Evaluation/baseline_org.txt"
TOPIC_FILE= "Evaluation/top40_docs.txt"
MODEL_NAME= "Finnish-NLP/Ahma-3B-Instruct"
#MODEL_NAME= "Finnish-NLP/Ahma-7B-Instruct"
LAMBDA_1 = 0.2
LAMBDA_2 = 0.2
TOP_K_LABELS = 5
TOP_N_DOCS = 40
#NGRAM_RANGE = (2, 4)
MIN_FREQ = 2
NGRAM_RANGE = (1, 3)
TOP_K_CANDIDATES = 30 


# === Step 1: Load stopwords ===
with open(STOPWORDS_PATH, encoding='utf-8') as f:
    STOPWORDS = set(f.read().strip().splitlines())

# === Step 2: Parse topic file ===
import re

def parse_topics_and_docs(path):
    """
    Returns:
        topics : {topic_id: [keyword, …]}
        docs   : {topic_id: [doc_text, …]}
    """
    topic_pat   = re.compile(r"^Topic #(\d+):\s*(.+)")
    docs_pat    = re.compile(r"^Top 40 documents for Topic #(\d+):")
    docline_pat = re.compile(r"^Document #\d+:\s*(.+)")

    topics, docs = {}, {}
    current_topic = None          # only used while collecting docs

    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue

            # -------- topic header --------
            m = topic_pat.match(line)
            if m:
                tid   = int(m.group(1))
                kws   = [k.strip() for k in m.group(2).split(" - ")]
                topics[tid] = kws
                continue                               # keep scanning

            # -------- start of a doc block --------
            m = docs_pat.match(line)
            if m:
                current_topic = int(m.group(1))        # <- **reset here**
                docs[current_topic] = []
                continue

            # -------- individual doc lines --------
            if current_topic is not None:
                m = docline_pat.match(line)
                if m:
                    docs[current_topic].append(m.group(1).strip())

    return topics, docs


topics, topic_docs = parse_topics_and_docs(TOPIC_FILE)




def extract_label_candidates(top_docs: list[str],
                             topic_keywords: set[str],
                             ngram_range: tuple[int, int] = NGRAM_RANGE,
                             top_k: int = TOP_K_CANDIDATES) -> list[str]:
    """
    1. TF-IDF rank n-grams in `top_docs`.
    2. Keep only n-grams that contain ≥1 token that is in `topic_keywords`
       (the ‘anchoring’ step requested by the supervisor).
    3. Return up to `top_k` candidates sorted by TF-IDF mass.
    """
    # --- TF-IDF over the concatenated top documents ----------------
    vec = TfidfVectorizer(ngram_range=ngram_range,
                          min_df=1, max_df=1.0,
                          lowercase=True)
    tfidf = vec.fit_transform(top_docs)
    feats = vec.get_feature_names_out()
    scores = tfidf.sum(axis=0).A1
    scored = sorted(zip(feats, scores), key=lambda z: -z[1])

    # --- filter by topic anchor ------------------------------------
    # anchored = [ng for ng, _ in scored
    #             if set(ng.split()) & topic_keywords]
    anchored = [
    ng for ng, _ in scored
    if (set(ng.split()) & topic_keywords)                # ← keyword anchor
    and (set(ng.split()) & SEED_WORDS) ]                 # ← minority / seed anchor

    return anchored[:top_k]


def score_labels(candidates: list[str],
                 docs: list[str],
                 topic_keywords: set[str],
                 seed_words: set[str],
                 λ1: float = LAMBDA_1,
                 λ2: float = LAMBDA_2,
                 k_out: int = TOP_K_LABELS) -> list[str]:
    """
    • Informativeness: frequency of label tokens *inside topic-specific zones only*.
      We approximate the ‘zones’ by counting tokens that are themselves in
      `topic_keywords`. (A stricter per-token topic assignment would need
      the word-to-topic allocations from the model; this proxy is quick.)
    • Phraseness: length of the n-gram (1, 2, 3).
    • Seed overlap: # lemma overlaps with `seed_words`.
    Returns the `k_out` best labels, sorted descending by score.
    """
    # --- build a topic-specific bag-of-words -----------------------
    topic_bow = Counter()
    for doc in docs:
        for tok in doc.lower().split():
            if tok in topic_keywords:
                topic_bow[tok] += 1
    topic_token_total = sum(topic_bow.values()) + 1e-6

    # --- score -----------------------------------------------------
    label_scores = {}
    for lab in candidates:
        toks = lab.split()
        # 1) informativeness
        freq = sum(topic_bow.get(t, 0) for t in toks)
        inform = freq / topic_token_total

        # 2) phraseness (longer == slightly better)
        phrase_len = min(len(toks), 3)             # cap at 3 so 4-grams not crazy
        phrase_score = phrase_len

        # 3) seed overlap (lemma / lower-case match)
        overlap = len(set(toks) & SEED_WORDS)

        label_scores[lab] = inform + λ1*phrase_score + λ2*overlap

    ranked = sorted(label_scores, key=label_scores.get, reverse=True)
    return ranked[:k_out]           # <- no padding
    

def score_labels_with_components(
    candidates: list[str],
    docs: list[str],
    topic_keywords: set[str],
    seed_words: set[str],
    λ1: float = LAMBDA_1,
    λ2: float = LAMBDA_2,
    k_out: int = TOP_K_LABELS
) -> tuple[list[str], dict[str, dict[str, float]]]:
    topic_bow = Counter()
    for doc in docs:
        for tok in doc.lower().split():
            if tok in topic_keywords:
                topic_bow[tok] += 1
    topic_token_total = sum(topic_bow.values()) + 1e-6

    label_scores = {}
    for lab in candidates:
        toks = lab.split()
        freq = sum(topic_bow.get(t, 0) for t in toks)
        inform = freq / topic_token_total
        phrase_len = min(len(toks), 3)
        overlap = len(set(toks) & SEED_WORDS)

        score = inform + λ1 * phrase_len + λ2 * overlap
        label_scores[lab] = {
            "total_score": score,
            "informativeness": inform,
            "phraseness": λ1 * phrase_len,
            "seed_overlap": λ2 * overlap
        }

    ranked = sorted(label_scores.items(), key=lambda x: x[1]["total_score"], reverse=True)
    top_labels = [label for label, _ in ranked[:k_out]]
    return top_labels, {label: label_scores[label] for label in top_labels}
def plot_label_scores(label_score_dict: dict[str, dict[str, float]]):
    labels = list(label_score_dict.keys())
    inform = [label_score_dict[l]["informativeness"] for l in labels]
    phrase = [label_score_dict[l]["phraseness"] for l in labels]
    overlap = [label_score_dict[l]["seed_overlap"] for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x, inform, label='Informativeness')
    plt.bar(x, phrase, bottom=inform, label='Phraseness')
    bottom2 = [i + p for i, p in zip(inform, phrase)]
    plt.bar(x, overlap, bottom=bottom2, label='Seed Overlap')

    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel("Total Score")
    plt.title("Label Score Components")
    plt.legend()
    plt.tight_layout()
    plt.show()


def build_prompt(keywords, doc_snippets, candidates):
    k_string = ", ".join(keywords[:8])
    d_string = "\n\n".join(f"- {d}" for d in doc_snippets[:2])
    c_string = "\n".join(f"- {c}" for c in candidates[:5])
    prompt = (

        ############PROMPT################################################
        f"Alla on keskustelun avainsanoja, katkelmia ja koneellisesti ehdotettuja etikettejä.\n\n"
        f"AVAINSANAT:\n{k_string}\n\n"
        f"KESKUSTELUKATKELMAT:\n{d_string}\n\n"
        f"EHDOTETUT ETIKETIT:\n{c_string}\n\n"
        "Nämä ehdotukset voivat auttaa sinua ymmärtämään keskustelun teemaa, mutta sinun ei tarvitse valita niistä suoraan.\n"
        "Perustele valintasi viittaamalla sekä keskustelukatkelmiin että ehdotuksiin, jos niistä on hyötyä.\n\n"
        "Anna lyhyt ja ytimekäs etiketti, joka tiivistää keskustelun pääaiheen.\n\n"
        "Vastaa täsmälleen seuraavassa muodossa:\n"
        "ETIKETTI: [kirjoita tähän lyhyt etiketti]\n"
        "PERUSTELU: [perustele etiketti keskustelun sisällön perusteella]"
        "Älä toista yllä olevia avainsanoja tai ehdokaslistaa sellaisenaan."

    )
    return prompt

def explan_llm(prompt, model, tokenizer, topic_id):
    # Encode prompt
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]         # ←  add this line

    # Make sure the model has a valid pad token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    # Generate
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,             # ←  pass it here
        max_new_tokens=300,
        do_sample=False,
        # temperature=0.4,
        # top_k=10,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Logging (unchanged)
    with open("llm_debug_log.txt", "a") as f:
        f.write(f"\n\nPrompt for topic {topic_id}:\n{prompt}")
        f.write(f"\nRaw LLM Output:\n{response}\n")

    # Return text after “VASTAUS:” if present
    if "VASTAUS:" in response:
        return response.split("VASTAUS:")[-1].strip()
    return response.strip()
def postprocess(resp: str) -> str:
    m = re.search(r"ETIKETTI\s*:\s*(.+?)\s*PERUSTELU\s*:\s*(.+)", resp, re.S | re.I)
    if m:
        return f"ETIKETTI: {m.group(1).strip()}\nPERUSTELU: {m.group(2).strip()}"
    return resp.strip()
def call_llm(user_message,
             max_new_tokens=300,
             temperature=0.9,
             top_p=0.85):
    # 1) build chat prompt the way Ahma expects
    #00 file promt
    # messages = [
    #     {"role": "system", "content": "Olet avulias ja ytimekäs avainsanalähtöinen tiivistäjä."},
    #     {"role": "user",   "content": user_message},
    # ]
    #01 file promt
    messages= [
        {"role": "system", "content": "Olet lyhyt ja ytimekäs tekstiluokittelija, joka palauttaa vain seuraavan muodon:\nETIKETTI: ...\nPERUSTELU: ..."}, 
        {"role": "user",   "content": user_message}]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    # 2) generate
    gen_ids = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # 3) decode **only the new tokens**
    generated_text = tokenizer.decode(gen_ids[0][prompt_ids.size(1):],
                                      skip_special_tokens=True)
    return generated_text.strip()


# === Run Steps 4–6 ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, legacy=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto")
device      = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
# ───────────────────────────────────────────────────────────────
#  6.  Final labelling loop  (copy-paste over the old one)
# ───────────────────────────────────────────────────────────────
final_labels = {}

for topic_id, topic_keywords in topics.items():          # topics == topic_word_dict
    # 1) 40 best documents we already parsed from topics_docs.txt
    doc_texts = topic_docs.get(topic_id, [])[:40]
    if len(doc_texts) < 3:
        print(f"⚠️  Topic {topic_id}: only {len(doc_texts)} docs → skipped")
        continue

    # 2) extract & score candidates
    cand_raw = extract_label_candidates(doc_texts, set(topic_keywords))
    if not cand_raw:
        cand_raw = [max(topic_keywords, key=len)]
    top5     = score_labels(
        cand_raw,
        doc_texts,
        set(topic_keywords),
        SEED_WORDS          
    )



    top_labels, label_score_dict = score_labels_with_components(
    candidates=cand_raw,
    docs=top5,
    topic_keywords=topic_keywords,
    seed_words=seed_words)




    # 3) build the prompt for the LLM  ➜  THEN call rerank_with_llm
    prompt   = build_prompt(
        list(topic_keywords),      # keep original order if you like
        doc_texts[:TOP_N_DOCS],    # three-snippet default (TOP_N_DOCS == 3)
        top5
    )
    #best_lbl = explan_llm(prompt, model, tokenizer, topic_id)
    best_lbl = call_llm(prompt)
   #best_lbl= postprocess(best_lbl)
    # 4) bookkeeping
    final_labels[topic_id] = {"top_candidates": top5, "final_label": best_lbl}
    print(f"✓  Topic {topic_id}  →  {best_lbl}")

# 5) persist
with open("outputs/Ours_labels_score__7B_04.json", "w", encoding="utf-8") as fh:
    json.dump(final_labels, fh, ensure_ascii=False, indent=2)

plot_label_scores(label_score_dict)




#Baseline
############################################################################
TOPIC_FILE_B = "Evaluation/baseline_org.txt"
topics_base,  docs_base  = parse_topics_and_docs(TOPIC_FILE_B)
def call_llm(prompt, max_new_tokens=300):
    toks = tokenizer(prompt, return_tensors="pt")
    output_ids = model.generate(
        input_ids=toks["input_ids"],
        attention_mask=toks["attention_mask"],
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)[len(prompt):].strip()

def call_llm_baseline(user_message,
                      max_new_tokens=300,
                      temperature=0.9,
                      top_p=0.85):
    messages = [
        {"role": "system", "content": "Olet avulias ja ytimekäs tiivistäjä."},
        {"role": "user",   "content": user_message},
    ]
    ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][ids.size(1):], skip_special_tokens=True).strip()
def postprocess(resp: str) -> str:
    m = re.search(r"ETIKETTI\s*:\s*(.+?)\s*PERUSTELU\s*:\s*(.+)", resp, re.S | re.I)
    if m:
        return f"ETIKETTI: {m.group(1).strip()}\nPERUSTELU: {m.group(2).strip()}"
    return resp.strip()

def build_prompt_baseline(keywords):
    k_string = ", ".join(keywords[:12])
    prompt = (
        f"Alla on avainsanoja, jotka liittyvät tiettyyn keskustelun aiheeseen:\n"
        f"{k_string}\n\n"
        f"Anna näiden avainsanojen perusteella lyhyt ja ytimekäs etiketti, joka kuvaa keskustelun pääaihetta. "
        f"Perustele lyhyesti, miksi valitsit tämän etiketin.\n\n"
        "Vastaa täsmälleen seuraavassa muodossa:\n"
        "ETIKETTI: [kirjoita tähän lyhyt etiketti]\n"
        "PERUSTELU: [perustele etiketti keskustelun sisällön perusteella]"
    )
    return prompt

baseline_results = {}
for topi_id, keywords in topics_base.items():
    prompt = build_prompt_baseline(keywords)
    llm_raw = call_llm_baseline(prompt)  # You can plug in your OpenAI, HuggingFace, or local model call here
    llm_response = postprocess(llm_raw)
    baseline_results[topi_id] = {
        "llm_response": llm_response}
    
output_path = "outputs/baseline_labels.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(baseline_results, f, indent=2, ensure_ascii=False)
print(f"Saved {len(baseline_results)} topics")










