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
STOPWORDS_PATH = '/Users/smn573/Documents/constrained_icml_experiment/stopwords-fi.txt'
#TOPIC_FILE = "Evaluation/baseline_org.txt"
TOPIC_FILE= "Evaluation/top40_docs_cmtm_org.txt"
#DOCUMENTS_PATH = "youtube_comments.jsonl"
#MODEL_NAME = "TurkuNLP/gpt3-finnish-small"
#MODEL_NAME= "Finnish-NLP/gpt2-medium-finnish"
MODEL_NAME= "Finnish-NLP/Ahma-3B-Instruct"
#MODEL_NAME= "Finnish-NLP/Ahma-7B-Instruct"
#MODEL_NAME= "Finnish-NLP/llama-7b-finnish-instruct-v0.2"
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



# # === Step 3: Candidate Label Extraction (TF-IDF n-grams) ===
# def extract_candidate_labels(docs):
#     vectorizer = TfidfVectorizer(ngram_range=NGRAM_RANGE,
#                                  token_pattern=r'\b[a-zA-ZåäöÅÄÖ]{2,}\b')
#     tfidf = vectorizer.fit_transform(docs)
#     print(f"TF-IDF vocab size: {len(vectorizer.vocabulary_)}")

#     freqs = np.asarray(tfidf.sum(axis=0)).ravel()
#     terms = np.array(vectorizer.get_feature_names_out())
#     keep = freqs >= MIN_FREQ
#     return list(terms[keep])
# def extract_label_candidates(top_docs, ngram_range=(1, 3), top_k=20):
#     vectorizer = TfidfVectorizer(
#         ngram_range=ngram_range,
#         min_df=1,
#         max_df=1.0,
#     )
#     tfidf = vectorizer.fit_transform(top_docs)
#     feature_names = vectorizer.get_feature_names_out()
#     scores = tfidf.sum(axis=0).A1
#     scored_ngrams = list(zip(feature_names, scores))
#     scored_ngrams.sort(key=lambda x: -x[1])

#     return [term for term, score in scored_ngrams[:top_k]]
# # === Step 4: Label Scoring ===
# def score_labels(candidates, docs, topic_keywords, seed_words):
#     doc_concat = " ".join(docs)
#     doc_tokens = doc_concat.lower().split()
#     doc_len = len(doc_tokens)
#     token_counts = Counter(doc_tokens)
#     scores = {}
#     for l in candidates:
#         words = l.split()
#         freq = sum(token_counts[w] for w in words if w in token_counts)
#         informativeness = freq / (doc_len + 1e-6)
#         #
#         phraseness = len(words)
#         seed_overlap = len(set(words) & set(seed_words))
#         score = informativeness + LAMBDA_1 * phraseness + LAMBDA_2 * seed_overlap
#         scores[l] = score
#     return sorted(scores.items(), key=lambda x: -x[1])[:TOP_K_LABELS]






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

    # ranked = sorted(label_scores, key=label_scores.get, reverse=True)
    # padded = (ranked + ["dummy"]*k_out)[:k_out]
    # return padded
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






# def build_prompt(keywords, doc_snippets, candidates):
#     k_string = ", ".join(keywords[:12])
#     d_string = "\n\n".join(f"- {snippet[:200]}" for snippet in doc_snippets[:3])  # truncate long ones
#     c_string = "\n".join(f"- {c}" for c in candidates[:3])

#     prompt = (
#         "Esimerkki:\n"
#         "AVAINSANAT: masennus, ahdistus, terapia\n"
#         "ETIKETTI: Nuorten mielenterveys\n"
#         "PERUSTELU: Avainsanat viittaavat nuorten mielenterveys­ongelmiin ja hoitomuotoihin.\n\n"
#         "--- NYT ALKAA UUSI TEHTÄVÄ ---\n\n"
#         f"AVAINSANAT: {k_string}\n\n"
#         f"Seuraavassa on otteita kyseisistä keskusteluista:\n{d_string}\n\n"
#         f"Ehdotetut aiheet:\n{c_string}\n\n"
#         f"Valitse paras lyhyt ja ytimekäs etiketti, joka kuvaa keskustelun aiheen mahdollisimman hyvin.\n"
#         f"Vastaa muodossa:\n"
#         f"ETIKETTI: <etiketti>\n"
#         f"PERUSTELU: <lyhyt perustelu>\n"
#     )
#     return prompt
def build_prompt(keywords, doc_snippets, candidates):
    k_string = ", ".join(keywords[:8])
    d_string = "\n\n".join(f"- {d}" for d in doc_snippets[:2])
    c_string = "\n".join(f"- {c}" for c in candidates[:5])
    prompt = (
        # "Esimerkki:\n"
        # "AVAINSANAT: masennus, ahdistus, terapia\n"
        # "ETIKETTI: Nuorten mielenterveys\n"
        # "PERUSTELU: Avainsanat viittaavat nuorten mielenterveys­ongelmiin ja hoitomuotoihin.\n\n"
        # "--- NYT ALKAA UUSI TEHTÄVÄ ---\n\n"




        ############WORKING PROMPT################################################
        # f"AVAINSANAT: {k_string}\n\n"
        # f"Keskustelukatkelmat:\n{d_string}\n\n"
        # "Anna lyhyt ja ytimekäs etiketti, joka kuvaa keskustelun pääaihetta.\n"
        # "Perustele lyhyesti valintasi.\n\n"
        # "Valitse täsmälleen yksi yllä olevista ehdokkaista tai muokkaa sitä korkeintaan kahden sanan verran."
        # "Vastaa muodossa:\nETIKETTI: <etiketti>\nPERUSTELU: <perustelu>\n"
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
        
        # "Palauta **vain** seuraavassa muodossa (max 4 sanaa per kenttä):"
        # "ETIKETTI: <1-4 sanaa>"
        # "PERUSTELU: <1 lause, max 50 sanaa>"
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
        SEED_WORDS               # ← the set returned by preprocess_text(...)
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
# def build_prompt_baseline(keywords):
#     k_string = ", ".join(keywords[:12])
#     prompt = (
        # "Esimerkki:\n"
        # "AVAINSANAT: masennus, ahdistus, terapia\n"
        # "ETIKETTI: Nuorten mielenterveys\n"
        # "PERUSTELU: Avainsanat viittaavat nuorten mielenterveys­ongelmiin ja hoitomuotoihin.\n\n"
        # "--- NYT ALKAA UUSI TEHTÄVÄ ---\n\n"
    #     f"Alla on avainsanoja, jotka liittyvät tiettyyn keskustelun aiheeseen:\n"
    #     f"{k_string}\n\n"
    #     f"Anna näiden avainsanojen perusteella lyhyt ja ytimekäs etiketti, joka kuvaa keskustelun pääaihetta. "
    #     f"Perustele lyhyesti, miksi valitsit tämän etiketin.\n\n"
    #     f"Vastaa seuraavassa muodossa:\n"
    #     f"ETIKETTI: <etiketti>\n"
    #     f"PERUSTELU: <perustelu>\n"
    # )
    # return prompt
def build_prompt_baseline(keywords):
    k_string = ", ".join(keywords[:12])
    prompt = (
        f"Alla on avainsanoja, jotka liittyvät tiettyyn keskustelun aiheeseen:\n"
        f"{k_string}\n\n"
        f"Anna näiden avainsanojen perusteella lyhyt ja ytimekäs etiketti, joka kuvaa keskustelun pääaihetta. "
        f"Perustele lyhyesti, miksi valitsit tämän etiketin.\n\n"
        # f"Vastaa seuraavassa muodossa:\n"
        # f"ETIKETTI: <etiketti>\n"
        # f"PERUSTELU: <perustelu>\n"
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
    # baseline_results[topi_id] = {
    #     "keywords": keywords,
    #     "prompt": prompt,
    #     "llm_response": llm_response.strip()}
    baseline_results[topi_id] = {
        "llm_response": llm_response}
    
output_path = "outputs/baseline_labels_keywords_7B_04.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(baseline_results, f, indent=2, ensure_ascii=False)
print(f"Saved {len(baseline_results)} topics")







# final_labels = {}

# for topic_id, keywords in topics.items():
#     docs = topic_docs[topic_id][:40]
#     candidates = extract_label_candidates(docs)
#     top_scored = score_labels(candidates, docs, keywords, seed_words=keywords)
#     top_labels = [l for l, _ in top_scored]

#     prompt = build_prompt(keywords, docs[:TOP_N_DOCS], top_labels)

#     print(f"TOP CANDIDATES for topic {topic_id}: {top_labels[:10]}")

#     best_label = rerank_with_llm(prompt, model, tokenizer, topic_id)
#     print(f"LLM label for topic {topic_id}: {best_label}")
#     final_labels[topic_id] = best_label.strip()

# # === Output results ===
# with open("final_labels.json", "w", encoding="utf-8") as f:
#     json.dump(final_labels, f, indent=2, ensure_ascii=False)





















# with open("final_labels.json", "r", encoding="utf-8") as f:
#     raw_labels = json.load(f)



# def clean_llm_label(raw_label):
#     # Remove bullets, numbering, and boilerplate LLM artifacts
#     label = re.sub(r'^[-–•\d.\s]*', '', raw_label.strip())
#     label = re.split(r'[,\n–-]', label)[0]  # Get first segment
#     return label.strip().capitalize()
# cleaned_labels = {int(k): clean_llm_label(v) for k, v in raw_labels.items()}




# def clean_label(raw_text):
#     if "VASTAUS:" in raw_text:
#         raw_text = raw_text.split("VASTAUS:")[-1]
#     label = raw_text.strip().strip("-").strip(",").split("\n")[0]
#     return label.lower()
# cleaned_labels = {int(k): clean_label(v) for k, v in final_labels.items()}




# topic_ids = list(final_labels.keys())
# labels = [final_labels[k] for k in topic_ids]
# plt.figure(figsize=(10, 6))
# plt.bar(topic_ids, range(len(labels)), tick_label=labels)
# plt.xticks(rotation=90)
# plt.ylabel("Topic ID")
# plt.title("LLM-assigned topic labels")
# plt.tight_layout()
# plt.show()



# all_label_text = ' '.join(labels)
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_label_text)
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.title("Word Cloud of LLM Labels")
# plt.show()



# label_counts = Counter(final_labels.values())
# # Bar chart
# plt.figure(figsize=(10, 5))
# plt.bar(label_counts.keys(), label_counts.values())
# plt.xticks(rotation=45, ha='right')
# plt.title("LLM-Generated Topic Labels (Frequency)")
# plt.tight_layout()
# plt.show()






















# def plot_js_heatmap_from_selected_docs(H, tfidf, top_docs_by_topic, doc_ids=None, save_path="JS_Heatmap_selected.png"):
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     from scipy.spatial.distance import jensenshannon

#     def normalize(vec):
#         return vec / (np.sum(vec) + 1e-12)

#     n_topics = H.shape[0]
#     max_top_docs = max(len(v) for v in top_docs_by_topic.values())
#     js_matrix = np.full((n_topics, max_top_docs), np.nan)

#     for topic_idx, doc_indices in top_docs_by_topic.items():
#         topic_dist = normalize(H[topic_idx, :])
#         for i, doc_id in enumerate(doc_indices):
#             doc_vec = tfidf[doc_id].toarray().flatten()
#             doc_dist = normalize(doc_vec)
#             js = jensenshannon(topic_dist, doc_dist)
#             js_matrix[topic_idx, i] = js

#     # Labels for x-axis
#     if doc_ids:
#         doc_labels = [f"Doc {i}" for i in doc_ids]
#     else:
#         doc_labels = [f"Doc {i}" for i in range(max_top_docs)]

#     # Plot heatmap
#     plt.figure(figsize=(12, 8))
#     sns.heatmap(js_matrix, cmap="YlGnBu", vmin=0, vmax=1.0,
#                 xticklabels=doc_labels[:max_top_docs],
#                 yticklabels=[f"Topic {i}" for i in range(n_topics)])
#     plt.title("Jensen-Shannon Divergence Heatmap (Topics × Preselected Top Documents)")
#     plt.xlabel("Top Documents")
#     plt.ylabel("Topics")
#     plt.xticks(rotation=45, ha="right")
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.show()





# from wordcloud import WordCloud
# import matplotlib.pyplot as plt

# top_words_text = " ".join([" ".join([w for w, _ in topic]) for topic in topics[:16]])
# #label_words = " ".join(final_labels)
# label_words = " ".join(set(v["final_label"] for v in final_labels.values()))

# wc1 = WordCloud(width=600, height=300, background_color='white').generate(top_words_text)
# wc2 = WordCloud(width=600, height=300, background_color='white').generate(label_words)

# fig, axs = plt.subplots(1, 2, figsize=(12, 4))
# axs[0].imshow(wc1, interpolation='bilinear')
# axs[0].axis("off")
# axs[0].set_title("Topic Top Words")
# axs[1].imshow(wc2, interpolation='bilinear')
# axs[1].axis("off")
# axs[1].set_title("Final Labels")
# plt.savefig("fig_wordcloud_comparison.png")







# def plot_js_heatmap(V, H, doc_titles=None, top_docs=20, figsize=(12, 8), save_path=None):
#     import warnings
#     js_matrix = np.zeros((H.shape[0], V.shape[0]))

#     for k in range(H.shape[0]):
#         topic_dist = H[k, :]
#         topic_dist = topic_dist / (np.sum(topic_dist) + 1e-12)
#         for d in range(V.shape[0]):
#             doc_vec = V[d].toarray().flatten()
#             if np.sum(doc_vec) == 0:
#                 js_matrix[k, d] = np.nan
#                 continue
#             doc_dist = doc_vec / (np.sum(doc_vec) + 1e-12)
#             try:
#                 js = jensenshannon(topic_dist, doc_dist)
#                 js_matrix[k, d] = js if np.isfinite(js) else np.nan
#             except Exception as e:
#                 warnings.warn(f"JS error for topic {k}, doc {d}: {e}")
#                 js_matrix[k, d] = np.nan

    

#     # Focus on most representative documents
#     with np.errstate(invalid='ignore'):
#         min_js = np.nanmin(js_matrix, axis=0)
#     top_doc_indices = np.argsort(min_js)[:top_docs]
#     top_js_matrix = js_matrix[:, top_doc_indices]
#     # Print sample values
#     print("JS values for topic 0:", top_js_matrix[0, :20])

#     # Prepare doc labels
#     if doc_titles:
#         doc_labels = [doc_titles[i][:30] + "..." for i in top_doc_indices]
#     else:
#         doc_labels = [f"Doc {i}" for i in top_doc_indices]

#     # Heatmap
#     plt.figure(figsize=figsize)
#     sns.heatmap(top_js_matrix, vmin=0, vmax=1.0, cmap="YlGnBu",
#                 xticklabels=doc_labels, yticklabels=[f"Topic {i}" for i in range(H.shape[0])])
#     plt.title("Jensen-Shannon Divergence Heatmap (Topics × Top Documents)")
#     plt.xlabel("Top Documents")
#     plt.ylabel("Topics")
#     plt.xticks(rotation=45, ha="right")
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path)
#     else:
#         plt.savefig("JS_Heatmap.png")
#     plt.show()

#     # Histogram of all valid JS values
#     valid_js = js_matrix[np.isfinite(js_matrix)].flatten()
#     plt.hist(valid_js, bins=50, color='skyblue')
#     plt.title("Distribution of Jensen-Shannon Divergences")
#     plt.xlabel("JS Divergence")
#     plt.ylabel("Frequency")
#     plt.tight_layout()
#     plt.savefig("JS_distribution.png")
#     plt.show()


    

# plot_js_heatmap(tfidf, H)




# # --- Helpers ---
# def clean_label_string(label):
#     # Remove bullets, punctuation, and lowercase
#     label = re.sub(r"^[\-\•\*]?\s*", "", label)  # remove leading dash or bullet
#     label = re.sub(r"[^a-zA-ZäöåÄÖÅ\s]", " ", label.lower())  # remove punctuation
#     return label

# def lemmatize_finnish(text, voikko):
#     return [voikko.analyze(word)[0]['BASEFORM'] if voikko.analyze(word) else word for word in text.split()]

# def clean_and_lemmatize(text, voikko):
#     text = clean_label_string(text)
#     tokens = text.strip().split()
#     lemmatized = []
#     for token in tokens:
#         analysis = voikko.analyze(token)
#         lemma = analysis[0]["BASEFORM"] if analysis else token
#         lemmatized.append(lemma)
#     return set(lemmatized)

# # --- Main Plot Function ---
# def plot_seed_overlap_distribution(labels_data, seed_words, normalize=False, show_examples=True):
#     voikko = libvoikko.Voikko("fi")

#     # Lemmatize seed words
#     seed_lemmas = set()
#     for word in seed_words:
#         seed_lemmas |= clean_and_lemmatize(word, voikko)  # union

#     print(f"\n✅ Loaded {len(seed_lemmas)} lemmatized seed words.\n")

#     overlaps_candidate = []
#     overlaps_final = []
#     label_examples = defaultdict(list)

#     for topic_id, topic_data in labels_data.items():
#         # Candidate labels
#         for label in topic_data.get("top_candidates", []):
#             lem_label = clean_and_lemmatize(label, voikko)
#             overlap = len(lem_label & seed_lemmas)
#             overlaps_candidate.append(overlap)
#             label_examples[("Candidate", overlap)].append(label)

#         # Final label
#         if "final_label" in topic_data:
#             final_label = topic_data["final_label"]
#             final_label_clean = clean_label_string(final_label)
#             lem_label = clean_and_lemmatize(final_label_clean, voikko)
#             overlap = len(lem_label & seed_lemmas)
#             overlaps_final.append(overlap)
#             label_examples[("Final", overlap)].append(final_label)

#             # Debug print for mismatches
#             if show_examples and overlap == 0:
#                 print(f"\n🔍 Final label with 0 overlap:\n  - {final_label}")
#                 print(f"  → Lemmatized: {lem_label}")
#                 print(f"  → Overlap: {lem_label & seed_lemmas}")

#     # Count overlap values
#     counter_cand = Counter(overlaps_candidate)
#     counter_final = Counter(overlaps_final)
#     x_vals = sorted(set(counter_cand.keys()) | set(counter_final.keys()))

#     y_cand = [counter_cand.get(x, 0) for x in x_vals]
#     y_final = [counter_final.get(x, 0) for x in x_vals]

#     if normalize:
#         total_cand = sum(y_cand)
#         total_final = sum(y_final)
#         y_cand = [y / total_cand for y in y_cand]
#         y_final = [y / total_final for y in y_final]

#     # --- Plot ---
#     plt.figure(figsize=(10, 5))
#     bar_width = 0.4
#     sns.set_style("whitegrid")

#     plt.bar([x - bar_width/2 for x in x_vals], y_cand, width=bar_width, label="Top Candidates", color='skyblue')
#     plt.bar([x + bar_width/2 for x in x_vals], y_final, width=bar_width, label="Final Labels", color='orange')

#     plt.xlabel("Number of Overlapping Seed Words (Lemmatized)")
#     plt.ylabel("Proportion of Labels" if normalize else "Number of Labels")
#     plt.title("Seed Word Overlap in Candidate and Final Labels")
#     plt.xticks(x_vals)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("fig_seed_overlap_comparison.png")
#     plt.show()

#     if show_examples:
#         for (label_type, overlap_val), labels in sorted(label_examples.items()):
#             print(f"\n📌 {label_type} labels with {overlap_val} overlapping seed words:")
#             for label in labels[:3]:
#                 print(f"  - {label}")




# plot_seed_overlap_distribution(final_labels, seed_words)