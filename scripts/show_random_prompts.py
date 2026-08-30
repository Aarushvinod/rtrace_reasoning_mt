"""
show_random_prompts.py
──────────────────────
Print randomly sampled WMT24++ few-shot prompts exactly as the generation
pipeline builds them (template-11 + the study's prompt header), using the
committed fixed split and the committed random-pool selections — so the
printed prompts are byte-identical to what the `random` method's jobs send.
(rrf / sentinel / edit_dist prompts differ only in WHICH pool examples are
selected; reproducing those requires the embedding matrices.)

    python scripts/show_random_prompts.py            # 2 samples, random k/lang
    python scripts/show_random_prompts.py 3          # 3 samples
    python scripts/show_random_prompts.py 3 5        # 3 samples, k=5
    python scripts/show_random_prompts.py 3 5 zul_Latn

Needs only `datasets` (downloads google/wmt24pp on first use).
"""

import json
import os
import random
import sys

# Target-side scripts (Mlym etc.) overflow legacy console codepages.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.wmt24pp import WMT_TGT_LANGS, load_wmt24pp_sentences

# Mirrors reasoning_main.PROMPT_HEADER / LANG_NAME and template-11 in
# retrieval_helpers (kept inline so this script needs no torch/openai).
PROMPT_HEADER = (
    'You are an expert translator who translates sentences from any specified src language to any specified tgt language.'
    ' You should reason over the demonstration sentences provided to you below, using them as a guide to accurately translate the final sentence.'
    ' Your final response should be the translation of the final untranslated src sentence to the tgt language with no other words or characters accompanying the translation.\n'
)
LANG_NAME = {
    "eng_Latn": "English", "cat_Latn": "Catalan", "zul_Latn": "Zulu",
    "mal_Mlym": "Malayalam", "slk_Latn": "Slovak", "isl_Latn": "Icelandic",
}
K_CHOICES = [1, 3, 5, 7, 10]


def template11_prompt(src_sentence, demonstrations, src_name, tgt_name):
    prompt = PROMPT_HEADER
    for x, y in demonstrations:
        prompt += f"{src_name} sentence\n{x}\n{tgt_name} translation\n{y}\n###\n"
    prompt += f"{src_name} sentence\n{src_sentence}\n{tgt_name} translation\n"
    return prompt


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    k_fixed = int(sys.argv[2]) if len(sys.argv) > 2 else None
    lang_fixed = sys.argv[3] if len(sys.argv) > 3 else None

    sel = json.load(open(os.path.join("data", "random_pool_selections", "wmt24pp_eng_random_pool.json")))
    eng = load_wmt24pp_sentences("eng_Latn")
    tgt_cache = {}

    for s in range(1, n + 1):
        lang = lang_fixed or random.choice(WMT_TGT_LANGS)
        k = k_fixed or random.choice(K_CHOICES)
        i = random.randrange(len(eng["devtest"]))
        if lang not in tgt_cache:
            tgt_cache[lang] = load_wmt24pp_sentences(lang)
        idxs = sel["selections"][i][:k]
        demos = [(eng["dev"][j], tgt_cache[lang]["dev"][j]) for j in idxs]
        prompt = template11_prompt(eng["devtest"][i], demos, "English", LANG_NAME[lang])
        print("=" * 100)
        print(f"[{s}] method=random | eng_Latn -> {lang} | k={k} | test sentence #{i} | "
              f"prompt length: {len(prompt)} chars, ~{len(prompt.split())} words")
        print("=" * 100)
        print(prompt)


if __name__ == "__main__":
    main()
