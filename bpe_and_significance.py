"""
Stage 1 add-on: BPE subword model + significance tests (RUHSOLD, coarse task).
Run:  pip install scikit-learn pandas numpy scipy sentencepiece
      python bpe_and_significance.py
Outputs: results/subword_validation.csv, results/significance_tests.csv
Self-contained: it reloads the data and refits the models, so it does not need the notebook.
"""
import os, re, csv, subprocess, sys, warnings
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

try:
    import sentencepiece as spm
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentencepiece"], check=True)
    import sentencepiece as spm

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
os.makedirs("results", exist_ok=True)
os.makedirs("sp", exist_ok=True)

# ---------------------------------------------------------------- data
BASE = "https://raw.githubusercontent.com/haroonshakeel/roman_urdu_hate_speech/main/"

def load_split(name):
    d = pd.read_csv(BASE + f"task_1_{name}.tsv", sep="\t", header=None, names=["tweet", "raw"],
                    quoting=csv.QUOTE_NONE, encoding="utf-8", dtype=str, keep_default_na=False)
    d["y"] = (d["raw"].str.strip().astype(int) == 0).astype(int)   # 1 = Hate/Offensive, 0 = Normal
    return d

def clean(t):
    t = t.lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    return re.sub(r"\s+", " ", t).strip()

train, val, test = load_split("train"), load_split("validation"), load_split("test")
for d in (train, val, test):
    d["clean"] = d["tweet"].map(clean)
print({"train": len(train), "val": len(val), "test": len(test)})

# ---------------------------------------------------------------- models
def word_vec():  return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
def char_vec():  return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
def both_vec():  return FeatureUnion([("word", word_vec()), ("char", char_vec())])
def LR():        return LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced", random_state=SEED)
def SVM():       return LinearSVC(class_weight="balanced", random_state=SEED)

CANDIDATES = {
    "B1 word + LR":      Pipeline([("feat", word_vec()), ("clf", LR())]),
    "B2 word + SVM":     Pipeline([("feat", word_vec()), ("clf", SVM())]),
    "M1 char + LR":      Pipeline([("feat", char_vec()), ("clf", LR())]),
    "M2 char + SVM":     Pipeline([("feat", char_vec()), ("clf", SVM())]),
    "M3 word+char + LR": Pipeline([("feat", both_vec()), ("clf", LR())]),
}
GRID = {"clf__C": [0.1, 0.3, 1, 3, 10]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def macro_f1(y, p): return f1_score(y, p, average="macro")

def scores(y, p):
    return {"macro_F1": macro_f1(y, p), "acc": accuracy_score(y, p),
            "P_hate": precision_score(y, p, pos_label=1), "R_hate": recall_score(y, p, pos_label=1),
            "F1_hate": f1_score(y, p, pos_label=1)}

fitted = {}
for name, pipe in CANDIDATES.items():
    gs = GridSearchCV(pipe, GRID, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True)
    gs.fit(train["clean"], train.y)
    fitted[name] = gs.best_estimator_
    print("fitted:", name, "C =", gs.best_params_["clf__C"])

# ---------------------------------------------------------------- BPE subword model
# SentencePiece is trained on the training tweets only (unlabelled text).
with open("sp/train_text.txt", "w", encoding="utf-8") as f:
    for t in train["clean"]:
        f.write(t.replace("\n", " ") + "\n")

SP_MODELS, CURRENT_SP = {}, None

def sp_pieces(texts):
    return [" ".join(SP_MODELS[CURRENT_SP].encode(t, out_type=str)) for t in texts]

rows, sp_fitted = [], {}
for vocab in [2000, 8000]:
    prefix = f"sp/bpe_{vocab}"
    spm.SentencePieceTrainer.train(input="sp/train_text.txt", model_prefix=prefix, vocab_size=vocab,
                                   model_type="bpe", character_coverage=1.0, hard_vocab_limit=False)
    SP_MODELS[vocab] = spm.SentencePieceProcessor(model_file=prefix + ".model")
    CURRENT_SP = vocab
    pipe = Pipeline([("pieces", FunctionTransformer(sp_pieces)),
                     ("feat", TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                     ("clf", LR())])
    gs = GridSearchCV(pipe, GRID, scoring="f1_macro", cv=cv, n_jobs=1, refit=True)   # n_jobs=1: the pieces function uses a global
    gs.fit(train["clean"], train.y)
    i = gs.best_index_
    name = f"S{vocab} subword (BPE {vocab}) + LR"
    rows.append({"model": name, "cv_macro_F1": gs.cv_results_["mean_test_score"][i],
                 "cv_std": gs.cv_results_["std_test_score"][i], "best_C": gs.best_params_["clf__C"],
                 **scores(val.y, gs.predict(val["clean"]))})
    sp_fitted[name] = (vocab, gs.best_estimator_)
    print("fitted:", name)

subword_res = pd.DataFrame(rows).round(4)
subword_res.to_csv("results/subword_validation.csv", index=False)
print("\nSubword models (validation):\n", subword_res.to_string(index=False))

# ---------------------------------------------------------------- significance tests (test set)
def predict_any(name, X):
    global CURRENT_SP
    if name in fitted:
        return fitted[name].predict(X)
    vocab, est = sp_fitted[name]
    CURRENT_SP = vocab
    return est.predict(X)

best_sp = max(sp_fitted, key=lambda n: subword_res.set_index("model").loc[n, "macro_F1"])
PAIRS = [("B1 word + LR", "M1 char + LR"),
         ("B2 word + SVM", "M2 char + SVM"),
         ("B1 word + LR", "M3 word+char + LR"),
         ("M1 char + LR", "M3 word+char + LR"),
         ("M1 char + LR", best_sp)]

rng = np.random.default_rng(SEED)
y = test.y.values
sig_rows = []
for a, b in PAIRS:
    pa, pb = predict_any(a, test["clean"]), predict_any(b, test["clean"])
    ca, cb = (pa == y), (pb == y)
    n01 = int((ca & ~cb).sum())          # A right, B wrong
    n10 = int((~ca & cb).sum())          # A wrong, B right
    p_mc = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if (n01 + n10) else 1.0   # exact McNemar
    diffs = []
    for _ in range(1000):                # paired bootstrap of the macro-F1 difference
        idx = rng.integers(0, len(y), len(y))
        diffs.append(macro_f1(y[idx], pb[idx]) - macro_f1(y[idx], pa[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    sig_rows.append({"A": a, "B": b, "F1_A": round(macro_f1(y, pa), 4), "F1_B": round(macro_f1(y, pb), 4),
                     "diff_B_minus_A": round(macro_f1(y, pb) - macro_f1(y, pa), 4),
                     "boot_CI_low": round(lo, 4), "boot_CI_high": round(hi, 4),
                     "A_right_B_wrong": n01, "A_wrong_B_right": n10, "McNemar_p": round(p_mc, 4)})

sig = pd.DataFrame(sig_rows)
sig.to_csv("results/significance_tests.csv", index=False)
print("\nSignificance tests (test set):\n", sig.to_string(index=False))
print("\nSaved: results/subword_validation.csv, results/significance_tests.csv")
