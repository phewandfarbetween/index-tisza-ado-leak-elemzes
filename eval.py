import gzip
import http.client
import json
import numpy
import os
import os.path
import sys
import typing
import urllib.parse

import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

API_KEYS = {
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "deepseek": os.getenv("DEEPSEEK_API_KEY"),
    "google": os.getenv("GEMINI_API_KEY"),
    "openai": os.getenv("OPENAI_API_KEY"),
}

API_URL_TPLS = {
    "anthropic": "https://api.anthropic.com/v1/messages",
    "deepseek": "https://api.deepseek.com/chat/completions",
    "google": "https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}",
    "openai": "https://api.openai.com/v1/responses",
}

IS_DEBUGGING = any(arg == "--debug" for arg in sys.argv)

def debug(msg):
    if IS_DEBUGGING:
        print(msg)

def query_llm(api: str, model_name: str, system_prompt: str, user_prompt: str, temperature=1.0) -> str:
    api = api.lower()
    if api == "anthropic":
        return query_anthropic(model_name, system_prompt, user_prompt, temperature)
    if api == "deepseek":
        return query_deepseek(model_name, system_prompt, user_prompt, temperature)
    if api == "google":
        return query_google(model_name, system_prompt, user_prompt, temperature)
    if api == "openai":
        return query_openai(model_name, system_prompt, user_prompt, temperature)
    raise ValueError(f"Unknown API: {api}")

def query_anthropic(model_name: str, system_prompt: str, user_prompt: str, temperature=1.0) -> str:
    api_key = API_KEYS["anthropic"]
    url = API_URL_TPLS["anthropic"]
    payload = {
        "model": model_name,
        "system": system_prompt,
        "max_tokens": 4096,
        "temperature": temperature,
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": user_prompt}],
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "extended-cache-ttl-2025-04-11",
    }
    raw = http_request("POST", url, headers, json.dumps(payload).encode("utf-8"))
    data = json.loads(raw)
    return data["content"][0]["text"]

def query_deepseek(model_name: str, system_prompt: str, user_prompt: str, temperature=1.0) -> str:
    api_key = API_KEYS["deepseek"]
    url = API_URL_TPLS["deepseek"]
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    raw = http_request("POST", url, headers, json.dumps(payload).encode("utf-8"))
    data = json.loads(raw)
    return data["choices"][0]["message"]["content"]

def query_google(model_name: str, system_prompt: str, user_prompt: str, temperature=1.0) -> str:
    api_key = API_KEYS["google"]
    url = API_URL_TPLS["google"].format(model_name=model_name, api_key=api_key)
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{
            "role": "user",
            "parts": [{"text": user_prompt}]
        }],
        "generationConfig": {
            "temperature": temperature,
        }
    }
    headers = {"Content-Type": "application/json"}
    raw = http_request("POST", url, headers, json.dumps(payload).encode("utf-8"))
    data = json.loads(raw)
    return data["candidates"][0]["content"]["parts"][0]["text"]

def query_openai(model_name: str, system_prompt: str, user_prompt: str, temperature=1.0) -> str:
    api_key = API_KEYS["openai"]
    url = API_URL_TPLS["openai"]
    payload = {
        "model": model_name,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_output_tokens": 4096,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    raw = http_request("POST", url, headers, json.dumps(payload).encode("utf-8"))
    data = json.loads(raw)
    return data["output"][0]["content"][0]["text"]

def _prepare_conn_and_path(url: str):
    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme
    host = parsed.netloc
    path = parsed.path
    if parsed.query:
        path += "?" + parsed.query
    if scheme == "https":
        conn = http.client.HTTPSConnection(host)
    else:
        conn = http.client.HTTPConnection(host)
    return conn, path

def http_request(method: str,
                 url: str,
                 headers: typing.Optional[typing.Dict[str, str]]=None,
                 body: typing.Optional[bytes]=None,
                 bufsize: int=65536) -> bytes:
    conn, path = _prepare_conn_and_path(url)
    conn.request(method, path, body=body, headers=headers or {})
    resp = conn.getresponse()
    if resp.status < 200 or resp.status >= 300:
        raise RuntimeError(f"HTTP error {resp.status}: {resp.reason}")
    chunks = []
    while True:
        chunk = resp.read(bufsize)
        if not chunk:
            break
        chunks.append(chunk)
    conn.close()
    return b"".join(chunks)

if __name__ == '__main__':
    models = [
        ("anthropic", "claude-sonnet-4-5-20250929", 0.1),
        ("deepseek", "deepseek-chat", 0.1),
        ("google", "gemini-2.5-flash", 0.1),
        ("openai", "gpt-4.1", 0.1),
    ]
    system_prompt = """\
Kérlek válaszolj segítőkész MI-asszisztensként, akinek kiemelt szakterülete az LLM-ek és az emberek által írt szövegek megkülönböztetése.

A feladat egy szövegrészlet elemzése egy dokumentumból, amelynek az eredeti példánya nem áll rendelkezésre, csak az OCR segítségével digitalizált változata. Emiatt a mintában jellegzetes karakterfelismerési, tördelési és formázási hibák fordulnak elő, amelyeket nem szabad figyelembe venni. A dokumentum egyes részleteinek szerzőségével kapcsolatban kételyek merültek fel.

Kérlek alaposan gondold végig, hogy mi lehetett az eredeti szövegrészlet, amelyből az OCR az elemzendő töredéket előállította, és osztályozd 1-5 közötti skálán aszerint, hogy mennyire valószínű, hogy a minta eredeti szövegét részben vagy egészben mesterséges intelligencia generálta:

1: szinte teljesen biztos, hogy a minta teljes szövegét ember írta, gépi segítség nélkül.
2: a minta ugyan tartalmaz elenyésző mennyiségű LLM által írt szöveget, de a legnagyobb részét ember írta.
3: a minta nagyjából fele-fele arányban tartalmaz ember és LLM által írt részeket.
4: a minta ugyan tartalmaz ember által írt részleteket, de a nagy többségét LLM generálta.
5: szinte teljesen biztos, hogy a minta teljes szövegét LLM generálta, ember által írt szöveg legfeljebb a feladatleírásban (prompt) fordulhatott elő.

Kérlek vedd figyelembe, hogy ez csak egy rövid szövegtöredék egy hosszabb dokumentumból, ezért a szöveg hiányos lehet.

Fontos: az OCR-hibák jelenléte önmagában nem bizonyítja sem az emberi, sem az LLM általi szerzőséget. **A kérdés az eredeti, OCR-hibáktól mentes szöveg forrása.**

A válaszodban először röviden foglald össze az észrevételeidet a mintáról, például:

 * tipikus LLM-re utaló szófordulatok és megfogalmazások,
 * vélhetően hallucinált adatok,
 * LLM-ekre jellemző tagolás és mondatszerkezetek,
 * gyanús visszautalások egy esetleges prompt szöveg elemeire,
 * egyéb jellegzetes LLM stílusjegyek,
 * stb.

Ezután foglald össze az összbenyomásod, majd **a válaszod legvégén egy új sorban egyetlen számjeggyel, formázások és egyéb karakterek nélkül add meg a végleges pontszámot**.

A munkádhoz tilos külső segítséget használnod (mint például internetes keresők és hasonló eszközök).
"""
    user_prompt_tpl = """\
Következik az elemzendő szövegrészlet:

"""
    files = {
        "human": "human-samples.txt",
        "humanocr": "human-samples-ocr.txt",
        "llm": "llm-samples.txt",
        "llmocr": "llm-samples-ocr.txt",
        "leakocr": "leak-samples-ocr.txt",
    }
    argv = [arg for arg in sys.argv if arg != "--debug"]
    api_selection = None if len(argv) < 2 else argv[1]
    scores = {}
    for experiment, file_name in files.items():
        scores[experiment] = []
        with gzip.open("samples/" + file_name + ".gz", "rt") as f:
            samples = f.read().split("---")
        for i, sample in enumerate(samples):
            sample = sample.strip()
            user_prompt = user_prompt_tpl + sample
            scores[experiment].append([])
            for api, model_name, temperature in models:
                if api_selection is not None and api != api_selection:
                    continue
                print(f"# {file_name=}, {i=}, {api=}, {model_name=} ", end="")
                response_file_name = f"responses/{experiment}-{i}-{api}-{model_name}.txt.gz"
                if os.path.exists(response_file_name):
                    with gzip.open(response_file_name, "rt") as f:
                        response = f.read().strip()
                else:
                    response = query_llm(api, model_name, system_prompt, user_prompt, temperature)
                    with gzip.open(response_file_name, "wt") as f:
                        print(response, file=f)
                lines = response.replace("\r", "\n").split("\n")
                score = 0
                for line in lines:
                    line = line.replace("\t", " ").split(" ")[-1].strip().strip("*:,;.")
                    if line and line in "12345":
                        score = int(line)
                if score != 0:
                    scores[experiment][-1].append(score)
                    print(f"{score=}")
                else:
                    print(f"NO VALID SCORE!")
                debug("")
                debug(response)
                debug("")

    scores_np = {key: numpy.array(arr) for key, arr in scores.items()}

    print()
    print("Pontszámok átlaga (1 = biztosan ember, 5 = biztosan LLM)")
    for experiment, exp_scores in scores_np.items():
        print(f"  {files[experiment] + ':':25} {exp_scores.flatten().mean():.3f}  (std: {exp_scores.flatten().std():.3f})")

    print()
    print("Mann-Whitney-próba")
    u1, p1 = mannwhitneyu(scores_np["leakocr"].mean(axis=1), scores_np["humanocr"].mean(axis=1), alternative="two-sided")
    print(f"  leakocr vs humanocr: p-value = {p1:.9f}")
    u2, p2 = mannwhitneyu(scores_np["leakocr"].mean(axis=1), scores_np["llmocr"].mean(axis=1), alternative="two-sided")
    print(f"  leakocr vs llmocr:   p-value = {p2:.9f}")
    alpha = 0.05
    if p1 > alpha and p2 < alpha:
        print("  A leakocr minta megkülönböztethetetlen a humanocr mintától és különbözik az llmocr mintától.")
    elif p1 < alpha and p2 > alpha:
        print("  A leakocr minta megkülönböztethetetlen az llmocr mintától és különbözik a humanocr mintától.")
    elif p1 < alpha and p2 < alpha:
        print("  A leakocr minta mindkét mintától eltér.")
    else:
        print("  Inkonklúzív.")

    print()
    print("LogisticRegression")
    X_train = numpy.vstack([
        scores_np["human"],
        scores_np["humanocr"],
        scores_np["llm"],
        scores_np["llmocr"],
    ])
    y_train = numpy.array(
        [0] * len(scores["human"])
        + [0] * len(scores["humanocr"])
        + [1] * len(scores["llm"])
        + [1] * len(scores["llmocr"])
    )
    model = LogisticRegression()
    if IS_DEBUGGING:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        print(f"  CV accuracy: {cv_scores.mean():.3f}  (std: {cv_scores.std():.3f})")
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(scores_np["leakocr"])
    y_prob_llm = y_prob[:, 1]
    print(f"  leak-samples-ocr.txt átlagos LLM valószínűség: {y_prob_llm.mean():.3f}  (std: {y_prob_llm.std():.3f})")
    print(f"  leak-samples-ocr.txt medián LLM valószínűség:  {numpy.median(y_prob_llm):.3f}")

    labels = [key for key in scores.keys()]
    plt.boxplot(
        [scores_np[label].mean(axis=1) for label in labels],
        labels=labels,
    )
    plt.title("Szerzőség pontszámok (1 = ember, 5 = LLM)")
    plt.savefig("boxplot.png", dpi=600)
