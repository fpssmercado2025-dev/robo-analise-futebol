"""
Robo de Analise de Futebol - Coleta Diaria
Roda no GitHub Actions todo dia as 09h (Brasilia).
"""
import os
import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from scipy.stats import poisson, nbinom
from scipy.optimize import minimize

# ---------- CONFIG ----------
API_KEY = os.environ.get("API_KEY", "8037966d59ea18080a27dff43c0dcffd")
BASE_URL = "https://v3.football.api-sports.io"
TIMEZONE = "America/Sao_Paulo"
HEADERS = {"x-apisports-key": API_KEY}

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

HIST_PATH = DATA_DIR / "historico_stats.csv"
PREV_PATH = DATA_DIR / "previsoes.csv"

LIGAS_ALVO = {
    # Clubes
    2:   "Champions League", 39:  "Premier League", 140: "La Liga",
    135: "Serie A", 78:  "Bundesliga", 61:  "Ligue 1",
    71:  "Brasileirao", 13:  "Libertadores",
    # Selecoes (sem amistosos)
    1:   "World Cup", 4: "Euro Championship", 9: "Copa America",
    5:   "UEFA Nations League", 6: "Africa Cup of Nations",
    7:   "AFC Asian Cup", 29: "WC Qualifiers CAF", 30: "WC Qualifiers OFC",
    31:  "WC Qualifiers CONCACAF", 32: "WC Qualifiers UEFA",
    33:  "WC Qualifiers AFC", 34: "WC Qualifiers CONMEBOL",
}

SELECOES = {1, 4, 9, 5, 6, 7, 29, 30, 31, 32, 33, 34}

MERCADOS = {
    "escanteios": (7.5, 8.5, 9.5, 10.5),
    "chutes":     (18.5, 20.5, 22.5, 24.5),
    "chutes_gol": (6.5, 7.5, 8.5, 9.5),
}

# ---------- CACHE ----------
def _cache_path(nome): return CACHE_DIR / f"{nome}.json"

def _salvar_cache(nome, dados):
    if not dados: return
    with open(_cache_path(nome), "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False)

def _ler_cache(nome):
    p = _cache_path(nome)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# ---------- API ----------
def chamar_api(endpoint, params, tentativas=5):
    url = f"{BASE_URL}/{endpoint}"
    for i in range(tentativas):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if r.status_code == 200:
                return r.json().get("response", [])
            elif r.status_code == 429:
                espera = 15 + i * 10
                print(f"[429] aguardando {espera}s (tentativa {i+1}/{tentativas})")
                time.sleep(espera)
            else:
                print(f"[aviso] status {r.status_code}")
                time.sleep(2 ** i)
        except Exception as e:
            print(f"[erro] {e}")
            time.sleep(2 ** i)
    return None

def get_fixtures(data, usar_cache=True):
    hoje = datetime.now().strftime("%Y-%m-%d")
    if data == hoje:
        usar_cache = False
    nome = f"fixtures_{data}"
    if usar_cache:
        c = _ler_cache(nome)
        if c and len(c) > 0:
            print(f"[cache] fixtures {data}")
            return c
    resp = chamar_api("fixtures", {"date": data, "timezone": TIMEZONE})
    if resp is None:
        antigo = _ler_cache(nome)
        return antigo if antigo else []
    if resp:
        _salvar_cache(nome, resp)
        print(f"[ok] {len(resp)} jogos em {data}")
    return resp

def get_estatisticas(fixture_id):
    nome = f"stats_{fixture_id}"
    c = _ler_cache(nome)
    if c: return c
    resp = chamar_api("fixtures/statistics", {"fixture": fixture_id})
    if resp:
        _salvar_cache(nome, resp)
    return resp or []

def extrair_stats_jogo(fixture_id, data_jogo=None, liga_nome=None):
    stats = get_estatisticas(fixture_id)
    if not stats or len(stats) < 2:
        return None
    def buscar(ts, chave):
        for item in ts.get("statistics", []):
            if item["type"] == chave:
                return item["value"]
        return None
    casa, fora = stats[0], stats[1]
    resultado = {
        "fixture_id": fixture_id,
        "time_casa": casa["team"]["name"],
        "time_fora": fora["team"]["name"],
        "escanteios_casa": buscar(casa, "Corner Kicks"),
        "escanteios_fora": buscar(fora, "Corner Kicks"),
        "chutes_casa": buscar(casa, "Total Shots"),
        "chutes_fora": buscar(fora, "Total Shots"),
        "chutes_gol_casa": buscar(casa, "Shots on Goal"),
        "chutes_gol_fora": buscar(fora, "Shots on Goal"),
    }
    if data_jogo:
        resultado["data_jogo"] = str(data_jogo)[:10]
    if liga_nome:
        resultado["liga_nome"] = liga_nome
    return resultado


def fixtures_para_df(fixtures):
    linhas = []
    for f in fixtures:
        linhas.append({
            "fixture_id": f["fixture"]["id"],
            "data": f["fixture"]["date"],
            "liga_id": f["league"]["id"],
            "liga": f["league"]["name"],
            "pais": f["league"]["country"],
            "time_casa": f["teams"]["home"]["name"],
            "time_fora": f["teams"]["away"]["name"],
            "status": f["fixture"]["status"]["short"],
            "gols_casa": f["goals"]["home"],
            "gols_fora": f["goals"]["away"],
        })
    return pd.DataFrame(linhas)

# ---------- MODELO v2 ----------
def nb_pmf(k, mu, alpha):
    if alpha <= 1e-6: return poisson.pmf(k, mu)
    n = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)
    return nbinom.pmf(k, n, p)

def estimar_alpha(valores, alpha_prior=0.10, min_obs=20):
    v = np.asarray(valores, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < min_obs: return alpha_prior
    mu = v.mean()
    if mu <= 0: return alpha_prior
    def nll(a):
        a = a[0]
        if a < 0: return 1e10
        return -sum(np.log(max(nb_pmf(int(k), mu, a), 1e-12)) for k in v)
    try:
        return float(minimize(nll, x0=[alpha_prior], bounds=[(0.0, 2.0)]).x[0])
    except Exception:
        return alpha_prior

def _media(df, col):
    return pd.concat([df[f"{col}_casa"], df[f"{col}_fora"]]).dropna().mean()

def _ataque_geral(df, t, col, min_jogos=3):
    cc, cf = f"{col}_casa", f"{col}_fora"
    media = _media(df, col)
    val = pd.concat([
        df[df["time_casa"] == t][cc],
        df[df["time_fora"] == t][cf]
    ]).dropna()
    n = len(val)
    if n == 0: return media, 0
    peso = min(n / min_jogos, 1.0)
    return peso * val.mean() + (1 - peso) * media, n

def _defesa_geral(df, t, col, min_jogos=3):
    cc, cf = f"{col}_casa", f"{col}_fora"
    media = _media(df, col)
    val = pd.concat([
        df[df["time_casa"] == t][cf],
        df[df["time_fora"] == t][cc]
    ]).dropna()
    n = len(val)
    if n == 0: return media, 0
    peso = min(n / min_jogos, 1.0)
    return peso * val.mean() + (1 - peso) * media, n

def _ataque_casa_fora(df, t, cond, col, min_jogos=3):
    cc, cf = f"{col}_casa", f"{col}_fora"
    media = _media(df, col)
    if cond == "casa":
        val = df[df["time_casa"] == t][cc].dropna()
    else:
        val = df[df["time_fora"] == t][cf].dropna()
    n = len(val)
    if n == 0: return media, 0
    peso = min(n / min_jogos, 1.0)
    return peso * val.mean() + (1 - peso) * media, n

def prever_mercado_v2(df, tc, tf, coluna, linhas, liga_id=None):
    cc, cf = f"{coluna}_casa", f"{coluna}_fora"
    d = df.dropna(subset=[cc, cf]).copy()
    if d.empty: return {"erro": "sem historico"}
    media_c = d[cc].mean()
    media_f = d[cf].mean()
    is_selecao = liga_id in SELECOES if liga_id else False

    if is_selecao:
        atq_c, _ = _ataque_geral(d, tc, coluna)
        dfn_f, _ = _defesa_geral(d, tf, coluna)
        atq_f, _ = _ataque_geral(d, tf, coluna)
        dfn_c, _ = _defesa_geral(d, tc, coluna)
    else:
        atq_c, _ = _ataque_casa_fora(d, tc, "casa", coluna)
        dfn_f, _ = _ataque_casa_fora(d, tf, "fora", coluna)
        atq_f, _ = _ataque_casa_fora(d, tf, "fora", coluna)
        dfn_c, _ = _ataque_casa_fora(d, tc, "casa", coluna)

    lam = media_c * (atq_c / media_c if media_c > 0 else 1) * (dfn_f / media_c if media_c > 0 else 1)
    mu  = media_f * (atq_f / media_f if media_f > 0 else 1) * (dfn_c / media_f if media_f > 0 else 1)
    alpha = estimar_alpha(pd.concat([d[cc], d[cf]]).dropna().values)

    max_k = 25
    pmf_c = np.array([nb_pmf(i, lam, alpha) for i in range(max_k + 1)])
    pmf_f = np.array([nb_pmf(j, mu, alpha) for j in range(max_k + 1)])
    M = np.outer(pmf_c, pmf_f)
    M /= M.sum()
    tot = np.add.outer(np.arange(max_k + 1), np.arange(max_k + 1))

    return {
        "lambda_casa": round(lam, 3), "lambda_fora": round(mu, 3),
        "alpha_dispersao": round(alpha, 4),
        "total_esperado": round(lam + mu, 2),
        "over": {f"Over {l}": round(float(M[tot > l].sum()), 4) for l in linhas},
        "odd_justa": {f"Over {l}": round(1 / float(M[tot > l].sum()), 3)
                      if M[tot > l].sum() > 0 else None for l in linhas},
    }

# ---------- CSV ----------
def _csv_valido(path):
    if not path.exists(): return None
    try: df = pd.read_csv(path)
    except Exception: return None
    if "fixture_id" not in df.columns: return None
    df = df.dropna(subset=["fixture_id"]).copy()
    try: df["fixture_id"] = df["fixture_id"].astype(int)
    except Exception: return None
    return df

def carregar_historico():
    if not HIST_PATH.exists(): return pd.DataFrame()
    df = pd.read_csv(HIST_PATH)
    df = df.dropna(subset=["fixture_id"]).copy()
    df["fixture_id"] = df["fixture_id"].astype(int)
    return df

# ---------- COLETA ----------
def coletar_lote(df_jogos, limite_reqs=35, pausa=7.0):
    if df_jogos is None or df_jogos.empty: return pd.DataFrame()
    df_jogos = df_jogos.dropna(subset=["fixture_id"]).copy()
    df_jogos["fixture_id"] = df_jogos["fixture_id"].astype(int)

    antigo = _csv_valido(HIST_PATH)
    if antigo is None:
        antigo, ids_ja = pd.DataFrame(), set()
    else:
        print(f"[info] {len(antigo)} jogos no CSV")
        ids_ja = set(antigo["fixture_id"].tolist())

    novos, reqs = [], 0
    for _, linha in df_jogos.iterrows():
        if reqs >= limite_reqs: break
        fid = int(linha["fixture_id"])
        if fid in ids_ja: continue
        liga = linha.get("liga_nome", "?")
        print(f"[{reqs+1}/{limite_reqs}] id={fid} | {liga} | {linha['time_casa']} x {linha['time_fora']}")
        stats = extrair_stats_jogo(fid, data_jogo=linha.get("data", ""), liga_nome=linha.get("liga_nome", ""))
        reqs += 1
        if stats is not None:
            novos.append(stats)
            print(f"       -> OK ({len(novos)})")
        else:
            print("       -> sem estatisticas")
        if reqs < limite_reqs:
            time.sleep(pausa)

    if novos:
        df_novos = pd.DataFrame(novos)
        if not antigo.empty:
            df_final = pd.concat([antigo, df_novos], ignore_index=True)
            df_final = df_final.dropna(subset=["fixture_id"])
            df_final["fixture_id"] = df_final["fixture_id"].astype(int)
            df_final = df_final.drop_duplicates(subset=["fixture_id"], keep="last")
        else:
            df_final = df_novos
        df_final.to_csv(HIST_PATH, index=False)
        print(f"[ok] {len(novos)} novos. Total: {len(df_final)}")
        return df_final
    else:
        print("[aviso] nada novo")
        return antigo

def coletar_ontem(limite_reqs=35):
    ontem = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"[info] coletando jogos de {ontem}")
    fixtures = get_fixtures(ontem)
    if not fixtures: return pd.DataFrame()
    df = fixtures_para_df(fixtures)
    df = df[df["liga_id"].isin(LIGAS_ALVO.keys())].copy()
    df["liga_nome"] = df["liga_id"].map(LIGAS_ALVO)
    df = df[df["status"].isin(["FT", "AET", "PEN"])]
    print(f"[ok] {len(df)} jogos encerrados das ligas alvo")
    if df.empty: return pd.DataFrame()
    return coletar_lote(df, limite_reqs=limite_reqs)

# ---------- PREVISAO ----------
def prever_jogos_do_dia(data_str=None):
    if data_str is None:
        data_str = datetime.now().strftime("%Y-%m-%d")
    fixtures = get_fixtures(data_str)
    if not fixtures: return pd.DataFrame()
    df_jogos = fixtures_para_df(fixtures)
    df_jogos = df_jogos[df_jogos["liga_id"].isin(LIGAS_ALVO.keys())].copy()
    df_jogos["liga_nome"] = df_jogos["liga_id"].map(LIGAS_ALVO)
    if df_jogos.empty: return pd.DataFrame()
    print(f"[ok] {len(df_jogos)} jogos das ligas alvo")
    df_hist = carregar_historico()

    resultados = []
    for _, jogo in df_jogos.iterrows():
        tc, tf, liga_id = jogo["time_casa"], jogo["time_fora"], jogo["liga_id"]
        linha = {
            "fixture_id": int(jogo["fixture_id"]),
            "liga": jogo["liga_nome"], "liga_id": liga_id,
            "time_casa": tc, "time_fora": tf,
            "status": jogo["status"], "data_jogo": data_str,
        }
        for mercado, linhas in MERCADOS.items():
            r = prever_mercado_v2(df_hist, tc, tf, mercado, linhas, liga_id)
            if r and "erro" not in r:
                linha[f"{mercado}_total"] = r["total_esperado"]
                for l in linhas:
                    linha[f"{mercado}_over_{l}"] = round(r["over"][f"Over {l}"] * 100, 1)
                    linha[f"{mercado}_odd_{l}"] = r["odd_justa"][f"Over {l}"]
            else:
                linha[f"{mercado}_total"] = None
                for l in linhas:
                    linha[f"{mercado}_over_{l}"] = None
                    linha[f"{mercado}_odd_{l}"] = None
        resultados.append(linha)
    return pd.DataFrame(resultados)

# ---------- PREVISOES SALVAS ----------
def _carregar_previsoes():
    if not PREV_PATH.exists(): return pd.DataFrame()
    return pd.read_csv(PREV_PATH)

def salvar_previsoes(df_prev_hoje, data_jogo=None):
    if df_prev_hoje is None or df_prev_hoje.empty: return
    if data_jogo is None:
        data_jogo = datetime.now().strftime("%Y-%m-%d")
    prev_antigas = _carregar_previsoes()
    if not prev_antigas.empty and "data_jogo" in prev_antigas.columns:
        prev_antigas = prev_antigas[prev_antigas["data_jogo"] != data_jogo]

    novas = []
    for _, jogo in df_prev_hoje.iterrows():
        for mercado, linhas in MERCADOS.items():
            for linha in linhas:
                col_over = f"{mercado}_over_{linha}"
                col_odd = f"{mercado}_odd_{linha}"
                col_tot = f"{mercado}_total"
                if col_over not in df_prev_hoje.columns: continue
                novas.append({
                    "data_jogo": data_jogo,
                    "fixture_id": int(jogo["fixture_id"]),
                    "liga": jogo["liga"],
                    "time_casa": jogo["time_casa"],
                    "time_fora": jogo["time_fora"],
                    "mercado": mercado,
                    "linha": linha,
                    "prob_over": jogo.get(col_over),
                    "odd_justa": jogo.get(col_odd),
                    "total_esperado": jogo.get(col_tot),
                    "resultado_real": None,
                    "hit": None,
                })
    if not novas: return
    df_novas = pd.DataFrame(novas)
    df_final = pd.concat([prev_antigas, df_novas], ignore_index=True)
    df_final.to_csv(PREV_PATH, index=False)
    print(f"[ok] {len(novas)} previsoes salvas para {data_jogo}")

def verificar_previsoes(data_jogo=None):
    if data_jogo is None:
        data_jogo = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    df_prev = _carregar_previsoes()
    if df_prev.empty: return
    mask = df_prev["data_jogo"] == data_jogo
    if not mask.any(): return
    df_hist = carregar_historico()
    if df_hist.empty: return

    resultados_map = {}
    for _, row in df_hist.iterrows():
        fid = int(row["fixture_id"])
        resultados_map[fid] = {
            "escanteios": (row.get("escanteios_casa") or 0) + (row.get("escanteios_fora") or 0),
            "chutes":     (row.get("chutes_casa") or 0) + (row.get("chutes_fora") or 0),
            "chutes_gol": (row.get("chutes_gol_casa") or 0) + (row.get("chutes_gol_fora") or 0),
        }

    verificadas, acertos = 0, 0
    for idx, row in df_prev[mask].iterrows():
        fid = int(row["fixture_id"])
        if fid not in resultados_map: continue
        total_real = resultados_map[fid].get(row["mercado"])
        if total_real is None: continue
        hit = 1 if total_real > row["linha"] else 0
        df_prev.at[idx, "resultado_real"] = total_real
        df_prev.at[idx, "hit"] = hit
        verificadas += 1
        acertos += hit

    if verificadas == 0: return
    df_prev.to_csv(PREV_PATH, index=False)

    acc = 100 * acertos / verificadas
    print(f"\n{'='*60}")
    print(f"RELATORIO - {data_jogo}")
    print(f"{'='*60}")
    print(f"Verificadas: {verificadas} | Acertos: {acertos} ({acc:.1f}%)")

# ---------- MAIN ----------
def main():
    print("=" * 60)
    print("ROBO ANALISE FUTEBOL - COLETA DIARIA")
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    print("\n[PASSO 1] Coletando jogos de ontem...")
    coletar_ontem(limite_reqs=35)

    print("\n[PASSO 2] Verificando previsoes de ontem...")
    verificar_previsoes()

    print("\n[PASSO 3] Prevendo jogos de hoje...")
    df_prev_hoje = prever_jogos_do_dia()
    if not df_prev_hoje.empty:
        salvar_previsoes(df_prev_hoje)

    print("\n[FIM]")

if __name__ == "__main__":
    main()
