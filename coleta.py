"""
Robo de Analise de Futebol - Coleta Diaria (v3.1 - Bzzoiro)
Roda no GitHub Actions todo dia as 07h Brasilia.
"""
import os
import time
import unicodedata
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from scipy.stats import poisson, nbinom
from scipy.optimize import minimize

# ---------- CONFIG ----------
BZZOIRO_TOKEN = os.environ.get("BZZOIRO_TOKEN", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

BZZOIRO_BASE = "https://sports.bzzoiro.com/api"
TIMEZONE = "America/Sao_Paulo"
TZ = ZoneInfo(TIMEZONE)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

HIST_PATH = DATA_DIR / "historico_stats.csv"
PREV_PATH = DATA_DIR / "previsoes.csv"

# Minimo de jogos que pelo menos UM dos times precisa ter no historico
MIN_JOGOS_HISTORICO = 1

LIGAS_ALVO_NOMES = [
    # Clubes Europa
    "Champions League",
    "Europa League",
    "Conference League",
    "Premier League",
    "La Liga",
    "Serie A",
    "Bundesliga",
    "Ligue 1",
    # Clubes Brasil / America do Sul
    "Brasileirão Serie A",
    "Brasileirão Serie B",
    "Copa Libertadores",
    "Copa Sudamericana",
    "Copa do Brasil",
    # Copas nacionais
    "Copa del Rey",
    "Coppa Italia",
    "Coupe de France",
    # Selecoes (datas FIFA)
    "UEFA Nations League",
    "CONCACAF Nations League",
    "World Cup",
    "Euro",
    "Copa America",
    "Africa Cup",
    "AFC Asian Cup",
    "WC Qualifiers",
]

LIGAS_EXCLUIR = [
    "CAF Champions League",
    "AFC Champions League",
    "Women",
    "Feminino",
    "Club Friendlies",
    "International Friendly",
]

MERCADOS = {
    "escanteios": (7.5, 8.5, 9.5, 10.5),
    "chutes":     (18.5, 20.5, 22.5, 24.5),
    "chutes_gol": (6.5, 7.5, 8.5, 9.5),
}

NOMES_SELECOES = [
    "World Cup", "Euro", "Copa America", "Nations League",
    "Africa Cup", "Asian Cup", "WC Qualifiers",
]


def _normalize(s):
    """Remove acentos e baixa caixa. Ex: 'Grêmio' -> 'gremio'."""
    if s is None:
        return ""
    return unicodedata.normalize("NFKD", str(s)).encode("ASCII", "ignore").decode("ascii").lower().strip()


def _liga_e_alvo(nome):
    if not nome:
        return False
    n = nome.lower()
    for exc in LIGAS_EXCLUIR:
        if exc.lower() in n:
            return False
    for alvo in LIGAS_ALVO_NOMES:
        if alvo.lower() in n:
            return True
    return False


def _liga_e_selecao(nome):
    if not nome:
        return False
    n = nome.lower()
    for s in NOMES_SELECOES:
        if s.lower() in n:
            return True
    return False


# ---------- TELEGRAM ALERT ----------
def enviar_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[aviso] Telegram nao configurado. Msg: {msg}")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        r = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg,
            "parse_mode": "Markdown",
        }, timeout=15)
        if r.status_code != 200:
            print(f"[erro telegram] {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"[erro telegram] {e}")


# ---------- BZZOIRO API ----------
def _headers():
    return {"Authorization": f"Token {BZZOIRO_TOKEN}"}


def chamar_api(caminho, params=None, tentativas=3):
    url = f"{BZZOIRO_BASE}{caminho}"
    for i in range(tentativas):
        try:
            r = requests.get(url, headers=_headers(), params=params, timeout=30)
            if r.status_code == 401:
                return None, "Token Bzzoiro invalido (401)"
            if r.status_code == 403:
                return None, "Acesso negado pela Bzzoiro (403)"
            if r.status_code == 404:
                return None, f"Endpoint nao encontrado: {caminho}"
            if r.status_code == 429:
                time.sleep(5 + i * 5)
                continue
            if r.status_code >= 500:
                time.sleep(2 ** i)
                continue
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict) and data.get("error"):
                    return None, f"Erro da API: {data.get('detail', data)}"
                return data, None
            return None, f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            print(f"[erro request] {e}")
            time.sleep(2 ** i)
    return None, "Falha apos tentativas"


def buscar_eventos(data_str):
    todos = []
    offset = 0
    limite = 100
    while True:
        params = {
            "date_from": data_str,
            "date_to": data_str,
            "limit": limite,
            "offset": offset,
        }
        data, erro = chamar_api("/events/", params)
        if erro:
            return None, erro
        if not data:
            return [], None
        results = data.get("results", [])
        todos.extend(results)
        if not data.get("next"):
            break
        offset += limite
        if offset > 2000:
            break
    return todos, None


def buscar_stats(evento_id):
    data, erro = chamar_api(f"/v2/events/{evento_id}/stats/")
    if erro:
        return None, erro
    return data, None


def extrair_stats_evento(evento, stats_data, data_jogo):
    if not stats_data:
        return None
    stats = stats_data.get("stats", {})
    home = stats.get("home", {})
    away = stats.get("away", {})

    def val(d, k):
        v = d.get(k)
        if isinstance(v, dict):
            return v.get("value")
        return v

    return {
        "fixture_id": evento["id"],
        "time_casa": evento["home_team"],
        "time_fora": evento["away_team"],
        "escanteios_casa": val(home, "corner_kicks"),
        "escanteios_fora": val(away, "corner_kicks"),
        "chutes_casa": val(home, "total_shots"),
        "chutes_fora": val(away, "total_shots"),
        "chutes_gol_casa": val(home, "shots_on_target"),
        "chutes_gol_fora": val(away, "shots_on_target"),
        "data_jogo": data_jogo,
        "liga_nome": evento.get("league", {}).get("name", ""),
    }


# ---------- CSV ----------
def carregar_historico():
    if not HIST_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(HIST_PATH)
    df = df.dropna(subset=["fixture_id"]).copy()
    df["fixture_id"] = df["fixture_id"].astype(int)
    return df


def _ids_ja_coletados():
    df = carregar_historico()
    if df.empty:
        return set()
    return set(df["fixture_id"].tolist())


def salvar_historico(df_novos, df_antigo):
    if df_novos is None or df_novos.empty:
        return df_antigo
    if df_antigo is None or df_antigo.empty:
        df_final = df_novos
    else:
        df_final = pd.concat([df_antigo, df_novos], ignore_index=True)
        df_final = df_final.dropna(subset=["fixture_id"])
        df_final["fixture_id"] = df_final["fixture_id"].astype(int)
        df_final = df_final.drop_duplicates(subset=["fixture_id"], keep="last")
    df_final.to_csv(HIST_PATH, index=False)
    return df_final


def _jogos_do_time(df_hist, time):
    """Conta quantos jogos o time tem no historico (ignora acentos)."""
    if df_hist.empty or not time:
        return 0
    t = _normalize(time)
    mask = (
        (df_hist["time_casa"].apply(_normalize) == t) |
        (df_hist["time_fora"].apply(_normalize) == t)
    )
    return int(mask.sum())


def _tem_historico_suficiente(df_hist, tc, tf, minimo=MIN_JOGOS_HISTORICO):
    """Retorna (tem, n_casa, n_fora)."""
    n_casa = _jogos_do_time(df_hist, tc)
    n_fora = _jogos_do_time(df_hist, tf)
    tem = n_casa >= minimo or n_fora >= minimo
    return tem, n_casa, n_fora


# ---------- COLETA DE ONTEM ----------
def coletar_ontem():
    agora = datetime.now(TZ)
    ontem = (agora - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"[info] coletando jogos de {ontem}")

    eventos, erro = buscar_eventos(ontem)
    if erro:
        return 0, f"Erro ao buscar eventos: {erro}"
    if not eventos:
        return 0, None

    alvo = [e for e in eventos if _liga_e_alvo(e.get("league", {}).get("name", ""))]
    finalizados = [e for e in alvo if e.get("status") in ("finished", "FT", "AET", "PEN")]
    print(f"[info] {len(eventos)} totais | {len(alvo)} das ligas alvo | {len(finalizados)} finalizados")

    if not finalizados:
        return 0, None

    ids_ja = _ids_ja_coletados()
    df_antigo = carregar_historico()

    novos = []
    for ev in finalizados:
        if ev["id"] in ids_ja:
            continue
        liga = ev.get("league", {}).get("name", "?")
        print(f"  [coletando] id={ev['id']} | {liga} | {ev['home_team']} x {ev['away_team']}")
        stats_data, erro = buscar_stats(ev["id"])
        if erro:
            print(f"    [erro] {erro}")
            continue
        linha = extrair_stats_evento(ev, stats_data, ontem)
        if linha:
            novos.append(linha)
            print(f"    [ok]")
        time.sleep(0.5)

    if not novos:
        print("[aviso] nenhum jogo novo coletado")
        return 0, None

    df_novos = pd.DataFrame(novos)
    df_final = salvar_historico(df_novos, df_antigo)
    print(f"[ok] {len(novos)} novos jogos. Total: {len(df_final)}")
    return len(novos), None


# ---------- MODELO ----------
def nb_pmf(k, mu, alpha):
    if alpha <= 1e-6:
        return poisson.pmf(k, mu)
    n = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)
    return nbinom.pmf(k, n, p)


def estimar_alpha(valores, alpha_prior=0.10, min_obs=20):
    v = np.asarray(valores, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < min_obs:
        return alpha_prior
    mu = v.mean()
    if mu <= 0:
        return alpha_prior
    def nll(a):
        a = a[0]
        if a < 0:
            return 1e10
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
    if n == 0:
        return media, 0
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
    if n == 0:
        return media, 0
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
    if n == 0:
        return media, 0
    peso = min(n / min_jogos, 1.0)
    return peso * val.mean() + (1 - peso) * media, n


def prever_mercado_v2(df, tc, tf, coluna, linhas, liga_nome=None):
    cc, cf = f"{coluna}_casa", f"{coluna}_fora"
    d = df.dropna(subset=[cc, cf]).copy()
    if d.empty:
        return {"erro": "sem historico"}
    media_c = d[cc].mean()
    media_f = d[cf].mean()
    is_selecao = _liga_e_selecao(liga_nome) if liga_nome else False

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
        "lambda_casa": round(lam, 3),
        "lambda_fora": round(mu, 3),
        "alpha_dispersao": round(alpha, 4),
        "total_esperado": round(lam + mu, 2),
        "over": {f"Over {l}": round(float(M[tot > l].sum()), 4) for l in linhas},
        "odd_justa": {f"Over {l}": round(1 / float(M[tot > l].sum()), 3)
                      if M[tot > l].sum() > 0 else None for l in linhas},
    }


# ---------- PREVISAO ----------
def prever_jogos_do_dia():
    hoje = datetime.now(TZ).strftime("%Y-%m-%d")
    print(f"[info] buscando jogos de {hoje}")
    eventos, erro = buscar_eventos(hoje)
    if erro:
        return pd.DataFrame(), erro
    if not eventos:
        return pd.DataFrame(), None

    alvo = [e for e in eventos if _liga_e_alvo(e.get("league", {}).get("name", ""))]
    print(f"[info] {len(alvo)} jogos das ligas alvo hoje")
    if not alvo:
        return pd.DataFrame(), None

    df_hist = carregar_historico()
    if df_hist.empty:
        return pd.DataFrame(), "historico vazio"

    resultados = []
    pulados = 0
    for ev in alvo:
        tc = ev["home_team"]
        tf = ev["away_team"]
        liga_nome = ev.get("league", {}).get("name", "")

        tem, n_c, n_f = _tem_historico_suficiente(df_hist, tc, tf)
        if not tem:
            print(f"  [skip] {tc} ({n_c}j) x {tf} ({n_f}j) - sem historico")
            pulados += 1
            continue

        linha = {
            "fixture_id": ev["id"],
            "liga": liga_nome,
            "liga_id": ev.get("league", {}).get("id"),
            "time_casa": tc,
            "time_fora": tf,
            "status": ev.get("status"),
            "data_jogo": hoje,
        }
        for mercado, linhas in MERCADOS.items():
            r = prever_mercado_v2(df_hist, tc, tf, mercado, linhas, liga_nome)
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

    print(f"[info] {len(resultados)} jogos com previsao | {pulados} pulados por falta de historico")
    return pd.DataFrame(resultados), None


# ---------- PREVISOES SALVAS ----------
def _carregar_previsoes():
    if not PREV_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(PREV_PATH)


def salvar_previsoes(df_prev_hoje):
    if df_prev_hoje is None or df_prev_hoje.empty:
        return
    hoje = datetime.now(TZ).strftime("%Y-%m-%d")
    prev_antigas = _carregar_previsoes()
    if not prev_antigas.empty and "data_jogo" in prev_antigas.columns:
        prev_antigas = prev_antigas[prev_antigas["data_jogo"] != hoje]

    novas = []
    for _, jogo in df_prev_hoje.iterrows():
        for mercado, linhas in MERCADOS.items():
            for linha in linhas:
                col_over = f"{mercado}_over_{linha}"
                col_odd = f"{mercado}_odd_{linha}"
                col_tot = f"{mercado}_total"
                if col_over not in df_prev_hoje.columns:
                    continue
                novas.append({
                    "data_jogo": hoje,
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
    if not novas:
        return
    df_novas = pd.DataFrame(novas)
    df_final = pd.concat([prev_antigas, df_novas], ignore_index=True)
    df_final.to_csv(PREV_PATH, index=False)
    print(f"[ok] {len(novas)} previsoes salvas para {hoje}")


def verificar_previsoes():
    ontem = (datetime.now(TZ) - timedelta(days=1)).strftime("%Y-%m-%d")
    df_prev = _carregar_previsoes()
    if df_prev.empty:
        return
    mask = df_prev["data_jogo"] == ontem
    if not mask.any():
        return
    df_hist = carregar_historico()
    if df_hist.empty:
        return

    resultados_map = {}
    for _, row in df_hist.iterrows():
        fid = int(row["fixture_id"])
        resultados_map[fid] = {
            "escanteios": (row.get("escanteios_casa") or 0) + (row.get("escanteios_fora") or 0),
            "chutes": (row.get("chutes_casa") or 0) + (row.get("chutes_fora") or 0),
            "chutes_gol": (row.get("chutes_gol_casa") or 0) + (row.get("chutes_gol_fora") or 0),
        }

    verificadas, acertos = 0, 0
    for idx, row in df_prev[mask].iterrows():
        fid = int(row["fixture_id"])
        if fid not in resultados_map:
            continue
        total_real = resultados_map[fid].get(row["mercado"])
        if total_real is None:
            continue
        hit = 1 if total_real > row["linha"] else 0
        df_prev.at[idx, "resultado_real"] = total_real
        df_prev.at[idx, "hit"] = hit
        verificadas += 1
        acertos += hit

    if verificadas == 0:
        return
    df_prev.to_csv(PREV_PATH, index=False)
    acc = 100 * acertos / verificadas
    print(f"[ok] {verificadas} previsoes verificadas. Acertos: {acertos} ({acc:.1f}%)")


# ---------- MAIN ----------
def main():
    print("=" * 60)
    print("ROBO ANALISE FUTEBOL - COLETA DIARIA (Bzzoiro)")
    print(f"Data: {datetime.now(TZ).strftime('%Y-%m-%d %H:%M')} ({TIMEZONE})")
    print("=" * 60)

    if not BZZOIRO_TOKEN:
        msg = "❌ *Coleta abortada:* BZZOIRO_TOKEN nao configurado."
        print(msg)
        enviar_telegram(msg)
        return

    erros = []
    n_novos = 0

    print("\n[PASSO 1] Coletando jogos de ontem...")
    try:
        n_novos, erro = coletar_ontem()
        if erro:
            erros.append(f"Coleta: {erro}")
    except Exception as e:
        erros.append(f"Coleta: excecao {e}")

    print("\n[PASSO 2] Verificando previsoes de ontem...")
    try:
        verificar_previsoes()
    except Exception as e:
        erros.append(f"Verificacao: {e}")

    print("\n[PASSO 3] Prevendo jogos de hoje...")
    df_prev = pd.DataFrame()
    try:
        df_prev, erro = prever_jogos_do_dia()
        if erro:
            erros.append(f"Previsao: {erro}")
        if not df_prev.empty:
            salvar_previsoes(df_prev)
    except Exception as e:
        erros.append(f"Previsao: excecao {e}")

    if erros:
        msg = "⚠️ *Coleta com problemas*\n\n" + "\n".join(f"• {e}" for e in erros)
        enviar_telegram(msg)
        print(f"\n[ALERTA ENVIADO] {len(erros)} problema(s)")

    print(f"\n[FIM] Coleta: {n_novos} jogos | Previsoes: {len(df_prev)} jogos")


if __name__ == "__main__":
    main()
