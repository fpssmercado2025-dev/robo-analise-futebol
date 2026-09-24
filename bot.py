"""
Robo de Analise de Futebol - Bot do Telegram
Responde comandos e envia mensagem automatica diaria.
"""
import os
import io
import requests
import pandas as pd
from datetime import datetime, time as dtime, timedelta, timezone
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ---------- CONFIG ----------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

GITHUB_USER = "fpssmercado2025-dev"
GITHUB_REPO = "robo-analise-futebol"
GITHUB_BRANCH = "main"
RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/data"

HORARIO_ENVIO_BRASILIA = dtime(hour=8, minute=0)  # 08h Brasilia = 11h UTC

# ---------- LEITURA DOS CSVs ----------
def _carregar_csv(nome):
    url = f"{RAW_BASE}/{nome}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and r.text.strip():
            return pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        print(f"[erro] {nome}: {e}")
    return pd.DataFrame()

def carregar_previsoes():
    return _carregar_csv("previsoes.csv")

def carregar_historico():
    return _carregar_csv("historico_stats.csv")

# ---------- FORMATACAO ----------
def formatar_jogos_do_dia(data_alvo=None):
    """Retorna LISTA de mensagens. Uma por liga (ou mais se a liga for grande)."""
    if data_alvo is None:
        data_alvo = datetime.now().strftime("%Y-%m-%d")

    df = carregar_previsoes()
    if df.empty:
        return ["Nenhuma previsao encontrada no repositorio."]

    mask = df["data_jogo"] == data_alvo
    if not mask.any():
        return [f"Nenhum jogo das ligas alvo em {data_alvo}."]

    df_dia = df[mask].copy()

    emojis = {"escanteios": "\U0001F6A9", "chutes": "\U0001F3AF", "chutes_gol": "\U0001F945"}
    nomes = {"escanteios": "Escanteios", "chutes": "Chutes", "chutes_gol": "Chutes no gol"}

    mensagens = []

    for liga in sorted(df_dia["liga"].unique()):
        df_liga = df_dia[df_dia["liga"] == liga]
        fixtures = df_liga.drop_duplicates(subset=["fixture_id"])

        header = f"\U0001F3C6 *{liga}*\n" + "=" * 30 + "\n\n"
        atual = header

        for _, fix in fixtures.iterrows():
            bloco = f"\u26bd {fix['time_casa']} x {fix['time_fora']}\n"
            df_fix = df_liga[df_liga["fixture_id"] == fix["fixture_id"]]

            for mercado in ["escanteios", "chutes", "chutes_gol"]:
                df_merc = df_fix[df_fix["mercado"] == mercado]
                if df_merc.empty:
                    continue
                total = df_merc.iloc[0].get("total_esperado", "?")
                bloco += f"  {emojis[mercado]} {nomes[mercado]} (total: {total})\n"
                for _, row in df_merc.iterrows():
                    prob = row.get("prob_over")
                    odd = row.get("odd_justa")
                    if pd.notna(prob):
                        bloco += f"     Over {row['linha']}: {prob:.0f}% (odd {odd})\n"
            bloco += "\n"

            # Se o bloco do jogo estourar o limite, fecha e abre novo
            if len(atual) + len(bloco) > 3800:
                mensagens.append(atual)
                atual = f"\U0001F3C6 *{liga}* (continuacao)\n" + "=" * 30 + "\n\n" + bloco
            else:
                atual += bloco

        if atual.strip():
            mensagens.append(atual)

    return mensagens


def formatar_parcial():
    df = carregar_previsoes()
    if df.empty:
        return "Nenhuma previsao salva ainda."

    df_ver = df.dropna(subset=["hit"])
    if df_ver.empty:
        return "Nenhuma previsao verificada ainda. Aguarde os jogos terminarem."

    total = len(df_ver)
    acertos = int(df_ver["hit"].sum())
    acc = 100 * acertos / total

    msg = f"📊 *PARCIAL DE ACERTOS*\n"
    msg += "=" * 30 + "\n\n"
    msg += f"Total verificado: *{total}*\n"
    msg += f"Acertos: *{acertos}* ({acc:.1f}%)\n"
    msg += f"Erros: *{total - acertos}* ({100-acc:.1f}%)\n\n"

    msg += "*Por mercado:*\n"
    for m in ["escanteios", "chutes", "chutes_gol"]:
        sub = df_ver[df_ver["mercado"] == m]
        if not sub.empty:
            a = int(sub["hit"].sum())
            t = len(sub)
            p = 100 * a / t
            emoji = "🔥" if p >= 75 else "✅" if p >= 65 else "⚠️" if p >= 55 else "❌"
            msg += f"  {emoji} {m}: {a}/{t} = {p:.1f}%\n"

    msg += "\n*Top linhas (acumulado):*\n"
    ranking = []
    for linha in sorted(df_ver["linha"].unique()):
        sub = df_ver[df_ver["linha"] == linha]
        if not sub.empty:
            p = 100 * sub["hit"].mean()
            ranking.append((linha, p, len(sub)))
    ranking.sort(key=lambda x: -x[1])
    for linha, p, n in ranking[:8]:
        msg += f"  Over {linha}: {p:.1f}% ({n} prev)\n"

    return msg


# ---------- COMANDOS ----------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = "🤖 *Robo de Analise de Futebol*\n\n"
    msg += "Comandos disponiveis:\n"
    msg += "  /jogos - Previsoes dos jogos de hoje\n"
    msg += "  /parcial - Parcial acumulada de acertos\n"
    msg += "  /help - Esta mensagem\n\n"
    msg += "Voce tambem recebe automaticamente todos os dias as 10h."
    await update.message.reply_text(msg)


async def cmd_jogos(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for msg in formatar_jogos_do_dia():
        await update.message.reply_text(msg, parse_mode="Markdown")


async def cmd_parcial(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(formatar_parcial())


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_start(update, context)


# ---------- ENVIO AUTOMATICO ----------
async def envio_automatico(context: ContextTypes.DEFAULT_TYPE):
    if not TELEGRAM_CHAT_ID:
        print("[aviso] TELEGRAM_CHAT_ID nao configurado")
        return
    try:
        for msg in formatar_jogos_do_dia():
            await context.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode="Markdown",
            )
        print(f"[ok] mensagem automatica enviada")
    except Exception as e:
        print(f"[erro] envio automatico: {e}")


# ---------- MAIN ----------
def main():
    if not TELEGRAM_TOKEN:
        print("[erro] TELEGRAM_TOKEN nao configurado")
        return

    print("[info] iniciando bot...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("jogos", cmd_jogos))
    app.add_handler(CommandHandler("parcial", cmd_parcial))

    # Agendamento diario (10h Brasilia = 13h UTC)
    horario_utc = dtime(hour=11, minute=0, tzinfo=timezone.utc)
    app.job_queue.run_daily(envio_automatico, time=horario_utc)

    print("[ok] bot rodando (polling)")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
