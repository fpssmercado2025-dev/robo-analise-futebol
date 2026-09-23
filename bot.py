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

HORARIO_ENVIO_BRASILIA = dtime(hour=10, minute=0)  # 10h Brasilia = 13h UTC

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
    if data_alvo is None:
        data_alvo = datetime.now().strftime("%Y-%m-%d")

    df = carregar_previsoes()
    if df.empty:
        return f"Nenhuma previsao encontrada no repositorio."

    mask = df["data_jogo"] == data_alvo
    if not mask.any():
        return f"Nenhum jogo das ligas alvo em {data_alvo}."

    df_dia = df[mask].drop_duplicates(subset=["fixture_id"])

    msg = f"⚽ *JOGOS DE HOJE* ({data_alvo})\n"
    msg += "=" * 30 + "\n\n"

    for _, jogo in df_dia.iterrows():
        msg += f"*{jogo['liga']}*\n"
        msg += f"  {jogo['time_casa']} x {jogo['time_fora']}\n"

        # Escanteios
        if pd.notna(jogo.get("escanteios_total")):
            msg += f"  🚩 Escanteios (total: {jogo['escanteios_total']})\n"
            for l in [7.5, 8.5, 9.5, 10.5]:
                col_o = f"escanteios_over_{l}"
                col_odd = f"escanteios_odd_{l}"
                if col_o in jogo and pd.notna(jogo[col_o]):
                    msg += f"     Over {l}: {jogo[col_o]:.0f}% (odd {jogo[col_odd]})\n"

        # Chutes
        if pd.notna(jogo.get("chutes_total")):
            msg += f"  🎯 Chutes (total: {jogo['chutes_total']})\n"
            for l in [18.5, 20.5, 22.5, 24.5]:
                col_o = f"chutes_over_{l}"
                col_odd = f"chutes_odd_{l}"
                if col_o in jogo and pd.notna(jogo[col_o]):
                    msg += f"     Over {l}: {jogo[col_o]:.0f}% (odd {jogo[col_odd]})\n"

        # Chutes no gol
        if pd.notna(jogo.get("chutes_gol_total")):
            msg += f"  🥅 Chutes no gol (total: {jogo['chutes_gol_total']})\n"
            for l in [6.5, 7.5, 8.5, 9.5]:
                col_o = f"chutes_gol_over_{l}"
                col_odd = f"chutes_gol_odd_{l}"
                if col_o in jogo and pd.notna(jogo[col_o]):
                    msg += f"     Over {l}: {jogo[col_o]:.0f}% (odd {jogo[col_odd]})\n"

        msg += "\n"

    return msg


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
    await update.message.reply_markdown(msg)


async def cmd_jogos(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_markdown(formatar_jogos_do_dia())


async def cmd_parcial(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_markdown(formatar_parcial())


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_start(update, context)


# ---------- ENVIO AUTOMATICO ----------
async def envio_automatico(context: ContextTypes.DEFAULT_TYPE):
    if not TELEGRAM_CHAT_ID:
        print("[aviso] TELEGRAM_CHAT_ID nao configurado")
        return
    try:
        msg = formatar_jogos_do_dia()
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
    horario_utc = dtime(hour=13, minute=0, tzinfo=timezone.utc)
    app.job_queue.run_daily(envio_automatico, time=horario_utc)

    print("[ok] bot rodando (polling)")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
