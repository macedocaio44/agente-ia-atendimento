import os
import json
import re
from pathlib import Path
from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI
from openai import OpenAI
from dotenv import load_dotenv

# =========================
# BOOT
# =========================
load_dotenv()
app = FastAPI()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN", "").strip(),
)

# =========================
# MEMÓRIA DE SESSÕES (RAM)
# =========================
SESSOES = {}  # {session_id: {"empresa":..., "loja":..., "historico":[...] } }

# =========================
# SAUDAÇÃO POR HORA
# =========================
def saudacao_por_hora() -> str:
    h = datetime.now().hour
    if 5 <= h < 12:
        return "Bom dia"
    if 12 <= h < 18:
        return "Boa tarde"
    return "Boa noite"

# =========================
# ARQUIVOS (manual + config)
# =========================
def carregar_manual() -> str:
    manual_path = Path(__file__).parent / "manual.txt"
    if not manual_path.exists():
        return ""
    return manual_path.read_text(encoding="utf-8").strip()

def carregar_config() -> dict:
    cfg_path = Path(__file__).parent / "cliente_config.json"
    if not cfg_path.exists():
        return {
            "cliente_nome": "Efinance",
            "produto_nome": "ERP/CRM",
            "prompts": {
                "suporte": [
                    "Você é um atendente humano de SUPORTE TÉCNICO da empresa {cliente_nome}.",
                    "Produto: {produto_nome}.",
                    "Responda em português do Brasil.",
                    "Nunca diga que é robô/IA.",
                    "Seja objetivo (máx 8 linhas). Quando precisar, faça até 3 perguntas.",
                    "Quando orientar, use passos numerados."
                ],
                "financeiro": [
                    "Você é um atendente humano do FINANCEIRO da empresa {cliente_nome}.",
                    "Responda em português do Brasil.",
                    "Ajude com boletos, notas fiscais, vencimentos, cobranças e negociação.",
                    "Faça até 3 perguntas: CNPJ/CPF, número do pedido e e-mail cadastrado.",
                    "Nunca diga que é robô/IA."
                ],
                "vendas": [
                    "Você é um atendente humano de VENDAS da empresa {cliente_nome}.",
                    "Responda em português do Brasil.",
                    "Seu objetivo é entender a necessidade e encaminhar para proposta/demonstração.",
                    "Faça até 3 perguntas (tamanho da empresa, módulos, prazo).",
                    "Nunca diga que é robô/IA."
                ]
            }
        }
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

MANUAL_TEXTO = carregar_manual()
CFG = carregar_config()

# =========================
# REGRAS FIXAS
# =========================
REGRAS_FIXAS_SUPORTE = (
    "REGRAS GERAIS DE ATENDIMENTO (OBRIGATÓRIO):\n"
    "1) Cumprimente o cliente de forma educada e humanizada.\n"
    "2) Identifique-se como atendente de suporte da Efinance.\n"
    "3) ANTES de qualquer orientação técnica, solicite o número da EMPRESA e o número da LOJA.\n"
    "4) Se o cliente não informar EMPRESA e LOJA, NÃO avance no atendimento técnico. Reforce a necessidade dos dados.\n"
)

def montar_prompt(modo: str) -> str:
    modo = (modo or "suporte").lower().strip()
    prompts = CFG.get("prompts", {})
    if modo not in prompts:
        modo = "suporte"

    linhas = prompts.get(modo, [])
    prompt_cliente = "\n".join(linhas).strip().format(
        cliente_nome=CFG.get("cliente_nome", "Efinance"),
        produto_nome=CFG.get("produto_nome", "ERP/CRM"),
    )

    partes = []
    if modo == "suporte":
        partes.append(REGRAS_FIXAS_SUPORTE)

    if MANUAL_TEXTO:
        partes.append("MANUAL / BASE DE CONHECIMENTO:\n" + MANUAL_TEXTO)

    partes.append("INSTRUÇÕES DO ATENDENTE:\n" + prompt_cliente)

    return "\n\n---\n\n".join(partes)

# =========================
# EXTRAÇÃO EMPRESA/LOJA
# =========================
def extrair_empresa_loja(texto: str):
    t = (texto or "").lower()
    m_emp = re.search(r"(empresa)\s*[:=]?\s*(\d{1,10})", t)
    m_loja = re.search(r"(loja)\s*[:=]?\s*(\d{1,10})", t)
    empresa = m_emp.group(2) if m_emp else None
    loja = m_loja.group(2) if m_loja else None
    return empresa, loja

def resposta_pedir_empresa_loja():
    saudacao = saudacao_por_hora()
    return (
        f"Efinance, {saudacao}! 😊\n"
        "Para eu identificar seu cadastro e seguir com o suporte, por favor me informe:\n"
        "1) Número da EMPRESA\n"
        "2) Número da LOJA\n\n"
        "Exemplo: empresa 123 loja 4"
    )

# =========================
# CLASSIFICAÇÃO AUTOMÁTICA
# =========================
def classificar_modo(texto: str) -> str:
    t = (texto or "").lower()

    if any(p in t for p in ["boleto", "pagamento", "nota fiscal", "nf", "vencimento", "cobrança", "fatura"]):
        return "financeiro"

    if any(p in t for p in ["preço", "valor", "plano", "proposta", "contratar", "comprar", "licença", "demonstração"]):
        return "vendas"

    return "suporte"

# =========================
# ROTAS
# =========================
@app.get("/")
def home():
    return {"status": "Agente IA rodando corretamente"}

@app.get("/pergunta")
def perguntar(texto: str, session_id: str | None = None):
    # gera session_id se não vier
    if not session_id:
        session_id = str(uuid4())

    # cria sessão
    if session_id not in SESSOES:
        SESSOES[session_id] = {"empresa": None, "loja": None, "historico": []}

    sessao = SESSOES[session_id]

    # modo automático
    modo_ok = classificar_modo(texto)

    # trava empresa/loja só no suporte
    if modo_ok == "suporte":
        emp, lj = extrair_empresa_loja(texto)
        if emp:
            sessao["empresa"] = emp
        if lj:
            sessao["loja"] = lj

        if not sessao["empresa"] or not sessao["loja"]:
            return {"session_id": session_id, "modo": modo_ok, "resposta": resposta_pedir_empresa_loja()}

    prompt_sistema = montar_prompt(modo_ok)

    contexto_cliente = ""
    if sessao["empresa"] and sessao["loja"]:
        contexto_cliente = f"Cliente identificado: empresa {sessao['empresa']}, loja {sessao['loja']}."

    mensagens = [{"role": "system", "content": prompt_sistema}]
    if contexto_cliente:
        mensagens.append({"role": "system", "content": contexto_cliente})

    # histórico curto
    for h in sessao["historico"][-6:]:
        mensagens.append(h)

    mensagens.append({"role": "user", "content": texto})

    r = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        temperature=0.3,
        max_tokens=260,
        messages=mensagens,
    )

    resposta = r.choices[0].message.content

    # salva histórico
    sessao["historico"].append({"role": "user", "content": texto})
    sessao["historico"].append({"role": "assistant", "content": resposta})

    return {"session_id": session_id, "modo": modo_ok, "resposta": resposta}