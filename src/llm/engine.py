# src/llm/engine.py
from __future__ import annotations
from typing import Dict, Optional
import os, requests, json, time
from src.telemetry.metrics import observe_hist, llm_latency, llm_first_token_latency, wrap_stream_for_first_token

class LLMLocal:
    def __init__(self, cfg: Dict, logger):
        self.cfg = cfg or {}
        self.log = logger

        provider = (self.cfg.get("provider") or self.cfg.get("backend") or "rule").lower()
        if provider == "echo":
            provider = "rule"
        self.provider = provider

        self.system = self.cfg.get("system_prompt", "")
        self.host = self.cfg.get("host", "http://localhost:11434")
        self.model = self.cfg.get("model", "llama3.2")
        self.max_tokens = int(self.cfg.get("max_tokens", 120))
        self.temperature = float(self.cfg.get("temperature", 0.4))

        self.default_mode = (self.cfg.get("default_mode") or "precise").lower()
        self.strict_facts = bool(self.cfg.get("strict_facts", True))

        self._openai = None
        if self.provider == "openai":
            try:
                from openai import OpenAI
                self._openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except Exception as e:
                self.log.error(f"OpenAI client indisponibil: {e}. Revin pe 'rule'.")
                self.provider = "rule"

        self.log.info(f"LLM provider activ: {self.provider}")

    def generate(self, user_text: str, lang_hint: str = "en", mode: Optional[str] = None) -> str:
        mode = (mode or self.default_mode).lower()
        with observe_hist(llm_latency):
            if self.provider == "rule":
                return self._rule_based(user_text, lang_hint)
            if self.provider == "ollama":
                return self._ollama_http(user_text, lang_hint, mode=mode)
            if self.provider == "openai" and self._openai:
                return self._openai_chat(user_text, lang_hint)
            return "No LLM provider configured."

    def generate_stream(self, user_text: str, lang_hint: str = "en", mode: Optional[str] = None):
        mode = (mode or self.default_mode).lower()
        if self.provider == "ollama":
            gen = self._ollama_stream(user_text, lang_hint, mode)
            return wrap_stream_for_first_token(gen, llm_first_token_latency)
        def _one():
            yield self.generate(user_text, lang_hint, mode)
        return _one()

    def _rule_based(self, user_text: str, lang_hint: str) -> str:
        if not (user_text or "").strip():
            return "Nu am auzit întrebarea. Poți repeta?"
        return f"{'Am înțeles' if lang_hint.startswith('ro') else 'I heard'}: \"{user_text}\"."

    def _ollama_http(self, user_text: str, lang_hint: str, mode: str = "precise") -> str:
        unknown_en = "That’s outside my current knowledge, but I’ll note it for improvement."
        unknown_ro = "Interesant, Nu am răspunsul încă, dar exact întrebări ca asta mă ajută să devin mai bun."
        unknown = unknown_ro if str(lang_hint).lower().startswith("ro") else unknown_en

        url = f"{self.host.rstrip('/')}/api/generate"

        if mode == "precise":
            safety = (
                "IMPORTANT: Answer only with verified facts. "
                f"If you are uncertain or the information may be outdated, reply exactly with: '{unknown}' and suggest checking a reliable source. "
                "Never invent names, dates, or sources. Be concise."
            )
            temperature = 0.0; top_p = 0.9; top_k = 40
        else:
            safety = "Be helpful and friendly."
            temperature = self.temperature; top_p = 0.95; top_k = 50

        sys = (self.system or "").strip()
        preface = f"{sys}\n{safety}".strip()
        prompt = f"{preface}\nUser ({lang_hint}): {user_text}\nAssistant:"

        try:
            resp = requests.post(url, json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repeat_penalty": 1.1,
                    "num_predict": self.max_tokens
                }
            }, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("response") or "").strip()
            if self.strict_facts and not text:
                return unknown
            return text or "…"
        except Exception as e:
            self.log.error(f"Ollama HTTP error: {e}")
            return self._rule_based(user_text, lang_hint)

    def _ollama_stream(self, user_text: str, lang_hint: str, mode: str = "precise"):
        unknown_en = "That’s outside my current knowledge, but I’ll note it for improvement."
        unknown_ro = "Interesant, Nu am răspunsul încă, dar exact întrebări ca asta mă ajută să devin mai bun."
        unknown = unknown_ro if str(lang_hint).lower().startswith("ro") else unknown_en

        url = f"{self.host.rstrip('/')}/api/generate"

        if mode == "precise":
            safety = (
                "IMPORTANT: Answer only with verified facts. "
                f"If uncertain or outdated, reply exactly with: '{unknown}' "
                "Keep answers concise."
            )
            temperature = 0.0; top_p = 0.9; top_k = 40
        else:
            safety = "Be helpful and friendly."
            temperature = self.temperature; top_p = 0.95; top_k = 50

        sys = (self.system or "").strip()
        prompt = f"{sys}\n{safety}\nUser ({lang_hint}): {user_text}\nAssistant:"

        start = time.perf_counter()
        with requests.post(url, json={
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repeat_penalty": 1.1,
                "num_predict": self.max_tokens
            }
        }, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            first_token_s = None
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    tok = (data.get("response") or "")
                    if tok:
                        if first_token_s is None:
                            first_token_s = time.perf_counter()
                            self.log.info(f"LLM first token in {first_token_s - start:.2f}s")
                        yield tok
                except Exception:
                    continue
            if first_token_s is not None:
                self.log.info(f"LLM stream completed in {time.perf_counter() - start:.2f}s")

    def _openai_chat(self, user_text: str, lang_hint: str) -> str:
        try:
            msg = [
                {"role": "system", "content": self.system or "You are concise."},
                {"role": "user", "content": f"[lang={lang_hint}] {user_text}"},
            ]
            r = self._openai.chat.completions.create(
                model=self.cfg.get("model", "gpt-4o-mini"),
                messages=msg,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            self.log.error(f"OpenAI error: {e}")
            return self._rule_based(user_text, lang_hint)
