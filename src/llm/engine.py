# src/llm/engine.py
from __future__ import annotations
from typing import Dict, Optional, List
from datetime import datetime
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

        self._system_base = self.cfg.get("system_prompt", "")
        self.host = self.cfg.get("host", "http://localhost:11434")
        self.model = self.cfg.get("model", "qwen2.5:3b")
        self.max_tokens = int(self.cfg.get("max_tokens", 120))
        self.temperature = float(self.cfg.get("temperature", 0.4))

        self.default_mode = (self.cfg.get("default_mode") or "precise").lower()
        self.strict_facts = bool(self.cfg.get("strict_facts", True))

        # Warm-up config
        self.warmup_enabled = bool(self.cfg.get("warmup_enabled", True))
        self.warmup_text = (self.cfg.get("warmup_text") or "Hello").strip()
        self.warmup_lang = (self.cfg.get("warmup_lang") or "en").lower()
        self._warmed_up = False

        # History config
        self.history_enabled = bool(self.cfg.get("history_enabled", True))
        self.max_history_turns = int(self.cfg.get("max_history_turns", 5))

        # Fallback responses
        self.fallback = self.cfg.get("fallback") or {}

        # Web Search config (Groq Compound)
        self.websearch_enabled = bool(self.cfg.get("websearch_enabled", False))
        self.websearch_model = self.cfg.get("websearch_model", "compound-beta")
        self.websearch_max_tokens = int(self.cfg.get("websearch_max_tokens", 300))



        # Groq client
        self._groq = None
        if self.provider == "groq":
            try:
                from groq import Groq
                self._groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
            except Exception as e:
                self.log.error(f"Groq client indisponibil: {e}. Revin pe 'rule'.")
                self.provider = "rule"

        self.log.info(f"LLM provider activ: {self.provider}")
        if self.websearch_enabled:
            self.log.info(f"🌐 Web search ENABLED (model: {self.websearch_model})")
        else:
            self.log.info(f"🌐 Web search DISABLED")
        
        # Warm-up la boot
        self._ensure_warm()

    @property
    def system(self) -> str:
        """Returnează system prompt cu data curentă injectată."""
        date_str = datetime.now().strftime("%A, %B %d, %Y")  # e.g., "Monday, December 23, 2024"
        date_prefix = f"Today is {date_str}.\n\n"
        return date_prefix + (self._system_base or "")

    def _ensure_warm(self):
        """Încarcă modelul în RAM prin request dummy."""
        if not self.warmup_enabled or self._warmed_up:
            return
        if self.provider != "ollama":
            self._warmed_up = True
            return
        try:
            self.log.info(f"🔥 LLM warm-up start (model={self.model})")
            start = time.perf_counter()
            # Request simplu, fără a folosi răspunsul
            url = f"{self.host.rstrip('/')}/api/generate"
            resp = requests.post(url, json={
                "model": self.model,
                "prompt": self.warmup_text,
                "stream": False,
                "options": {"num_predict": 5}  # răspuns scurt
            }, timeout=60)
            resp.raise_for_status()
            elapsed = time.perf_counter() - start
            self._warmed_up = True
            self.log.info(f"✅ LLM warm-up gata ({elapsed:.2f}s)")
        except Exception as e:
            self.log.warning(f"LLM warm-up eșuat: {e}")

    def _get_fallback(self, key: str, lang: str) -> str:
        """Returnează mesajul de fallback pentru cheie și limbă."""
        suffix = "_ro" if str(lang).lower().startswith("ro") else "_en"
        return self.fallback.get(f"{key}{suffix}", "")

    def generate(self, user_text: str, lang_hint: str = "en", mode: Optional[str] = None) -> str:
        mode = (mode or self.default_mode).lower()
        with observe_hist(llm_latency):
            if self.provider == "rule":
                return self._rule_based(user_text, lang_hint)
            if self.provider == "ollama":
                return self._ollama_http(user_text, lang_hint, mode=mode)

            return "No LLM provider configured."

    def generate_stream(self, user_text: str, lang_hint: str = "en", mode: Optional[str] = None, history: Optional[List[Dict]] = None):
        """Generează răspuns cu streaming. history = [{"role": "user"/"assistant", "content": ...}, ...]"""
        mode = (mode or self.default_mode).lower()
        if self.provider == "groq":
            gen = self._groq_stream(user_text, lang_hint, mode, history)
            return wrap_stream_for_first_token(gen, llm_first_token_latency)
        if self.provider == "ollama":
            gen = self._ollama_stream(user_text, lang_hint, mode, history)
            return wrap_stream_for_first_token(gen, llm_first_token_latency)
        def _one():
            yield self.generate(user_text, lang_hint, mode)
        return _one()

    def _rule_based(self, user_text: str, lang_hint: str) -> str:
        if not (user_text or "").strip():
            return "Nu am auzit întrebarea. Poți repeta?"
        return f"{'Am înțeles' if lang_hint.startswith('ro') else 'I heard'}: \"{user_text}\"."

    def _ollama_http(self, user_text: str, lang_hint: str, mode: str = "precise") -> str:
        # Fallback-uri din config
        unknown = self._get_fallback("unknown", lang_hint) or "I don't know."
        error_msg = self._get_fallback("error", lang_hint) or "Technical error."
        empty_msg = self._get_fallback("empty", lang_hint) or unknown

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
                return empty_msg
            return text or "…"
        except requests.exceptions.Timeout:
            self.log.error("Ollama timeout")
            return self._get_fallback("timeout", lang_hint) or error_msg
        except Exception as e:
            self.log.error(f"Ollama HTTP error: {e}")
            return error_msg

    def _ollama_stream(self, user_text: str, lang_hint: str, mode: str = "precise", history: Optional[List[Dict]] = None):
        # Fallback-uri din config
        unknown = self._get_fallback("unknown", lang_hint) or "I don't know."

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
        
        # Formatează history în prompt
        history_text = ""
        if self.history_enabled and history:
            # Limitează la max_history_turns
            limited = history[-(self.max_history_turns * 2):]
            for msg in limited:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    history_text += f"User: {content}\n"
                elif role == "assistant":
                    history_text += f"Assistant: {content}\n"
        
        # Construiește prompt-ul final
        if history_text:
            prompt = f"{sys}\n{safety}\n\n{history_text}User: {user_text}\nAssistant:"
        else:
            prompt = f"{sys}\n{safety}\nUser ({lang_hint}): {user_text}\nAssistant:"

        start = time.perf_counter()
        try:
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
        except requests.exceptions.Timeout:
            self.log.error("Ollama stream timeout")
            yield self._get_fallback("timeout", lang_hint) or "Taking too long. Try again."
        except Exception as e:
            self.log.error(f"Ollama stream error: {e}")
            yield self._get_fallback("error", lang_hint) or "Technical error. Try again."



    def _groq_stream(self, user_text: str, lang_hint: str, mode: str = "precise", history: Optional[List[Dict]] = None):
        """Streaming cu API-ul Groq. Suportă web search prin Groq Compound."""
        unknown = self._get_fallback("unknown", lang_hint) or "I don't know."
        
        sys_content = (self.system or "You are a helpful assistant.").strip()
        if mode == "precise":
            sys_content += f"\nIMPORTANT: Answer only with verified facts. If uncertain, reply with: '{unknown}'"
        
        messages = [{"role": "system", "content": sys_content}]
        
        # Adaugă history
        if self.history_enabled and history:
            limited = history[-(self.max_history_turns * 2):]
            for msg in limited:
                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        
        messages.append({"role": "user", "content": user_text})
        
        # compound-beta decide singur când să facă web search
        
        start = time.perf_counter()
        try:
            stream = self._groq.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0 if mode == "precise" else self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            
            first_token_s = None
            for chunk in stream:
                delta = chunk.choices[0].delta
                tok = delta.content or ""
                if tok:
                    if first_token_s is None:
                        first_token_s = time.perf_counter()
                        self.log.info(f"LLM first token in {first_token_s - start:.2f}s")
                    yield tok
            
            if first_token_s is not None:
                self.log.info(f"LLM stream completed in {time.perf_counter() - start:.2f}s")
        except Exception as e:
            self.log.error(f"Groq stream error: {e}")
            yield self._get_fallback("error", lang_hint) or "Technical error. Try again."

    def _needs_websearch(self, text: str) -> bool:
        """Detectează dacă întrebarea necesită informații actuale de pe web."""
        text_lower = text.lower()
        
        # Cuvinte cheie care indică nevoie de info actuală
        current_info_keywords = [
            # English - time-sensitive
            "news", "today", "latest", "current", "recent", "now",
            "weather", "price", "stock", "score", "result",
            "who won", "what happened", "breaking",
            # English - factual questions that benefit from search
            "who is the", "who is", "president", "prime minister",
            "ceo of", "founder of", "how much does", "how much is",
            # English - elections & politics
            "election", "elected", "candidate", "vote", "voting",
            "parliament", "congress", "senator", "governor",
            # English - sports
            "match", "game", "championship", "tournament", "league",
            "world cup", "olympics", "fifa", "nba", "nfl",
            # English - entertainment
            "movie", "film", "actor", "actress", "oscar", "grammy",
            "album", "song", "concert", "tour", "netflix", "spotify",
            # English - tech & business
            "iphone", "android", "google", "apple", "microsoft", "tesla",
            "chatgpt", "cryptocurrency", "bitcoin", "gpt-4", "gpt-5",
            # Romanian - time-sensitive
            "știri", "stiri", "azi", "acum", "recent", "ultima", "moment",
            "vreme", "preț", "pret", "scor", "rezultat", "valoare", "curs",
            "euro", "dolar", "criptomonede",
            "cine a câștigat", "cine a castigat", "ce s-a întâmplat",
            "cine este", "președinte", "presedinte", "prim-ministru",
            # Romanian - elections & politics
            "alegeri", "ales", "candidat", "vot", "votat", "votare",
            "parlament", "senator", "deputat", "partid", "guvern",
            "tur", "turul doi", "turul întâi", "campanie",
            # Romanian - sports
            "meci", "joc", "campionat", "liga", "fotbal", "nationala",
            "steaua", "dinamo", "cfr", "fcsb", "simona halep",
            # Romanian - entertainment
            "film", "actor", "actriță", "actrita", "serial", "netflix",
            "muzică", "muzica", "concert", "album", "cântăreț", "cantaret",
            # Romanian - tech & business
            "telefon", "aplicație", "aplicatie", "emag", "olx"
        ]
        
        for keyword in current_info_keywords:
            if keyword in text_lower:
                self.log.info(f"🔍 Web search triggered by keyword: '{keyword}'")
                return True
        
        return False

