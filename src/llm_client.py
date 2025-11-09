"""Local-only LLM client that interacts with an Ollama server."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Coroutine

try:  # pragma: no cover - optional dependency handling
    import aiohttp
except ImportError:  # pragma: no cover - optional dependency handling
    aiohttp = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


def format_size(size_bytes: int) -> str:
    """Format a raw byte size using human-friendly units."""

    if size_bytes <= 0:
        return "0 B"
    gb = size_bytes / (1024**3)
    if gb >= 1:
        return f"{gb:.1f} GB"
    mb = size_bytes / (1024**2)
    if mb >= 1:
        return f"{mb:.0f} MB"
    kb = size_bytes / 1024
    return f"{kb:.0f} KB"


@dataclass
class _OllamaConfig:
    """Configuration for the local Ollama backend."""

    base_url: str
    model: str
    temperature: float
    max_tokens: int
    request_timeout: int
    num_ctx: int


class LLMClient:
    """Client responsible for querying a local Ollama instance."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        request_timeout: int = 60,
        num_ctx: int = 2048,
        auto_select_model: bool = False,
    ) -> None:
        resolved_base = base_url or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"
        requested_model = model or os.environ.get("OLLAMA_MODEL") or "auto"

        self.auto_select_model = auto_select_model or requested_model in {None, "", "auto"}
        self._ensure_aiohttp()
        self.cfg = _OllamaConfig(
            base_url=resolved_base.rstrip("/"),
            model=requested_model,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
            num_ctx=num_ctx,
        )

        if self.auto_select_model:
            selected = self._select_lightest_model()
            if selected:
                logger.info("Using lightest available Ollama model: %s", selected)
                self.cfg.model = selected
            else:
                logger.warning(
                    "Auto model selection requested but no local models were found."
                )
        else:
            logger.info("Using configured Ollama model: %s", self.cfg.model)

        self.conversation_history: List[Dict[str, str]] = []
        self.stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "avg_response_time": 0.0,
        }
        self._available_models: List[Dict[str, Any]] = []
        self._history_limit = 8

    # ------------------------------------------------------------------
    # Prompt construction helpers
    # ------------------------------------------------------------------
    def create_system_prompt(self, primitive_descriptions: Dict[str, str]) -> str:
        prompt = (
            "Tu es un expert en optimisation de systèmes de production reconfigurables. "
            "Analyse l'état courant, propose des actions concrètes et justifie tes choix.\n\n"
        )
        if primitive_descriptions:
            prompt += "Primitives disponibles :\n"
            for name, description in primitive_descriptions.items():
                prompt += f"- {name} : {description}\n"

        prompt += (
            "\nObjectifs :\n"
            "1. Minimiser le makespan.\n"
            "2. Réduire les retards et pénalités.\n"
            "3. Limiter les reconfigurations inutiles.\n"
            "4. Préserver l'efficacité énergétique.\n\n"
            "Réponds en expliquant ta stratégie et propose des actions au format :\n"
            "ACTION: primitive(param=val, ...)"
        )
        return prompt

    def format_state_for_llm(self, state: Dict[str, Any]) -> str:
        formatted = ["## État Courant du Système\n"]
        formatted.append(f"**Temps actuel :** {state['current_time']:.2f}\n")
        formatted.append(f"**Jobs en attente :** {state['pending_jobs']}\n")
        formatted.append(f"**Jobs terminés :** {state['completed_jobs']}\n\n")

        formatted.append("### Statut des Machines\n")
        machines = state.get("machines", [])
        for machine in machines[:5]:
            formatted.append(
                "- Machine {machine_id} : État={state}, Config={config}, Utilisation={util:.1%}, "
                "Disponible à t={next_available:.2f}\n".format(
                    machine_id=machine.get("machine_id"),
                    state=machine.get("state"),
                    config=machine.get("current_config"),
                    util=machine.get("utilization", 0.0),
                    next_available=machine.get("next_available", 0.0),
                )
            )
        if len(machines) > 5:
            formatted.append(f"... et {len(machines) - 5} machines supplémentaires\n")

        formatted.append("\n### Files d'attente (Top 10)\n")
        for job in state.get("job_queue", [])[:10]:
            formatted.append(
                "- Job {job_id} : Priorité={priority}, Ops restantes={remaining}, "
                "Échéance={due:.2f}, Marge={slack:.2f}\n".format(
                    job_id=job.get("job_id"),
                    priority=job.get("priority"),
                    remaining=job.get("remaining_operations"),
                    due=job.get("due_date", 0.0),
                    slack=job.get("slack", 0.0),
                )
            )
        queue_len = len(state.get("job_queue", []))
        if queue_len > 10:
            formatted.append(f"... et {queue_len - 10} jobs supplémentaires\n")

        metrics = state.get("metrics", {})
        formatted.append("\n### Indicateurs de performance\n")
        formatted.append(f"- Makespan : {metrics.get('makespan', 0.0):.2f}\n")
        formatted.append(f"- Retard moyen : {metrics.get('avg_tardiness', 0.0):.2f}\n")
        formatted.append(f"- Utilisation : {metrics.get('utilization', 0.0):.1%}\n")
        formatted.append(f"- Énergie : {metrics.get('energy_consumption', 0.0):.2f}\n")
        formatted.append(f"- Reconfigurations : {metrics.get('reconfigurations', 0)}\n")

        return "".join(formatted)

    def format_similar_episodes(self, episodes: List[Dict[str, Any]]) -> str:
        if not episodes:
            return "Aucun épisode similaire trouvé.\n"

        lines = ["## Épisodes Similaires\n"]
        for ep in episodes[:3]:
            lines.append(
                "### Épisode {eid} (Distance : {dist:.3f})\n".format(
                    eid=ep.get("episode_id"),
                    dist=ep.get("distance", 0.0),
                )
            )
            lines.append(f"**Récompense :** {ep.get('reward', 0.0):.2f}\n")
            action = ep.get("action", {})
            lines.append(f"**Action réalisée :** {action.get('type', 'inconnue')}\n")
            metrics = ep.get("state", {}).get("metrics", {})
            if metrics:
                lines.append(f"- Makespan : {metrics.get('makespan', 0.0):.2f}\n")
                lines.append(f"- Retard : {metrics.get('avg_tardiness', 0.0):.2f}\n")
            lines.append("\n")

        return "".join(lines)

    # ------------------------------------------------------------------
    # Interaction methods
    # ------------------------------------------------------------------
    def generate_response(
        self,
        user_message: str,
        state: Dict[str, Any],
        similar_episodes: Optional[List[Dict[str, Any]]] = None,
        primitive_descriptions: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        return self._generate_response_internal(
            user_message,
            state,
            similar_episodes=similar_episodes,
            primitive_descriptions=primitive_descriptions,
            allow_retry=True,
        )

    def _generate_response_internal(
        self,
        user_message: str,
        state: Dict[str, Any],
        similar_episodes: Optional[List[Dict[str, Any]]],
        primitive_descriptions: Optional[Dict[str, str]],
        allow_retry: bool,
    ) -> Dict[str, Any]:
        start_time = time.time()

        formatted_state = self.format_state_for_llm(state)
        full_message = f"{user_message}\n\n{formatted_state}"
        if similar_episodes:
            full_message += "\n" + self.format_similar_episodes(similar_episodes)

        system_prompt = (
            self.create_system_prompt(primitive_descriptions)
            if primitive_descriptions
            else None
        )

        try:
            response_text, token_usage = self._run_async(
                self._async_chat_request(system_prompt, full_message)
            )
        except Exception as exc:  # pragma: no cover - depends on external runtime
            error_message = str(exc)
            lowered = error_message.lower()
            is_memory_issue = "requires more system memory" in lowered or "unable to load full model" in lowered

            logger.error("Erreur lors de l'appel à Ollama : %s", error_message)

            if is_memory_issue:
                self._log_memory_recommendations()
                if allow_retry and self.auto_select_model:
                    alternative = self._select_lightest_model()
                    if alternative and alternative != self.cfg.model:
                        logger.info(
                            "Switching to lighter Ollama model %s after memory error",
                            alternative,
                        )
                        self.cfg.model = alternative
                        return self._generate_response_internal(
                            user_message,
                            state,
                            similar_episodes,
                            primitive_descriptions,
                            allow_retry=False,
                        )

            return {
                "success": False,
                "error": error_message,
                "response": None,
                "fatal": is_memory_issue,
            }

        elapsed = time.time() - start_time
        self._update_statistics(elapsed, token_usage)

        self.conversation_history.extend(
            [
                {"role": "user", "content": full_message},
                {"role": "assistant", "content": response_text},
            ]
        )
        if len(self.conversation_history) > self._history_limit * 2:
            self.conversation_history = self.conversation_history[-self._history_limit * 2 :]

        logger.info(
            "Réponse générée via Ollama (%s) en %.2fs",
            self.cfg.model,
            elapsed,
        )
        return {
            "success": True,
            "response": response_text,
            "tokens": token_usage,
            "time": elapsed,
        }

    async def _async_chat_request(
        self, system_prompt: Optional[str], full_message: str
    ) -> Tuple[str, int]:
        self._ensure_aiohttp()
        messages = self._build_messages(system_prompt, full_message)

        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.max_tokens,
                "num_ctx": self.cfg.num_ctx,
            },
        }

        timeout = aiohttp.ClientTimeout(total=self.cfg.request_timeout)
        url = f"{self.cfg.base_url}/api/chat"

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    text = await response.text()
                    if response.status != 200:
                        raise RuntimeError(
                            f"HTTP {response.status} error from {url}: {text}"
                        )
                    data = json.loads(text)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f"Timeout while connecting to {url}") from exc
        except aiohttp.ClientError as exc:
            raise RuntimeError(f"Connection error to {url}: {exc}") from exc

        message = data.get("message") or {}
        content = message.get("content") or data.get("response", "")
        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        completion_tokens = int(data.get("eval_count") or 0)
        return content, prompt_tokens + completion_tokens

    def _build_messages(
        self, system_prompt: Optional[str], user_message: str
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if self.conversation_history:
            messages.extend(self.conversation_history[-self._history_limit :])

        messages.append({"role": "user", "content": user_message})
        return messages

    def list_available_models(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        if force_refresh or not self._available_models:
            try:
                self._available_models = self._run_async(self._fetch_available_models())
            except Exception as exc:  # pragma: no cover - depends on local runtime
                logger.warning("Unable to query Ollama models: %s", exc)
                self._available_models = []
        return list(self._available_models)

    async def _fetch_available_models(self) -> List[Dict[str, Any]]:
        self._ensure_aiohttp()
        timeout = aiohttp.ClientTimeout(total=self.cfg.request_timeout)
        url = f"{self.cfg.base_url}/api/tags"

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise RuntimeError(
                            f"HTTP {response.status} error from {url}: {text}"
                        )
                    data = await response.json()
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f"Timeout while connecting to {url}") from exc
        except aiohttp.ClientError as exc:
            raise RuntimeError(f"Connection error to {url}: {exc}") from exc

        return data.get("models", [])

    def _select_lightest_model(self) -> Optional[str]:
        models = self.list_available_models(force_refresh=True)
        local_models = [m for m in models if ":cloud" not in m.get("name", "")]
        if not local_models:
            return None

        local_models.sort(key=lambda m: m.get("size", float("inf")))
        return local_models[0].get("name")

    def _log_memory_recommendations(self) -> None:
        models = self.list_available_models(force_refresh=False)
        if not models:
            logger.error(
                "Aucun modèle Ollama local détecté. Téléchargez un modèle léger avec "
                "`ollama pull llama3.2:1b`."
            )
            return

        local_models = [m for m in models if ":cloud" not in m.get("name", "")]
        if not local_models:
            logger.error(
                "Seuls des modèles cloud sont disponibles. Téléchargez un modèle local léger, "
                "par exemple `ollama pull llama3.2:1b`."
            )
            return

        logger.error(
            "Le modèle '%s' semble trop volumineux pour la mémoire disponible. Modèles locaux "
            "détectés :",
            self.cfg.model,
        )
        for model in local_models[:5]:
            name = model.get("name", "inconnu")
            size = format_size(int(model.get("size", 0)))
            logger.error(" - %s (%s)", name, size)

        logger.error(
            "Utilisez `ollama pull llama3.2:1b` ou un autre modèle compact puis mettez à jour "
            "votre configuration."
        )

    @staticmethod
    def _run_async(coro: Coroutine[Any, Any, Any]) -> Any:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def _update_statistics(self, elapsed: float, tokens: int) -> None:
        previous = self.stats["total_requests"]
        total_requests = previous + 1
        self.stats["total_requests"] = total_requests
        self.stats["total_tokens"] += tokens
        self.stats["avg_response_time"] = (
            (self.stats["avg_response_time"] * previous) + elapsed
        ) / total_requests

    @staticmethod
    def _ensure_aiohttp() -> None:
        if aiohttp is None:
            raise RuntimeError(
                "Le client LLM nécessite le paquet `aiohttp`. Installez-le avec `pip install aiohttp`."
            )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def parse_action_from_response(self, response_text: str) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        for line in response_text.splitlines():
            stripped = line.strip()
            if not (stripped.startswith("ACTION:") or stripped.startswith("PRIMITIVE:")):
                continue
            action_str = stripped.split(":", 1)[1].strip()
            try:
                if "(" in action_str and ")" in action_str:
                    prim_name = action_str[: action_str.index("(")].strip()
                    params_str = action_str[action_str.index("(") + 1 : action_str.rindex(")")]
                    params: Dict[str, Any] = {}
                    if params_str:
                        for part in params_str.split(","):
                            if "=" not in part:
                                continue
                            key, value = part.split("=", 1)
                            key = key.strip()
                            value = value.strip()
                            params[key] = self._coerce_value(value)
                    actions.append({"primitive": prim_name, "parameters": params})
            except Exception as exc:  # pragma: no cover - parsing guard
                logger.warning("Échec de l'analyse de l'action '%s' : %s", action_str, exc)
        return actions

    @staticmethod
    def _coerce_value(raw_value: str) -> Any:
        lowered = raw_value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        try:
            if "." in raw_value:
                return float(raw_value)
            return int(raw_value)
        except ValueError:
            pass
        try:
            return json.loads(raw_value)
        except json.JSONDecodeError:
            return raw_value.strip("'\"")

    def reset_conversation(self) -> None:
        self.conversation_history = []

    def get_statistics(self) -> Dict[str, Any]:
        return dict(self.stats)

