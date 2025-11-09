"""Local-only LLM client that interacts with an Ollama server."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request


logger = logging.getLogger(__name__)


def _request_json(
    url: str,
    *,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout: int,
) -> Dict[str, Any]:
    """Send an HTTP request and decode the JSON response."""

    data: Optional[bytes] = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib_request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib_request.urlopen(req, timeout=timeout) as response:
            charset = "utf-8"
            if hasattr(response.headers, "get_content_charset"):
                charset = response.headers.get_content_charset() or "utf-8"
            body = response.read().decode(charset, errors="replace")
            if not body:
                return {}
            return json.loads(body)
    except urllib_error.HTTPError as exc:  # pragma: no cover - relies on local server
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code} error from {url}: {detail}") from exc
    except urllib_error.URLError as exc:  # pragma: no cover - relies on local server
        raise RuntimeError(f"Connection error to {url}: {exc}") from exc


def _format_size(size_bytes: int) -> str:
    """Return a human readable representation of a model size."""

    if size_bytes <= 0:
        return "unknown"
    gigabytes = size_bytes / (1024 ** 3)
    if gigabytes >= 1:
        return f"{gigabytes:.1f} GB"
    megabytes = size_bytes / (1024 ** 2)
    return f"{megabytes:.0f} MB"


@dataclass
class _ModelInfo:
    name: str
    size: int


@dataclass
class _OllamaConfig:
    """Configuration for the local Ollama backend."""

    base_url: str
    model: str
    temperature: float
    max_tokens: int
    request_timeout: int
    context_window: int
    history_limit: int


class LLMClient:
    """Client responsible for querying a local Ollama instance."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        request_timeout: int = 60,
        context_window: int = 2048,
        history_limit: int = 6,
    ) -> None:
        resolved_base = base_url or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"
        base = resolved_base.rstrip("/")

        requested_model = model or os.environ.get("OLLAMA_MODEL")
        available_models = self._fetch_available_models(base, request_timeout)
        resolved_model = self._resolve_model_name(requested_model, available_models)

        self.cfg = _OllamaConfig(
            base_url=base,
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
            context_window=context_window,
            history_limit=max(history_limit, 0),
        )

        self._available_models = available_models
        self.conversation_history: List[Dict[str, str]] = []
        self.stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "avg_response_time": 0.0,
        }

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
            response_text, token_usage = self._dispatch_request(system_prompt, full_message)
        except Exception as exc:  # pragma: no cover - depends on external runtime
            error_message = str(exc)
            lowered = error_message.lower()
            is_memory_issue = "requires more system memory" in lowered or "unable to load full model" in lowered
            logger.error("Erreur lors de l'appel à Ollama : %s", error_message)
            return {
                "success": False,
                "error": error_message,
                "response": None,
                "fatal": is_memory_issue,
            }

        elapsed = time.time() - start_time
        self._update_statistics(elapsed, token_usage)

        self._append_history({"role": "user", "content": full_message})
        self._append_history({"role": "assistant", "content": response_text})

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

    def _dispatch_request(
        self, system_prompt: Optional[str], full_message: str
    ) -> (str, int):
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if self.conversation_history:
            history_slice = self.conversation_history[-self.cfg.history_limit :]
            messages.extend(history_slice)

        messages.append({"role": "user", "content": full_message})

        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.max_tokens,
                "num_ctx": self.cfg.context_window,
            },
        }

        data = _request_json(
            f"{self.cfg.base_url}/api/chat",
            method="POST",
            payload=payload,
            timeout=self.cfg.request_timeout,
        )

        message = data.get("message", {})
        content = message.get("content", "")
        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        completion_tokens = int(data.get("eval_count") or 0)
        return content, prompt_tokens + completion_tokens

    def _update_statistics(self, elapsed: float, tokens: int) -> None:
        previous = self.stats["total_requests"]
        total_requests = previous + 1
        self.stats["total_requests"] = total_requests
        self.stats["total_tokens"] += tokens
        self.stats["avg_response_time"] = (
            (self.stats["avg_response_time"] * previous) + elapsed
        ) / total_requests

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

    # ------------------------------------------------------------------
    # Model discovery helpers
    # ------------------------------------------------------------------
    def available_models(self) -> List[Dict[str, Any]]:
        """Return the cached list of models discovered at initialisation."""

        return [dict(name=m.name, size=m.size) for m in self._available_models]

    def _fetch_available_models(
        self, base_url: str, timeout: int
    ) -> List[_ModelInfo]:
        url = f"{base_url}/api/tags"
        try:
            response = _request_json(url, timeout=timeout)
        except Exception as exc:  # pragma: no cover - relies on external runtime
            logger.warning("Unable to query Ollama models from %s: %s", url, exc)
            return []

        models: List[_ModelInfo] = []
        for entry in response.get("models", []):
            name = entry.get("name")
            size = int(entry.get("size") or 0)
            if not name:
                continue
            models.append(_ModelInfo(name=name, size=size))
        if models:
            summary = ", ".join(f"{m.name} ({_format_size(m.size)})" for m in models)
            logger.info("Modèles Ollama détectés : %s", summary)
        return models

    def _resolve_model_name(
        self,
        requested_model: Optional[str],
        available_models: List[_ModelInfo],
    ) -> str:
        if requested_model:
            logger.info("Utilisation du modèle Ollama demandé : %s", requested_model)
            return requested_model

        local_models = [m for m in available_models if ":cloud" not in m.name]
        if local_models:
            lightest = min(local_models, key=lambda item: item.size or float("inf"))
            logger.info(
                "Aucun modèle spécifié - sélection automatique du modèle le plus léger : %s (%s)",
                lightest.name,
                _format_size(lightest.size),
            )
            return lightest.name

        logger.warning(
            "Aucun modèle Ollama local détecté - utilisation du modèle par défaut 'llama3.2:1b'."
        )
        return "llama3.2:1b"

    def _append_history(self, message: Dict[str, str]) -> None:
        if self.cfg.history_limit <= 0:
            return
        self.conversation_history.append(message)
        if len(self.conversation_history) > self.cfg.history_limit:
            self.conversation_history = self.conversation_history[-self.cfg.history_limit :]

