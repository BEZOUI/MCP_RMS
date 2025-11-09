"""LLM client supporting DeepSeek, Groq, and Ollama backends."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

try:  # Optional dependency – only required when the Groq backend is used
    from groq import Groq  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully at runtime
    Groq = None  # type: ignore


logger = logging.getLogger(__name__)


def _post_json(
    url: str,
    payload: Dict[str, Any],
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: int,
) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(url, data=data, headers=headers or {}, method="POST")

    try:
        with urllib_request.urlopen(req, timeout=timeout) as response:
            charset = "utf-8"
            if hasattr(response.headers, "get_content_charset"):
                charset = response.headers.get_content_charset() or "utf-8"
            body = response.read().decode(charset, errors="replace")
            return json.loads(body)
    except urllib_error.HTTPError as exc:  # pragma: no cover - network errors
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code} error from {url}: {detail}") from exc
    except urllib_error.URLError as exc:  # pragma: no cover - network errors
        raise RuntimeError(f"Connection error to {url}: {exc}") from exc


@dataclass
class _ClientConfig:
    """Configuration resolved for the current backend."""

    provider: str
    model: str
    temperature: float
    max_tokens: int
    request_timeout: int
    groq_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: Optional[str] = None


class LLMClient:
    """Client capable of interacting with multiple LLM providers."""

    DEFAULT_MODELS = {
        "deepseek": "deepseek-reasoner",
        "groq": "deepseek-r1-distill-llama-70b",
        "ollama": "deepseek-r1:latest",
    }

    def __init__(
        self,
        provider: str = "deepseek",
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 8192,
        request_timeout: int = 60,
        groq_api_key: Optional[str] = None,
        deepseek_api_key: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: Optional[str] = None,
    ) -> None:
        resolved_provider = (provider or "deepseek").lower()

        if resolved_provider == "auto":
            resolved_provider = self._auto_select_provider(groq_api_key, deepseek_api_key)

        resolved_model = model or self.DEFAULT_MODELS.get(resolved_provider)
        if resolved_model is None:
            raise ValueError(f"No default model defined for provider '{resolved_provider}'")

        self.cfg = _ClientConfig(
            provider=resolved_provider,
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
            groq_api_key=groq_api_key or os.environ.get("GROQ_API_KEY"),
            deepseek_api_key=deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY"),
            ollama_base_url=(
                ollama_base_url
                or os.environ.get("OLLAMA_BASE_URL")
                or "http://localhost:11434"
            ),
            ollama_model=ollama_model
            or os.environ.get("OLLAMA_MODEL")
            or model
            or self.DEFAULT_MODELS["ollama"],
        )

        self._validate_backend()
        self._initialize_backend_client()

        self.conversation_history: List[Dict[str, str]] = []
        self.stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "avg_response_time": 0.0,
        }

    def _auto_select_provider(self, groq_key: Optional[str], deepseek_key: Optional[str]) -> str:
        if deepseek_key or os.environ.get("DEEPSEEK_API_KEY"):
            return "deepseek"
        if groq_key or os.environ.get("GROQ_API_KEY"):
            return "groq"
        return "ollama"

    def _validate_backend(self) -> None:
        if self.cfg.provider == "groq":
            if self.cfg.groq_api_key is None:
                raise ValueError("GROQ_API_KEY is required for Groq provider")
            if Groq is None:
                raise ImportError(
                    "groq package is not installed. Add it to requirements or install at runtime."
                )
        elif self.cfg.provider == "deepseek":
            if self.cfg.deepseek_api_key is None:
                raise ValueError("DEEPSEEK_API_KEY is required for DeepSeek provider")
        elif self.cfg.provider == "ollama":
            # No additional validation necessary; HTTP errors will surface during calls.
            pass
        else:
            raise ValueError(f"Unsupported provider '{self.cfg.provider}'")

    def _initialize_backend_client(self) -> None:
        if self.cfg.provider == "groq":
            self._client = Groq(api_key=self.cfg.groq_api_key)  # type: ignore[arg-type]
        else:
            self._client = None

    # ------------------------------------------------------------------
    # Prompt construction helpers
    # ------------------------------------------------------------------
    def create_system_prompt(self, primitive_descriptions: Dict[str, str]) -> str:
        prompt = (
            "You are an expert in manufacturing optimization and scheduling for "
            "Reconfigurable Manufacturing Systems (RMS). Your role is to optimize "
            "production schedules by composing primitive operations intelligently.\n\n"
        )
        if primitive_descriptions:
            prompt += "Available Primitives:\n"
            for name, description in primitive_descriptions.items():
                prompt += f"- {name}: {description}\n"

        prompt += (
            "\nObjectives:\n"
            "1. Minimize makespan.\n"
            "2. Minimize tardiness penalties.\n"
            "3. Minimize energy consumption.\n"
            "4. Minimize unnecessary reconfigurations.\n\n"
            "Follow a deliberate reasoning process, explain your rationale, and "
            "produce executable primitive actions in the format:\n"
            "ACTION: primitive_name(param=value, ...)"
        )
        return prompt

    def format_state_for_llm(self, state: Dict[str, Any]) -> str:
        formatted = ["## Current System State\n"]
        formatted.append(f"**Current Time:** {state['current_time']:.2f}\n")
        formatted.append(f"**Pending Jobs:** {state['pending_jobs']}\n")
        formatted.append(f"**Completed Jobs:** {state['completed_jobs']}\n\n")

        formatted.append("### Machines Status\n")
        machines = state.get("machines", [])
        for machine in machines[:5]:
            formatted.append(
                "- Machine {machine_id}: State={state}, Config={config}, Util={util:.1%}, "
                "Next Available={next_available:.2f}\n".format(
                    machine_id=machine.get("machine_id"),
                    state=machine.get("state"),
                    config=machine.get("current_config"),
                    util=machine.get("utilization", 0.0),
                    next_available=machine.get("next_available", 0.0),
                )
            )
        if len(machines) > 5:
            formatted.append(f"... and {len(machines) - 5} more machines\n")

        formatted.append("\n### Job Queue (Top 10)\n")
        for job in state.get("job_queue", [])[:10]:
            formatted.append(
                "- Job {job_id}: Priority={priority}, Remaining Ops={remaining}, "
                "Due={due:.2f}, Slack={slack:.2f}\n".format(
                    job_id=job.get("job_id"),
                    priority=job.get("priority"),
                    remaining=job.get("remaining_operations"),
                    due=job.get("due_date", 0.0),
                    slack=job.get("slack", 0.0),
                )
            )
        queue_len = len(state.get("job_queue", []))
        if queue_len > 10:
            formatted.append(f"... and {queue_len - 10} more jobs\n")

        metrics = state.get("metrics", {})
        formatted.append("\n### Performance Metrics\n")
        formatted.append(f"- Makespan: {metrics.get('makespan', 0.0):.2f}\n")
        formatted.append(f"- Avg Tardiness: {metrics.get('avg_tardiness', 0.0):.2f}\n")
        formatted.append(f"- Utilization: {metrics.get('utilization', 0.0):.1%}\n")
        formatted.append(f"- Energy: {metrics.get('energy_consumption', 0.0):.2f}\n")
        formatted.append(f"- Reconfigurations: {metrics.get('reconfigurations', 0)}\n")

        return "".join(formatted)

    def format_similar_episodes(self, episodes: List[Dict[str, Any]]) -> str:
        if not episodes:
            return "No similar past episodes found.\n"

        lines = ["## Similar Past Episodes\n"]
        for ep in episodes[:3]:
            lines.append(
                "### Episode {eid} (Distance: {dist:.3f})\n".format(
                    eid=ep.get("episode_id"),
                    dist=ep.get("distance", 0.0),
                )
            )
            lines.append(f"**Reward:** {ep.get('reward', 0.0):.2f}\n")
            action = ep.get("action", {})
            lines.append(f"**Action Taken:** {action.get('type', 'unknown')}\n")
            metrics = ep.get("state", {}).get("metrics", {})
            if metrics:
                lines.append(f"- Makespan: {metrics.get('makespan', 0.0):.2f}\n")
                lines.append(f"- Tardiness: {metrics.get('avg_tardiness', 0.0):.2f}\n")
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
        except Exception as exc:  # pragma: no cover - network/IO heavy
            logger.error("LLM API error: %s", exc)
            return {"success": False, "error": str(exc), "response": None}

        elapsed = time.time() - start_time
        self._update_statistics(elapsed, token_usage)

        self.conversation_history.extend(
            [
                {"role": "user", "content": full_message},
                {"role": "assistant", "content": response_text},
            ]
        )

        logger.info("LLM response generated via %s in %.2fs", self.cfg.provider, elapsed)
        return {
            "success": True,
            "response": response_text,
            "tokens": token_usage,
            "time": elapsed,
        }

    def _dispatch_request(
        self, system_prompt: Optional[str], full_message: str
    ) -> (str, int):
        if self.cfg.provider == "groq":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": full_message})

            response = self._client.chat.completions.create(  # type: ignore[union-attr]
                model=self.cfg.model,
                messages=messages,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            )
            content = response.choices[0].message.content  # type: ignore[index]
            usage = getattr(response, "usage", None)
            total_tokens = 0
            if usage is not None:
                total_tokens = getattr(usage, "total_tokens", 0) or (
                    getattr(usage, "prompt_tokens", 0)
                    + getattr(usage, "completion_tokens", 0)
                )
            return content, total_tokens

        if self.cfg.provider == "deepseek":
            headers = {
                "Authorization": f"Bearer {self.cfg.deepseek_api_key}",
                "Content-Type": "application/json",
            }
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": full_message})

            payload = {
                "model": self.cfg.model,
                "messages": messages,
                "temperature": self.cfg.temperature,
                "max_tokens": self.cfg.max_tokens,
            }

            data = _post_json(
                "https://api.deepseek.com/v1/chat/completions",
                payload,
                headers=headers,
                timeout=self.cfg.request_timeout,
            )
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            total_tokens = int(usage.get("total_tokens") or 0)
            if not total_tokens:
                total_tokens = int(usage.get("prompt_tokens", 0)) + int(
                    usage.get("completion_tokens", 0)
                )
            return content, total_tokens

        if self.cfg.provider == "ollama":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": full_message})

            payload = {
                "model": self.cfg.ollama_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.cfg.temperature,
                },
            }

            data = _post_json(
                f"{self.cfg.ollama_base_url.rstrip('/')}/api/chat",
                payload,
                headers={"Content-Type": "application/json"},
                timeout=self.cfg.request_timeout,
            )
            message = data.get("message", {})
            content = message.get("content", "")
            total_tokens = int(data.get("eval_count") or 0)
            return content, total_tokens

        raise ValueError(f"Unsupported provider '{self.cfg.provider}'")

    def _update_statistics(self, elapsed: float, tokens: int) -> None:
        prev_requests = self.stats["total_requests"]
        total_requests = prev_requests + 1
        self.stats["total_requests"] = total_requests
        self.stats["total_tokens"] += tokens
        self.stats["avg_response_time"] = (
            (self.stats["avg_response_time"] * prev_requests) + elapsed
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
            except Exception as exc:  # pragma: no cover - defensive parsing
                logger.warning("Failed to parse action '%s': %s", action_str, exc)
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

