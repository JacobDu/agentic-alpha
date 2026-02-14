from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

PROXY_URL = "http://127.0.0.1:7890"


@dataclass
class NetworkResult:
    used_proxy: bool
    attempts: list[dict[str, Any]]


def _proxy_env(enabled: bool) -> dict[str, str]:
    if not enabled:
        return {}
    return {
        "http": PROXY_URL,
        "https": PROXY_URL,
        "all": PROXY_URL,
    }


def _request_with_retry(
    method: str,
    url: str,
    *,
    timeout: int = 120,
    stream: bool = False,
) -> tuple[requests.Response, NetworkResult]:
    attempts: list[dict[str, Any]] = []
    last_error: Exception | None = None
    for use_proxy in (False, True):
        try:
            response = requests.request(
                method,
                url,
                timeout=timeout,
                stream=stream,
                proxies=_proxy_env(use_proxy) or None,
                headers={"User-Agent": "agentic-alpha/1.0"},
            )
            response.raise_for_status()
            attempts.append({"via_proxy": use_proxy, "success": True})
            return response, NetworkResult(used_proxy=use_proxy, attempts=attempts)
        except Exception as exc:  # noqa: BLE001
            attempts.append({"via_proxy": use_proxy, "success": False, "error": str(exc)})
            last_error = exc
    assert last_error is not None
    raise RuntimeError(f"Network request failed after retries: {url}\n{attempts}") from last_error


def fetch_json(url: str, *, timeout: int = 120) -> tuple[Any, NetworkResult]:
    response, meta = _request_with_retry("GET", url, timeout=timeout)
    return response.json(), meta


def fetch_text(url: str, *, timeout: int = 120) -> tuple[str, NetworkResult]:
    response, meta = _request_with_retry("GET", url, timeout=timeout)
    return response.text, meta


def download_file(url: str, destination: Path, *, timeout: int = 300) -> NetworkResult:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    response, meta = _request_with_retry("GET", url, timeout=timeout, stream=True)
    with temp_path.open("wb") as file_obj:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file_obj.write(chunk)
    temp_path.replace(destination)
    return meta
