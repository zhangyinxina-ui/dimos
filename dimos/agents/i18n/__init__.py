# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Internationalization (i18n) utilities for DimOS agent system prompts."""

from typing import Literal

from dimos.agents.i18n.zh import (
    DRONE_SYSTEM_PROMPT_ZH,
    G1_SYSTEM_PROMPT_ZH,
    GO2_SYSTEM_PROMPT_ZH,
)

SupportedLanguage = Literal["en", "zh"]

_ZH_PROMPTS: dict[str, str] = {
    "go2": GO2_SYSTEM_PROMPT_ZH,
    "g1": G1_SYSTEM_PROMPT_ZH,
    "drone": DRONE_SYSTEM_PROMPT_ZH,
}


def get_system_prompt(robot: str, language: SupportedLanguage, en_prompt: str) -> str:
    """Return the system prompt for the given robot and language.

    Falls back to ``en_prompt`` for unsupported languages or unknown robots.
    """
    if language == "zh":
        return _ZH_PROMPTS.get(robot, en_prompt)
    return en_prompt


__all__ = [
    "SupportedLanguage",
    "get_system_prompt",
    "GO2_SYSTEM_PROMPT_ZH",
    "G1_SYSTEM_PROMPT_ZH",
    "DRONE_SYSTEM_PROMPT_ZH",
]
