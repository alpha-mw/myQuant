"""Structural source/config parsers for the cutover dependency graph."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import PurePosixPath
import shlex
from typing import Any, Final

try:  # Python 3.13 is the package floor; the fallback helps isolated lint tooling.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from .errors import (
    DYNAMIC_IMPORT_ALLOWLIST_MISMATCH,
    DYNAMIC_IMPORT_NOT_ALLOWLISTED,
    UNPARSEABLE_PYTHON,
    UNPARSEABLE_SHELL,
    UNPARSEABLE_TOML,
    UNPARSEABLE_YAML,
    UnifiedCutoverError,
)
from .rules import DynamicImportAllowance

_SHELL_COMMAND_KEYS: Final = frozenset({"cmd", "command", "run", "script", "shell"})
_PYTHON_EXECUTABLES: Final = frozenset(
    {"python", "python3", "python3.13", "python.exe", "pypy", "pypy3"}
)


@dataclass(frozen=True, order=True)
class ImportEdge:
    module: str
    line: int
    kind: str


@dataclass(frozen=True, order=True)
class ConfigEdge:
    target_kind: str
    target: str
    line: int
    source_kind: str


@dataclass(frozen=True)
class ParsedPython:
    imports: tuple[ImportEdge, ...]
    used_allowlist_keys: frozenset[tuple[str, int, str]]


def ast_node_sha256(node: ast.AST) -> str:
    """Return the exact location-independent identity used by the allowlist."""

    dumped = ast.dump(node, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def module_name_for_path(relative_path: str) -> tuple[str, bool] | None:
    path = PurePosixPath(relative_path)
    if path.suffix != ".py":
        return None
    parts = list(path.with_suffix("").parts)
    is_package = bool(parts and parts[-1] == "__init__")
    if is_package:
        parts.pop()
    if not parts or any(not part.isidentifier() for part in parts):
        return None
    return ".".join(parts), is_package


def _resolve_from_module(
    current_module: str,
    *,
    is_package: bool,
    level: int,
    imported_module: str | None,
) -> str:
    if level == 0:
        return imported_module or ""
    package = current_module.split(".") if is_package else current_module.split(".")[:-1]
    ascend = level - 1
    if ascend >= len(package):
        return ""
    prefix = package[: len(package) - ascend]
    if imported_module:
        prefix.extend(imported_module.split("."))
    return ".".join(prefix)


def _literal_dynamic_module(call: ast.Call) -> str | None:
    if not call.args:
        return None
    value = call.args[0]
    if isinstance(value, ast.Constant) and type(value.value) is str and value.value:
        module = value.value
        if module.startswith("."):
            if len(call.args) < 2:
                return None
            package = call.args[1]
            if not isinstance(package, ast.Constant) or type(package.value) is not str:
                return None
            leading = len(module) - len(module.lstrip("."))
            package_parts = package.value.split(".")
            if leading > len(package_parts):
                return None
            base = package_parts[: len(package_parts) - leading + 1]
            tail = module.lstrip(".")
            return ".".join(base + ([tail] if tail else []))
        return module
    return None


def parse_python_imports(
    raw: bytes,
    *,
    relative_path: str,
    module_name: str,
    is_package: bool,
    allowlist: Mapping[tuple[str, int, str], DynamicImportAllowance],
) -> ParsedPython:
    try:
        source = raw.decode("utf-8")
        tree = ast.parse(source, filename=relative_path, mode="exec")
    except (UnicodeDecodeError, SyntaxError, ValueError) as exc:
        raise UnifiedCutoverError(
            UNPARSEABLE_PYTHON, f"cannot parse Python source {relative_path}"
        ) from exc

    importlib_names = {"importlib"}
    import_module_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    importlib_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    import_module_names.add(alias.asname or alias.name)

    imports: set[ImportEdge] = set()
    used: set[tuple[str, int, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(ImportEdge(alias.name, node.lineno, "AST_IMPORT"))
            continue
        if isinstance(node, ast.ImportFrom):
            base = _resolve_from_module(
                module_name,
                is_package=is_package,
                level=node.level,
                imported_module=node.module,
            )
            if base:
                imports.add(ImportEdge(base, node.lineno, "AST_FROM"))
                for alias in node.names:
                    if alias.name != "*":
                        imports.add(
                            ImportEdge(f"{base}.{alias.name}", node.lineno, "AST_FROM_MEMBER")
                        )
            elif node.level:
                imports.add(ImportEdge("<BROKEN_RELATIVE_IMPORT>", node.lineno, "AST_FROM"))
            continue
        if not isinstance(node, ast.Call):
            continue

        is_dynamic_import = False
        if isinstance(node.func, ast.Name) and node.func.id in (
            {"__import__"} | import_module_names
        ):
            is_dynamic_import = True
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in importlib_names
        ):
            is_dynamic_import = True
        if not is_dynamic_import:
            continue

        literal = _literal_dynamic_module(node)
        if literal is not None:
            imports.add(ImportEdge(literal, node.lineno, "AST_LITERAL_DYNAMIC"))
            continue
        digest = ast_node_sha256(node)
        key = (relative_path, node.lineno, digest)
        allowance = allowlist.get(key)
        if allowance is None:
            raise UnifiedCutoverError(
                DYNAMIC_IMPORT_NOT_ALLOWLISTED,
                f"dynamic import {relative_path}:{node.lineno} AST {digest} is not allowlisted",
            )
        if allowance.key != key or not allowance.modules:
            raise UnifiedCutoverError(
                DYNAMIC_IMPORT_ALLOWLIST_MISMATCH,
                f"dynamic import allowance mismatch at {relative_path}:{node.lineno}",
            )
        used.add(key)
        for allowed_module in allowance.modules:
            imports.add(ImportEdge(allowed_module, node.lineno, "AST_ALLOWLIST_DYNAMIC"))

    return ParsedPython(tuple(sorted(imports)), frozenset(used))


def parse_toml_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise UnifiedCutoverError(UNPARSEABLE_TOML, f"cannot parse TOML {label}") from exc
    if type(value) is not dict:
        raise UnifiedCutoverError(UNPARSEABLE_TOML, f"TOML {label} has no mapping root")
    return value


def _strip_yaml_comment(value: str) -> str:
    single = False
    double = False
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\" and double:
            escaped = True
            continue
        if char == "'" and not double:
            single = not single
            continue
        if char == '"' and not single:
            double = not double
            continue
        if char == "#" and not single and not double and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    if single or double:
        raise UnifiedCutoverError(UNPARSEABLE_YAML, "unterminated YAML quote")
    return value.rstrip()


def _yaml_colon(value: str) -> int | None:
    single = False
    double = False
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\" and double:
            escaped = True
        elif char == "'" and not double:
            single = not single
        elif char == '"' and not single:
            double = not double
        elif char == ":" and not single and not double:
            if index + 1 == len(value) or value[index + 1].isspace():
                return index
    return None


def _yaml_scalar(value: str, *, line: int) -> Any:
    text = value.strip()
    if not text:
        return None
    if text.startswith(("&", "*", "!")) or "<<:" in text:
        raise UnifiedCutoverError(
            UNPARSEABLE_YAML, f"YAML aliases, tags, and merges are forbidden at line {line}"
        )
    if text.startswith('"'):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise UnifiedCutoverError(UNPARSEABLE_YAML, f"invalid quoted scalar at {line}") from exc
        if type(parsed) is not str:
            raise UnifiedCutoverError(UNPARSEABLE_YAML, f"invalid quoted scalar at {line}")
        return parsed
    if text.startswith("'"):
        if len(text) < 2 or not text.endswith("'"):
            raise UnifiedCutoverError(UNPARSEABLE_YAML, f"invalid quoted scalar at {line}")
        return text[1:-1].replace("''", "'")
    lowered = text.lower()
    if lowered in {"null", "~"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    if text.startswith(("[", "{")):
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise UnifiedCutoverError(
                UNPARSEABLE_YAML,
                f"flow YAML must be strict JSON at line {line}",
            ) from exc
    return text


@dataclass(frozen=True)
class YamlScalar:
    key: str | None
    value: Any
    line: int


def parse_strict_yaml_scalars(raw: bytes, *, label: str) -> tuple[YamlScalar, ...]:
    """Validate a strict safe YAML subset and return structural scalar leaves.

    The subset intentionally rejects executable tags, aliases, merges, tabs and
    ambiguous flow syntax.  It supports nested mappings/lists and literal/folded
    command blocks, which covers repo workflow and automation files.
    """

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UnifiedCutoverError(UNPARSEABLE_YAML, f"{label} is not UTF-8") from exc
    if "\t" in text:
        raise UnifiedCutoverError(UNPARSEABLE_YAML, f"{label} contains a tab")
    physical = text.splitlines()
    results: list[YamlScalar] = []
    context_at_indent: dict[int, tuple[str, set[str]]] = {0: ("mapping", set())}
    previous_indent = 0
    previous_open = True
    index = 0
    while index < len(physical):
        original = physical[index]
        if not original.strip() or original.lstrip().startswith("#"):
            index += 1
            continue
        indent = len(original) - len(original.lstrip(" "))
        if indent % 2:
            raise UnifiedCutoverError(
                UNPARSEABLE_YAML, f"{label} uses non-two-space indentation at {index + 1}"
            )
        content = _strip_yaml_comment(original[indent:])
        if not content:
            index += 1
            continue
        if content in {"---", "..."}:
            if indent:
                raise UnifiedCutoverError(UNPARSEABLE_YAML, f"invalid document marker at {index+1}")
            index += 1
            continue
        if indent > previous_indent and not previous_open:
            raise UnifiedCutoverError(
                UNPARSEABLE_YAML, f"unexpected indentation in {label} at line {index + 1}"
            )
        for level in tuple(context_at_indent):
            if level > indent:
                context_at_indent.pop(level, None)

        is_list = content == "-" or content.startswith("- ")
        body = content[1:].lstrip() if is_list else content
        colon = _yaml_colon(body)
        key: str | None = None
        value_text = body
        opens = False
        if colon is not None:
            raw_key = body[:colon].strip()
            if not raw_key:
                raise UnifiedCutoverError(UNPARSEABLE_YAML, f"empty YAML key at {index+1}")
            key_value = _yaml_scalar(raw_key, line=index + 1)
            if type(key_value) is not str or not key_value:
                raise UnifiedCutoverError(UNPARSEABLE_YAML, f"invalid YAML key at {index+1}")
            key = key_value
            value_text = body[colon + 1 :].strip()
            mapping_level = indent + (2 if is_list else 0)
            kind, seen = context_at_indent.setdefault(mapping_level, ("mapping", set()))
            if kind != "mapping":
                raise UnifiedCutoverError(UNPARSEABLE_YAML, f"mixed YAML node at {index+1}")
            if key in seen:
                raise UnifiedCutoverError(
                    UNPARSEABLE_YAML, f"duplicate YAML key {key!r} at line {index + 1}"
                )
            seen.add(key)
            context_at_indent[mapping_level] = (kind, seen)
        elif not is_list:
            raise UnifiedCutoverError(UNPARSEABLE_YAML, f"expected YAML mapping at {index+1}")

        if value_text in {"|", "|-", "|+", ">", ">-", ">+"}:
            block_indent: int | None = None
            block: list[str] = []
            cursor = index + 1
            while cursor < len(physical):
                candidate = physical[cursor]
                if not candidate.strip():
                    block.append("")
                    cursor += 1
                    continue
                candidate_indent = len(candidate) - len(candidate.lstrip(" "))
                if candidate_indent <= indent:
                    break
                if block_indent is None:
                    block_indent = candidate_indent
                if candidate_indent < block_indent:
                    raise UnifiedCutoverError(
                        UNPARSEABLE_YAML, f"invalid block indentation at line {cursor+1}"
                    )
                block.append(candidate[block_indent:])
                cursor += 1
            separator = " " if value_text.startswith(">") else "\n"
            results.append(YamlScalar(key, separator.join(block), index + 1))
            index = cursor
            previous_indent = indent
            previous_open = False
            continue

        scalar = _yaml_scalar(value_text, line=index + 1)
        if value_text:
            results.append(YamlScalar(key, scalar, index + 1))
        else:
            opens = True
            child_level = indent + 2
            context_at_indent.pop(child_level, None)
        previous_indent = indent
        previous_open = opens or (is_list and (not body or colon is not None))
        index += 1
    return tuple(results)


def shell_tokens(command: str, *, label: str) -> tuple[str, ...]:
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()<>")
        lexer.whitespace_split = True
        lexer.commenters = "#"
        return tuple(lexer)
    except (ValueError, TypeError) as exc:
        raise UnifiedCutoverError(UNPARSEABLE_SHELL, f"cannot tokenize shell {label}") from exc


def _segments(tokens: Sequence[str]) -> list[list[str]]:
    separators = {";", "&&", "||", "|", "&", "(", ")"}
    result: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in separators:
            if current:
                result.append(current)
                current = []
        else:
            current.append(token)
    if current:
        result.append(current)
    return result


def shell_config_edges(
    command: str,
    *,
    label: str,
    line: int,
    console_scripts: Mapping[str, str],
) -> tuple[ConfigEdge, ...]:
    edges: set[ConfigEdge] = set()
    for segment in _segments(shell_tokens(command, label=label)):
        while segment and "=" in segment[0] and not segment[0].startswith(("/", "./")):
            name, _, _value = segment[0].partition("=")
            if not name.isidentifier():
                break
            segment = segment[1:]
        if not segment:
            continue
        executable = PurePosixPath(segment[0]).name
        if executable in console_scripts:
            module = console_scripts[executable].split(":", 1)[0]
            edges.add(ConfigEdge("module", module, line, "CONSOLE_SCRIPT"))
        if executable in _PYTHON_EXECUTABLES:
            if "-m" in segment:
                position = segment.index("-m")
                if position + 1 >= len(segment) or segment[position + 1].startswith(("$", "{")):
                    raise UnifiedCutoverError(
                        UNPARSEABLE_SHELL, f"dynamic python -m target in {label}"
                    )
                edges.add(ConfigEdge("module", segment[position + 1], line, "SHELL_PYTHON_M"))
            else:
                script = next(
                    (
                        token
                        for token in segment[1:]
                        if token.endswith(".py") and not token.startswith("-")
                    ),
                    None,
                )
                if script is not None:
                    normalized = script[2:] if script.startswith("./") else script
                    edges.add(ConfigEdge("path", normalized, line, "SHELL_PYTHON_FILE"))
        elif executable.endswith(".py"):
            normalized = segment[0][2:] if segment[0].startswith("./") else segment[0]
            edges.add(ConfigEdge("path", normalized, line, "SHELL_EXEC_FILE"))
    return tuple(sorted(edges))


def _walk_toml_strings(value: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], str]]:
    result: list[tuple[tuple[str, ...], str]] = []
    if type(value) is dict:
        for key in sorted(value):
            result.extend(_walk_toml_strings(value[key], path + (str(key),)))
    elif type(value) is list:
        for index, item in enumerate(value):
            result.extend(_walk_toml_strings(item, path + (str(index),)))
    elif type(value) is str:
        result.append((path, value))
    return result


def parse_toml_config_edges(raw: bytes, *, relative_path: str) -> tuple[ConfigEdge, ...]:
    document = parse_toml_bytes(raw, label=relative_path)
    project = document.get("project")
    scripts = project.get("scripts", {}) if type(project) is dict else {}
    if type(scripts) is not dict:
        raise UnifiedCutoverError(UNPARSEABLE_TOML, "project.scripts must be a mapping")
    console_scripts: dict[str, str] = {}
    edges: set[ConfigEdge] = set()
    for name, target in sorted(scripts.items()):
        if type(name) is not str or type(target) is not str or not target:
            raise UnifiedCutoverError(UNPARSEABLE_TOML, "project.scripts entry is invalid")
        module = target.split(":", 1)[0]
        console_scripts[name] = target
        edges.add(ConfigEdge("module", module, 1, "TOML_ENTRYPOINT"))
    for key_path, value in _walk_toml_strings(document):
        if key_path and key_path[-1].lower() in _SHELL_COMMAND_KEYS:
            edges.update(
                shell_config_edges(
                    value,
                    label=f"{relative_path}:{'.'.join(key_path)}",
                    line=1,
                    console_scripts=console_scripts,
                )
            )
    return tuple(sorted(edges))


def parse_yaml_config_edges(
    raw: bytes,
    *,
    relative_path: str,
    console_scripts: Mapping[str, str],
) -> tuple[ConfigEdge, ...]:
    edges: set[ConfigEdge] = set()
    for scalar in parse_strict_yaml_scalars(raw, label=relative_path):
        if (
            scalar.key is not None
            and scalar.key.lower() in _SHELL_COMMAND_KEYS
            and type(scalar.value) is str
        ):
            edges.update(
                shell_config_edges(
                    scalar.value,
                    label=f"{relative_path}:{scalar.line}",
                    line=scalar.line,
                    console_scripts=console_scripts,
                )
            )
    return tuple(sorted(edges))


def parse_shell_config_edges(
    raw: bytes,
    *,
    relative_path: str,
    console_scripts: Mapping[str, str],
) -> tuple[ConfigEdge, ...]:
    try:
        command = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UnifiedCutoverError(UNPARSEABLE_SHELL, f"{relative_path} is not UTF-8") from exc
    return shell_config_edges(
        command,
        label=relative_path,
        line=1,
        console_scripts=console_scripts,
    )


__all__ = [
    "ConfigEdge",
    "ImportEdge",
    "ParsedPython",
    "YamlScalar",
    "ast_node_sha256",
    "module_name_for_path",
    "parse_python_imports",
    "parse_shell_config_edges",
    "parse_strict_yaml_scalars",
    "parse_toml_bytes",
    "parse_toml_config_edges",
    "parse_yaml_config_edges",
    "shell_config_edges",
    "shell_tokens",
]
