from __future__ import annotations

from quant_investor.llm_gateway import LLMClient


def test_parse_json_content_repairs_trailing_commas_in_code_fence():
    payload = """```json
{
  "branch_name": "kline",
  "conviction": "buy",
  "symbol_views": {
    "000001.SZ": "趋势偏强",
  },
}
```"""

    parsed = LLMClient._parse_json_content(payload)

    assert parsed["branch_name"] == "kline"
    assert parsed["symbol_views"]["000001.SZ"] == "趋势偏强"


def test_parse_json_content_accepts_python_like_dict_literals():
    payload = """{
        'branch_name': 'kline',
        'conviction': 'buy',
        'confidence': 0.7,
    }"""

    parsed = LLMClient._parse_json_content(payload)

    assert parsed["branch_name"] == "kline"
    assert parsed["confidence"] == 0.7
