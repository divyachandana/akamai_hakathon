"""Local MCP server exposing tools to the Qwen3 tool agent (run automatically by Qwen-Agent)."""
import ast
import datetime
import operator
import random
import re
import urllib.request
from zoneinfo import ZoneInfo

try:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server
except ModuleNotFoundError:  # mcp 2.x renamed FastMCP -> MCPServer
    from mcp.server.mcpserver import MCPServer as _Server

mcp = _Server("hackathon-tools")

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Only basic arithmetic is allowed")


@mcp.tool()
def get_current_time(timezone: str = "America/Los_Angeles") -> str:
    """Get the current date and time in an IANA timezone, e.g. 'Asia/Tokyo', 'Europe/Paris'."""
    try:
        now = datetime.datetime.now(ZoneInfo(timezone))
    except Exception:
        return f"Unknown timezone: {timezone}"
    return now.strftime(f"%A %Y-%m-%d %H:%M:%S ({timezone})")


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression exactly, e.g. '23*47 + 2**10'."""
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def roll_dice(sides: int = 20, count: int = 1) -> str:
    """Roll `count` dice with `sides` sides each and return the rolls and total."""
    count = max(1, min(count, 20))
    rolls = [random.randint(1, max(2, sides)) for _ in range(count)]
    return f"Rolls: {rolls}, total: {sum(rolls)}"


@mcp.tool()
def fetch_url(url: str, max_chars: int = 2000) -> str:
    """Fetch a web page and return its visible text (truncated)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req, timeout=10).read().decode("utf-8", "ignore")
    except Exception as e:
        return f"Error fetching {url}: {e}"
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()
    return text[:max_chars]


if __name__ == "__main__":
    mcp.run()
