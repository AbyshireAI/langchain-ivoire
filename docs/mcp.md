# MCP server

`ivoire-mcp` exposes the Baoulé tools to any [Model Context Protocol](https://modelcontextprotocol.io)
client. It is built on the official MCP Python SDK (2.x).

```bash
pip install "langchain-ivoire[mcp]"
ivoire-mcp                                            # stdio
ivoire-mcp --transport streamable-http --port 8000    # http://127.0.0.1:8000/mcp
ivoire-mcp --backend fake                             # offline demo
```

| tool | input | output |
|---|---|---|
| `ivoire_transcribe` | `audio_base64`, or a public http(s) `audio_url` | text, `confidence`, `low_confidence`, `audio_s` |
| `ivoire_translate` | `text`, `source`, `target` (`bci`/`fra`/`eng`), `style` | the translation |
| `ivoire_speak` | Baoulé `text` | MCP `audio` content (WAV) |
| `ivoire_languages` | (none) | languages, speech and translation support |

## Configuration

| variable | meaning |
|---|---|
| `IVOIRE_API_KEY` | API key for the hosted Abyshire API |
| `IVOIRE_BASE_URL` | another server running the same API (default: the hosted API) |

## MCP client configuration

Most MCP clients (desktop assistants, IDEs) take a JSON entry like this one. Add it to your MCP
client's config file, replacing the placeholder. Never commit a real key.

```json
{
  "mcpServers": {
    "baoule": {
      "command": "uvx",
      "args": ["--from", "langchain-ivoire[mcp]", "ivoire-mcp"],
      "env": {
        "IVOIRE_API_KEY": "<your-ivoire-api-key>",
        "IVOIRE_BASE_URL": "https://api.abyshire.ci"
      }
    }
  }
}
```

## Security

`ivoire_transcribe` never reads local files. URLs must be http(s), and loopback, private and
link-local addresses are refused unless you start the server with `--allow-private-urls`. Downloads
are capped at 25 MiB. The HTTP transport binds to `127.0.0.1` by default.
