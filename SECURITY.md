# Security policy

## Reporting a vulnerability

Please report security problems **privately**, through GitHub's private vulnerability reporting:
open the repository's **Security** tab and choose **Report a vulnerability**. Do not open a public
issue, discussion or pull request for a security problem.

Please include what you found, how to reproduce it, which versions are affected, and the impact
you expect. We aim to acknowledge reports within 5 working days and to agree a disclosure date with
you; we will credit you in the advisory unless you prefer otherwise.

## Supported versions

| version | supported |
|---|---|
| 0.1.x | yes |
| < 0.1 (pre-releases) | no |

## Scope

In scope: the `langchain-ivoire` package in this repository, including the
`ivoire-mcp` server, and the release workflow. Examples of what we want to hear about:

- an LLM-controlled tool argument that reads local files, reaches internal network addresses or
  otherwise escapes the documented limits (the transcribe tools refuse file paths and private URLs);
- API keys leaking into logs, exceptions, traces, benchmark results or packages;
- supply-chain problems in the build or publishing workflows.

Out of scope: the hosted Abyshire API itself (report those to Abyshire directly through the same
private channel and we will forward them), and translation or recognition quality problems (please
use normal issues for those).

## Hardening already in place

- No telemetry. API keys come only from the constructor or `IVOIRE_API_KEY`, and are never logged or
  included in exceptions.
- The transcribe tool and MCP server accept base64 audio or public `http(s)` URLs only, refuse
  loopback/private/link-local addresses (opt-in override), re-check redirects and cap downloads at
  25 MiB.
- Unit tests run with the network blocked (pytest-socket).
- Releases use PyPI trusted publishing with PEP 740 attestations, from a protected environment,
  and ship a CycloneDX SBOM. Workflow actions are pinned by commit SHA and updated by Dependabot.
