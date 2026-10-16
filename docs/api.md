# API reference

## Middleware and chat

::: langchain_ivoire.middleware.IvoireLanguageMiddleware

::: langchain_ivoire.chat.with_language

::: langchain_ivoire.chat.IvoireLanguageChat
    options:
      members: false

## Components

::: langchain_ivoire.transcriber.IvoireTranscriber
    options:
      members: false

::: langchain_ivoire.translator.IvoireTranslator
    options:
      members: [translate_texts, atranslate_texts]

::: langchain_ivoire.speaker.IvoireSpeaker
    options:
      members: false

## Tools

::: langchain_ivoire.tools
    options:
      members: [IvoireTranslateTool, IvoireTranscribeTool, IvoireSpeakTool, get_tools]

## Voice agent

::: langchain_ivoire.graph.create_voice_agent

## Audio content blocks

::: langchain_ivoire.audio

## Backends

::: langchain_ivoire.backends.base.IvoireBackend

::: langchain_ivoire.backends.remote.RemoteBackend
    options:
      members: [translate, languages, models, health]

::: langchain_ivoire.backends.openrouter.OpenRouterBackend
    options:
      members: [translate, transcribe, synthesize, embed]

::: langchain_ivoire.backends.fake.FakeBackend

## MCP server

::: langchain_ivoire.mcp_server

## Results and errors

::: langchain_ivoire.types

::: langchain_ivoire.exceptions
