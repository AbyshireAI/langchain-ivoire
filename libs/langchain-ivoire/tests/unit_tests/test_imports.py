import langchain_ivoire

EXPECTED = {
    "IvoireTranscriber",
    "IvoireTranslator",
    "IvoireSpeaker",
    "with_language",
    "IvoireLanguageChat",
    "create_voice_agent",
    "IvoireAudioLoader",
    "IvoireTranslateTool",
    "IvoireTranscribeTool",
    "IvoireSpeakTool",
    "Phrasebook",
    "RemoteBackend",
    "FakeBackend",
}


def test_public_api() -> None:
    assert set(langchain_ivoire.__all__) >= EXPECTED
    for name in langchain_ivoire.__all__:
        assert hasattr(langchain_ivoire, name), name
    assert langchain_ivoire.__version__ == "0.1.0a3"
