# Voice agent

`create_voice_agent(llm, lang="bci")` builds a LangGraph graph for one spoken turn:

```
listen (ASR) -> [phrase_snap | low_confidence] -> translate_in -> agent (French LLM)
             -> translate_out -> speak (TTS) -> out
```

```python
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ivoire import create_voice_agent
from langchain_ivoire.audio import from_audio_block, to_audio_block

agent = create_voice_agent(llm, lang="bci", phrasebook="phrasebook.json",
                           checkpointer=InMemorySaver())
out = agent.invoke({"audio": to_audio_block(open("question.wav", "rb").read())},
                   {"configurable": {"thread_id": "family-call"}})
out["transcript"], out["user_fra"], out["reply_fra"], out["reply"]
open("reply.wav", "wb").write(from_audio_block(out["reply_audio"]))
```

## Audio as content blocks

Audio in the state is a standard LangChain audio content block:

```python
{"type": "audio", "base64": "...", "mime_type": "audio/wav", "extras": {"sample_rate": 16000}}
```

The state is therefore JSON and checkpoints cleanly. `langchain_ivoire.audio` converts in both
directions (`to_audio_block`, `from_audio_block`). The `audio` input also accepts base64 text or raw
bytes. `IvoireSpeakTool` returns the same kind of block as its tool-message artifact.

## Phrase snap and low confidence

- **Phrase snap:** when a phrasebook is given and the utterance clearly matches a phrase (by
  sound-alike skeletons), the phrase's checked text and translation are used and machine translation
  is skipped.
- **Low confidence:** turns below `low_confidence_threshold` (default 0.80) are marked
  `low_confidence=True` and still translated; your app can ask the speaker to repeat.
- **LLM failure:** if the LLM fails, the turn still returns the transcript and its French.
