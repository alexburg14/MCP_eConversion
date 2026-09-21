"""A scripted stand-in for the OpenAI streaming client.

Chunks are ``SimpleNamespace`` trees mirroring the SDK's attribute paths
(``chunk.choices[0].delta.content``, ``delta.tool_calls[i].function.arguments``,
``chunk.usage``) so ``agent.run_turn`` exercises exactly the code it runs in
production, without the network.
"""
from __future__ import annotations

from types import SimpleNamespace


def _delta(**fields):
    base = {"content": None, "tool_calls": None}
    base.update(fields)
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(**base), finish_reason=None)],
                           usage=None)


def text_chunk(text: str):
    return _delta(content=text)


def reasoning_chunk(text: str):
    return _delta(reasoning_content=text)


def tool_chunk(index: int | None, id: str | None = None, name: str | None = None,
               arguments: str = "", omit_index: bool = False):
    tc = {"id": id, "function": SimpleNamespace(name=name, arguments=arguments)}
    if not omit_index:
        tc["index"] = index
    return _delta(tool_calls=[SimpleNamespace(**tc)])


def usage_chunk(prompt: int, completion: int, total: int):
    return SimpleNamespace(choices=[], usage=SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=total))


class FakeStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False

    def __iter__(self):
        yield from self._chunks

    def close(self):
        self.closed = True


class FakeOpenAI:
    """``scripts`` is one chunk list per model round, consumed in order.

    An entry that is an ``Exception`` instance is raised by ``create`` instead.
    ``calls`` records the kwargs of every ``create`` call; ``streams`` the
    returned streams (to assert ``close()``).
    """

    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.calls: list[dict] = []
        self.streams: list[FakeStream] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._scripts:
            raise AssertionError("FakeOpenAI: more rounds requested than scripted")
        script = self._scripts.pop(0)
        if isinstance(script, Exception):
            raise script
        stream = FakeStream(script)
        self.streams.append(stream)
        return stream
