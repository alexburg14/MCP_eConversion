"""uvicorn entry point:  uvicorn main:app --app-dir src --port 8000

``AppState.from_server()`` imports ``server``, which loads every cache and
starts the sentence-transformer preload thread. That import MUST happen in the
process main thread before any request is served (torch has to be imported
from the main thread first, see server._preload_semantic_model), which is why
the app is built at module import here and never lazily inside a handler.
"""
from web import create_app

app = create_app()
