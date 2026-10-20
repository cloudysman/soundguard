"""Constants for the AI suite.

These live outside conftest.py on purpose. Every suite has its own conftest.py, so a test module that
does `from conftest import ...` resolves to whichever conftest landed on sys.path first — which makes
the suites impossible to run in one pytest invocation. A uniquely named module cannot collide.
"""
CLASSES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music",
]
