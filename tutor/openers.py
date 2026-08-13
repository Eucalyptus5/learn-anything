import numpy as np

from tutor.tts import KokoroSynthesizer

OPENER_PHRASES: dict[str, str] = {
    "thinking": "looking that up",
    "file_hit": "found it in",
    "many_files": "several places touch this",
    "empty": "no hits on that",
}


def synthesize_openers(synth: KokoroSynthesizer) -> dict[str, np.ndarray]:
    return {key: synth.synthesize(phrase) for key, phrase in OPENER_PHRASES.items()}
