# # """Voice input utilities for EchoAnalytics.

# # This module provides:
# #  - capture_voice(): low-level microphone → text
# #  - handle_voice_query(): orchestration that captures voice and executes a data query

# # handle_voice_query returns a triple so the UI can update context immediately:
# #     (result, updated_context, raw_query)
# # If capture fails, result and updated_context are None and raw_query contains the error token.
# # """

# # import speech_recognition as sr
# # from typing import Any, Tuple

# # try:  # Prevent circular import in certain test orders
# #     from .query_parser import run_query  # type: ignore
# # except Exception:  # pragma: no cover - fallback for circular import in tests
# #     run_query = None  # type: ignore


# # ERROR_NO_SPEECH = "No speech detected"
# # ERROR_SERVICE = "Speech service error"

# # def capture_voice() -> str:
# #     """Capture speech from microphone and return recognized text or an error token.

# #     Returns one of:
# #       - recognized phrase (str)
# #       - ERROR_NO_SPEECH
# #       - ERROR_SERVICE
# #     """
# #     recognizer = sr.Recognizer()
# #     mic = sr.Microphone()
# #     with mic as source:
# #         print("Listening...")
# #         recognizer.adjust_for_ambient_noise(source, duration=0.4)
# #         audio = recognizer.listen(source)
# #     try:
# #         text = recognizer.recognize_google(audio)
# #         print(f"Recognized: {text}")
# #         return text.strip()
# #     except sr.UnknownValueError:
# #         print("Speech Recognition could not understand audio")
# #         return ERROR_NO_SPEECH
# #     except sr.RequestError:
# #         print("Speech Recognition service error")
# #         return ERROR_SERVICE


# # def handle_voice_query(df, *, capture_fn=capture_voice) -> Tuple[Any, dict | None, str | None]:
# #     """Capture a voice query and execute it against df.

# #     Returns:
# #         result: The query result (scalar/DataFrame/list/etc) or None on failure.
# #         updated_context: The context dict returned by run_query or None if not executed.
# #         raw_query: The raw recognized text or error token.
# #     """
# #     raw_query = capture_fn()
# #     if not raw_query or raw_query.lower() in {ERROR_NO_SPEECH.lower(), ERROR_SERVICE.lower()}:
# #         return None, None, raw_query
# #     if run_query is None:  # Should not happen in normal runtime
# #         return None, None, raw_query
# #     # run_query may return either (result, updated_context) or expanded signature in future
# #     out = run_query(df, raw_query)  # type: ignore[arg-type]
# #     if isinstance(out, tuple) and len(out) == 2:
# #         result, updated_context = out
# #     else:  # Fallback if implementation changes
# #         result, updated_context = out, {}
# #     return result, updated_context, raw_query


# """Voice input utilities for EchoAnalytics (Phase 6).
# - capture_voice(): microphone → text (or error token)
# We let app.py run the unified query path to avoid double-execution.
# """
# import speech_recognition as sr

# ERROR_NO_SPEECH = "No speech detected"
# ERROR_SERVICE = "Speech service error"

# def capture_voice() -> str:
#     """Capture speech from microphone and return recognized text or an error token."""
#     recognizer = sr.Recognizer()
#     mic = sr.Microphone()
#     with mic as source:
#         print("Listening...")
#         recognizer.adjust_for_ambient_noise(source, duration=0.4)
#         audio = recognizer.listen(source)
#     try:
#         text = recognizer.recognize_google(audio)
#         print(f"Recognized: {text}")
#         return text.strip()
#     except sr.UnknownValueError:
#         print("Speech Recognition could not understand audio")
#         return ERROR_NO_SPEECH
#     except sr.RequestError:
#         print("Speech Recognition service error")
#         return ERROR_SERVICE


"""Voice input utilities for EchoAnalytics.

capture_voice(): mic → text or error token.
We keep execution of queries inside app.py to avoid double-execution bugs.
"""
import speech_recognition as sr

ERROR_NO_SPEECH = "No speech detected"
ERROR_SERVICE = "Speech service error"

def capture_voice() -> str:
    """Capture speech from microphone and return recognized text or an error token."""
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.4)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"Recognized: {text}")
        return text.strip()
    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio")
        return ERROR_NO_SPEECH
    except sr.RequestError:
        print("Speech Recognition service error")
        return ERROR_SERVICE
