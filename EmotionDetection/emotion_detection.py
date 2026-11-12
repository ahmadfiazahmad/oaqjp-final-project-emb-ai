"""Emotion detection using the Watson NLP EmotionPredict service."""
import json
import requests

URL = ("https://sn-watson-emotion.labs.skills.network/v1/"
       "watson.runtime.nlp.v1/NlpService/EmotionPredict")
HEADERS = {"grpc-metadata-mlmodel-id": "emotion_aggregated-workflow_lang_en_stock"}
EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness"]


def emotion_detector(text_to_analyze):
    """Return the emotion scores and dominant emotion for the given text.

    On blank input or a 400 response from the service, every value in the
    returned dictionary is None.
    """
    empty_result = {emotion: None for emotion in EMOTIONS}
    empty_result["dominant_emotion"] = None

    if not text_to_analyze or not str(text_to_analyze).strip():
        return empty_result

    payload = {"raw_document": {"text": text_to_analyze}}
    response = requests.post(URL, json=payload, headers=HEADERS, timeout=30)

    if response.status_code == 400:
        return empty_result

    response.raise_for_status()
    scores = json.loads(response.text)["emotionPredictions"][0]["emotion"]

    result = {emotion: scores[emotion] for emotion in EMOTIONS}
    result["dominant_emotion"] = max(EMOTIONS, key=lambda emotion: result[emotion])
    return result
