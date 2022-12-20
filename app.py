"""Deploying AI Voice Chatbot Gradio App."""
from gradio import Audio, Interface, Textbox

from utils import (TextGenerationPipeline, from_en_translation,
                   html_audio_autoplay, stt, to_en_translation, tts,
                   tts_to_bytesio)

max_answer_length = 100
desired_language = "de"
response_generator_pipe = TextGenerationPipeline(max_length=max_answer_length)


def main(audio: object):
    """Calls functions for deploying gradio app.

    It responds both verbally and in text
    by taking voice input from user.

    Args:
        audio (object): recorded speech of user

    Returns:
        tuple containing

        - user_speech_text (str) : recognized speech
        - bot_response_de (str) : translated answer of bot
        - bot_response_en (str) : bot's original answer
        - html (object) : autoplayer for bot's speech
    """
    user_speech_text = stt(audio, desired_language)
    tranlated_text = to_en_translation(user_speech_text, desired_language)
    bot_response_en = response_generator_pipe(tranlated_text)
    bot_response_de = from_en_translation(bot_response_en, desired_language)
    bot_voice = tts(bot_response_de, desired_language)
    bot_voice_bytes = tts_to_bytesio(bot_voice)
    html = html_audio_autoplay(bot_voice_bytes)
    return user_speech_text, bot_response_de, bot_response_en, html


Interface(
    fn=main,
    inputs=[
        Audio(
            source="microphone",
            type="filepath",
        ),
    ],
    outputs=[
        Textbox(label="You said: "),
        Textbox(label="AI said: "),
        Textbox(label="AI said (English): "),
        "html",
    ],
    live=True,
    allow_flagging="never",
).launch(debug=True)
