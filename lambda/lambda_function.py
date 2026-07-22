import os
import logging
import ask_sdk_core.utils as ask_utils

from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.dispatch_components import AbstractExceptionHandler
from ask_sdk_core.handler_input import HandlerInput

from ask_sdk_model import Response

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "aqui va la ApiKey")

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL = "gemini-3-flash-preview"

SYSTEM_INSTRUCTION = (
    "Eres un asistente muy util. Responde de forma clara y concisa en espanol. "
    "Tu respuesta debe tener maximo 400 caracteres y el texto debe ser corrido, "
    "sin saltos de linea ni listas."
)

conversation_history = []

class LaunchRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        speak_output = "Dime"
        return (
            handler_input.response_builder.speak(speak_output)
            .ask(speak_output)
            .response
        )

class GptQueryIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("GptQueryIntent")(handler_input)

    def handle(self, handler_input):
        query = handler_input.request_envelope.request.intent.slots["query"].value
        response = generate_gemini_response(query)

        return (
            handler_input.response_builder.speak(response)
            .ask("Puedes hacer otra pregunta o decir: salir.")
            .response
        )

def generate_gemini_response(query):
    try:
        contents = []
        for entry in conversation_history:
            contents.append(types.Content(
                role=entry["role"],
                parts=[types.Part.from_text(entry["content"])]
            ))
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(query)]
        ))

        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                max_output_tokens=500,
            )
        )
        reply = response.text[:400]
        conversation_history.append({"role": "user", "content": query})
        conversation_history.append({"role": "model", "content": reply})
        return reply
    except Exception as e:
        logger.error(f"Error in Gemini call: {e}", exc_info=True)
        return f"Error al generar respuesta."

class HelpIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input):
        speak_output = "Como puedo ayudarte?"
        return (
            handler_input.response_builder.speak(speak_output)
            .ask(speak_output)
            .response
        )

class CancelOrStopIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return (ask_utils.is_intent_name("AMAZON.CancelIntent")(handler_input) or
                ask_utils.is_intent_name("AMAZON.StopIntent")(handler_input))

    def handle(self, handler_input):
        speak_output = "Hasta luego."
        return handler_input.response_builder.speak(speak_output).response

class SessionEndedRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input):
        return handler_input.response_builder.response

class CatchAllExceptionHandler(AbstractExceptionHandler):
    def can_handle(self, handler_input, exception):
        return True

    def handle(self, handler_input, exception):
        logger.error(exception, exc_info=True)
        speak_output = "Lo siento, no pude procesar tu solicitud."
        return (
            handler_input.response_builder.speak(speak_output)
            .ask(speak_output)
            .response
        )

sb = SkillBuilder()
sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(GptQueryIntentHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())
sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()
