import azure.cognitiveservices.speech as speechsdk
import ollama
import re
from dotenv import load_dotenv
import os
import time

# Carregar variáveis de ambiente
load_dotenv()

class AzureSpeechClient:
    def __init__(self):
        # Configuração do reconhecedor de fala
        self.speech_config = speechsdk.SpeechConfig(
            subscription=os.getenv("AZURE_SPEECH_KEY"), 
            region=os.getenv("AZURE_SERVICE_REGION")
        )
        self.speech_config.speech_recognition_language = "pt-BR"
        
        # Configuração da síntese de voz
        self.speech_config.speech_synthesis_voice_name = "pt-BR-FranciscaNeural"
        self.synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config
        )

    def recognize_from_microphone(self):
        """Reconhecimento de fala usando o microfone padrão"""
        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config
        )
        
        try:
            print("Ouvindo...")
            result = recognizer.recognize_once_async().get()
            
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return result.text
            elif result.reason == speechsdk.ResultReason.NoMatch:
                raise Exception("Não foi possível entender o áudio")
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = speechsdk.CancellationDetails.from_result(result)
                raise Exception(f"Erro no reconhecimento: {cancellation.error_details}")
                
        except Exception as e:
            raise RuntimeError(f"Erro no STT: {str(e)}")

    def synthesize_speech(self, text):
        """Síntese de voz com SSML para melhor controle da voz"""
        ssml = f"""
        <speak version="1.0" xml:lang="pt-BR">
            <voice name="pt-BR-FranciscaNeural">
                <prosody rate="medium" pitch="default">
                    {text}
                </prosody>
            </voice>
        </speak>
        """
        
        try:
            result = self.synthesizer.speak_ssml_async(ssml).get()
            if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                raise Exception(f"Erro na síntese de voz: {result.error_details}")
        except Exception as e:
            raise RuntimeError(f"Erro no TTS: {str(e)}")

class BetterAIVoiceAgent:
    def __init__(self):
        self.speech_client = AzureSpeechClient()
        self.full_transcript = [
            {"role": "system", "content": """Seja uma assistente virtual brasileira chamada ARI (Área de Recomendações Inteligentes). 
            Responda de forma clara e concisa, seguindo estas regras:
            1. Respostas curtas (até 300 caracteres)
            2. Gramática brasileira correta
            3. Sem anglicismos ou regionalismos
            4. Formatação simples sem caracteres especiais
            5. Sem emojis ou notação markdown"""}
        ]

    def process_request(self):
        try:
            # Captura de áudio e reconhecimento
            question = self.speech_client.recognize_from_microphone()
            print(f"Pergunta: {question}")
            
            # Geração da resposta
            resposta = self.generate_ai_response(question)
            print(f"Resposta: {resposta}")
            
            # Síntese de voz
            self.speech_client.synthesize_speech(resposta)
            
        except Exception as e:
            error_message = f"Desculpe, ocorreu um erro: {str(e)}"
            print(error_message)
            self.speech_client.synthesize_speech(error_message)

    def generate_ai_response(self, question):
        try:
            self.full_transcript.append({"role": "user", "content": question})
            
            stream_response = ollama.chat(
                model="llama3.2:latest",
                messages=self.full_transcript,
                stream=True,
            )
            
            resposta = "".join(chunk['message']['content'] for chunk in stream_response)
            resposta = self.clean_response(resposta)
            
            if not resposta.strip():
                return "Olá! Como posso ajudar hoje?"
                
            self.full_transcript.append({"role": "assistant", "content": resposta})
            return resposta
            
        except Exception as e:
            print(f"Erro na geração da resposta: {str(e)}")
            return "Houve um problema ao processar sua solicitação"

    def clean_response(self, text):
        """Limpeza de formatação e conteúdo indesejado"""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        cleaned = re.sub(r"[*()\"#@]", "", cleaned)
        return cleaned.strip()

    def run(self):
        print("Sistema inicializado. Aguardando comando de voz...")
        while True:
            self.process_request()
            time.sleep(1)

if __name__ == "__main__":
    agent = BetterAIVoiceAgent()
    agent.run()
