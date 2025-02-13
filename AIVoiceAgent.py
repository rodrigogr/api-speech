import speech_recognition as sr
import pyttsx3
import ollama
import re
from dotenv import load_dotenv
import os

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Inicializar o motor de síntese de voz
engine = pyttsx3.init()

# Inicializar o reconhecedor de fala
recognizer = sr.Recognizer()

def remove_think_tags(response):
    # Define o padrão: tudo entre <think> e </think> (incluindo as tags)
    pattern = r"<think>.*?</think>"
    # Remove o conteúdo encontrado, inclusive com quebras de linha
    cleaned_response = re.sub(pattern, "", response, flags=re.DOTALL)
    return cleaned_response

class BetterAIVoiceAgent:
    def __init__(self):
        # Inicializa o histórico de conversa com a mensagem de sistema
        self.full_transcript = [
            {"role": "system", "content": "Seja uma assistente virtual brasileira chamada ARI Área de Recomendações Inteligentes. Responda de forma clara e concisa e de acordo com a gramática Brasileira, com respostas curtas de até 300 caracteres. Não use palavras em inglês e sotaque de Portugal. Não use caracteres especiais como *, (), "", etc. E não responda com emojis"},
        ]

    def process_request(self):
        with sr.Microphone() as source:
            print("Aguardando solicitação...")
            audio_data = recognizer.listen(source)
            try:
                # Transcreve o áudio usando o serviço Google (pt-BR)
                question = recognizer.recognize_google(audio_data, language='pt-BR')
                print("Pergunta recebida:", question)
                self.full_transcript.append({"role": "user", "content": question})

                # Gera a resposta da AI através do método abaixo
                resposta = self.generate_ai_response()
                print("Resposta: ", resposta)
                self.full_transcript.append({"role": "assistant", "content": resposta})

                # Fala a resposta utilizando o pyttsx3
                engine.say(resposta)
                engine.runAndWait()
            except sr.UnknownValueError:
                mensagem = "Desculpe, não entendi sua pergunta"
                print(mensagem)
                engine.say(mensagem)
                engine.runAndWait()
            except sr.RequestError as e:    
                mensagem = f"Erro ao acessar o serviço de reconhecimento de fala: {e}"
                print(mensagem)
                engine.say(mensagem)
                engine.runAndWait()

    def generate_ai_response(self):
        ai_response = ""
        try:
            # Chama o modelo AI (DeepSeek R1) com o histórico completo
            stream_response = ollama.chat(
                model="llama3.2:latest",
                messages=self.full_transcript,
                stream=True,
            )
            for chunk in stream_response:
                ai_response += chunk['message']['content']
            # Se a resposta estiver vazia, define uma resposta padrão
            if not ai_response.strip():
                ai_response = "Olá! Posso ajudar?"
        except Exception as e:
            ai_response = "Erro ao gerar resposta da AI."
            print("Erro ao chamar o modelo AI:", e)
        
        # Remove o conteúdo entre as tags <think> e </think>
        ai_response = remove_think_tags(ai_response)
        return ai_response

    def run(self):
        while True:
            self.process_request()

if __name__ == "__main__":
    agent = BetterAIVoiceAgent()
    agent.run()
