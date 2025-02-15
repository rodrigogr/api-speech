import json
import re
import time
from vosk import Model, KaldiRecognizer
import pyaudio
import pyttsx3
import ollama
from dotenv import load_dotenv
import os

# Configurações iniciais
load_dotenv()

class OfflineSpeechEngine:
    def __init__(self):
        # Carregar modelo Vosk para português brasileiro
        self.model = Model("model/pt-small-model")
        self.recognizer = KaldiRecognizer(self.model, 16000)
        
        # Configurar captura de áudio com PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=8192
        )
        
        # Configurar síntese de voz offline
        self.tts = pyttsx3.init()
        self.configurar_voz()

    def configurar_voz(self):
        """Configura propriedades da voz sintetizada"""
        self.tts.setProperty('rate', 160)  # Velocidade da fala
        self.tts.setProperty('volume', 1.0)  # Volume (0.0 a 1.0)
        # Selecionar voz feminina em português se disponível
        for voz in self.tts.getProperty('voices'):
            if 'portugues' in voz.id.lower():
                self.tts.setProperty('voice', voz.id)
                break

    def capturar_audio(self):
        """Captura e transcreve áudio em tempo real"""
        print("\033[1;33mAguardando comando...\033[0m")
        while True:
            dados = self.stream.read(4096, exception_on_overflow=False)
            if self.recognizer.AcceptWaveform(dados):
                resultado = json.loads(self.recognizer.Result())
                return resultado.get('text', '').strip()

    def sintetizar_voz(self, texto):
        """Converte texto em fala com tratamento de erros"""
        try:
            # Limpar texto antes da síntese
            texto_limpo = re.sub(r'[^\w\sà-úÀ-Ú]', '', texto)
            self.tts.say(texto_limpo)
            self.tts.runAndWait()
        except Exception as e:
            print(f"\033[1;31mErro na síntese de voz: {str(e)}\033[0m")

class AssistenteVirtual:
    def __init__(self):
        self.engine = OfflineSpeechEngine()
        self.historico = [
            {"role": "system", "content": """Você é ARI, uma assistente virtual brasileira. 
            Responda de forma clara e concisa em português do Brasil, seguindo estas regras:
            1. Apena uma resposta objetiva com no máximo 300 caracteres
            2. Linguagem natural sem termos técnicos
            3. Foco na solução de problemas cotidianos"""}
        ]

    def processar_comando(self):
        try:
            # Fase 1: Captura e transcrição de voz
            pergunta = self.engine.capturar_audio()
            if not pergunta:
                return
            
            print(f"\033[1;34mUsuário:\033[0m {pergunta}")
            
            # Fase 2: Geração da resposta com Ollama
            resposta = self.gerar_resposta(pergunta)
            print(f"\033[1;32mARI:\033[0m {resposta}")
            
            # Fase 3: Síntese vocal
            self.engine.sintetizar_voz(resposta)

        except Exception as e:
            erro = f"Desculpe, ocorreu um erro: {str(e)}"
            print(f"\033[1;31mERRO:\033[0m {erro}")
            self.engine.sintetizar_voz(erro)

    def gerar_resposta(self, pergunta):
        """Gera resposta usando modelo local Ollama"""
        self.historico.append({"role": "user", "content": pergunta})
        
        try:
            resposta_stream = ollama.chat(
                model="llama3.2:latest",
                messages=self.historico,
                stream=True,
            )
            
            # Construir resposta incrementalmente
            resposta_completa = []
            for pedaco in resposta_stream:
                conteudo = pedaco['message']['content']
                resposta_completa.append(conteudo)
                print(conteudo, end='', flush=True)
            
            print()  # Nova linha após o stream
            resposta_final = self.limpar_resposta(''.join(resposta_completa))
            self.historico.append({"role": "assistant", "content": resposta_final})
            return resposta_final

        except Exception as e:
            print(f"\033[1;31mErro no modelo: {str(e)}\033[0m")
            return "Houve um problema ao processar sua solicitação"

    def limpar_resposta(self, texto):
        """Remove caracteres especiais e formatação indesejada"""
        texto_limpo = re.sub(r'[\*\_\[\]\(\)]', '', texto)
        return re.sub(r'\s+', ' ', texto_limpo).strip()

    def executar(self):
        """Loop principal de execução"""
        print("\033[1;36mSistema inicializado. Diga 'ARI' para ativar.\033[0m")
        while True:
            self.processar_comando()
            time.sleep(0.5)

if __name__ == "__main__":
    assistente = AssistenteVirtual()
    assistente.executar()