import json
import re
import time
import numpy as np
import pyaudio
from vosk import Model, KaldiRecognizer
import pyttsx3
import ollama
from dotenv import load_dotenv
import queue

load_dotenv()

class OfflineSpeechEngine:
    def __init__(self):
        # Configuração do modelo Vosk com taxa de amostragem otimizada
        self.model = Model("model/pt-small-model")
        self.recognizer = KaldiRecognizer(self.model, 16000)
        self.audio = pyaudio.PyAudio()
        self.fila_transcricoes = queue.Queue()
        
        # Configuração avançada do stream de áudio
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=self._encontrar_microfone_ideal(),
            frames_per_buffer=8000,
            stream_callback=self._processar_audio,
            input_host_api_specific_stream_info=None
        )
        
        # Configuração da síntese de voz
        self.tts = pyttsx3.init()
        self._configurar_voz()

    def _encontrar_microfone_ideal(self):
        """Seleciona o melhor dispositivo de entrada de áudio disponível"""
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0 and info['defaultSampleRate'] == 16000:
                return i
        return 0

    def _processar_audio(self, in_data, frame_count, time_info, status):
        """Callback para processamento de áudio em tempo real"""
        try:
            # Pré-processamento do áudio
            audio_array = np.frombuffer(in_data, dtype=np.int16)
            audio_array = self._aplicar_filtros(audio_array)
            
            if self.recognizer.AcceptWaveform(audio_array.tobytes()):
                resultado = json.loads(self.recognizer.Result())
                self.fila_transcricoes.put(resultado.get('text', '').strip())
        except Exception as e:
            print(f"Erro no processamento de áudio: {str(e)}")
        return (None, pyaudio.paContinue)

    def _aplicar_filtros(self, audio_array):
        """Aplica filtros de melhoria de qualidade de áudio"""
        # Normalização
        audio_array = audio_array.astype(np.float32) / 32768.0
        # Redução de ruído simples
        audio_array *= 1.2  # Amplificação seletiva
        return audio_array.astype(np.int16)

    def _configurar_voz(self):
        """Otimiza as configurações da voz sintetizada"""
        self.tts.setProperty('rate', 155)
        self.tts.setProperty('volume', 0.9)
        # Selecionar voz mais natural disponível
        for voz in self.tts.getProperty('voices'):
            if 'portugues' in voz.id.lower() and 'female' in voz.id.lower():
                self.tts.setProperty('voice', voz.id)
                break

    def capturar_audio(self):
        """Obtém a transcrição do áudio processado"""
        try:
            return self.fila_transcricoes.get(timeout=5)
        except queue.Empty:
            return ''

    def sintetizar_voz(self, texto):
        """Síntese de voz com tratamento de erros aprimorado"""
        try:
            texto_limpo = re.sub(r'[^\w\sà-úÀ-Ú]', '', texto)
            self.tts.say(texto_limpo)
            self.tts.runAndWait()
        except RuntimeError as e:
            print(f"Erro na síntese de voz: {str(e)}")

class AssistenteVirtual:
    def __init__(self):
        self.engine = OfflineSpeechEngine()
        self.historico = [
            {"role": "system", "content": "Você é a ARI - Assistente de Recomendações Inteligentes para pequenos empreendedores. Forneça respostas objetivas (até 300 caracteres) em português brasileiro, com foco em soluções práticas para gestão de negócios."}
        ]

    def processar_comando(self):
        try:
            pergunta = self.engine.capturar_audio()
            if not pergunta:
                return

            print(f"\033[1;34mUsuário:\033[0m {pergunta}")
            resposta = self._gerar_resposta(pergunta)
            print(f"\033[1;32mARI:\033[0m {resposta}")
            self.engine.sintetizar_voz(resposta)

        except Exception as e:
            erro = f"Erro no processamento: {str(e)}"
            print(f"\033[1;31mERRO:\033[0m {erro}")
            self.engine.sintetizar_voz("Desculpe, ocorreu um problema. Poderia repetir?")

    def _gerar_resposta(self, pergunta):
        """Gera resposta usando modelo local com otimizações"""
        self.historico.append({"role": "user", "content": pergunta})
        
        try:
            resposta_stream = ollama.chat(
                model="llama3.2:latest",
                messages=self.historico,
                stream=True,
                options={'temperature': 0.7, 'max_tokens': 150}
            )
            
            resposta_completa = []
            for pedaco in resposta_stream:
                conteudo = pedaco['message']['content']
                resposta_completa.append(conteudo)
                print(conteudo, end='', flush=True)
            
            print()
            resposta_final = self._limpar_resposta(''.join(resposta_completa))
            self.historico.append({"role": "assistant", "content": resposta_final})
            return resposta_final
        
        except Exception as e:
            print(f"\033[1;31mErro no modelo: {str(e)}\033[0m")
            return "Houve uma dificuldade ao processar sua solicitação. Poderia reformular?"

    def _limpar_resposta(self, texto):
        """Remove formatação indesejada mantendo pontuação essencial"""
        return re.sub(r'[\*\#\_\[\]\(\)]', '', texto).strip()

    def executar(self):
        """Loop principal de execução"""
        print("\033[1;36mSistema pronto para uso. Faça sua pergunta...\033[0m")
        while True:
            self.processar_comando()
            time.sleep(0.1)

if __name__ == "__main__":
    assistente = AssistenteVirtual()
    assistente.executar()
