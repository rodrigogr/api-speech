import azure.cognitiveservices.speech as speechsdk
import ollama
import re
from dotenv import load_dotenv
import os
import time

# Carregar variáveis de ambiente
load_dotenv()

# Base de Conhecimento ARI
ARI_KNOWLEDGE_BASE = """
Sobre a ARI:
A ARI Área de Recomendações Inteligentes é uma solução inovadora do Banco do Brasil que utiliza Inteligência Artificial Generativa e Analytics para fornecer recomendações personalizadas e insights valiosos para empresas.

Público alvo da ARI:
A ARI foi projetada inicialmente para empresas clientes do Banco do Brasil, mas em breve atenderá pessoas físicas também. 
A solução é ideal para negócios que buscam otimizar a gestão financeira, aumentar a eficiência operacional e tomar decisões mais informadas com base em dados.

Como funciona a ARI:
A ARI utiliza dados brutos de extratos bancários, saldos, créditos, investimentos e transações para criar recomendações únicas para cada cliente. Algoritmos de IA generativa processam esses dados, que passam por curadoria humana para garantir segurança, qualidade e relevância das sugestões.

Quais os tipos de recomendações oferecidas pela ARI:
* Desempenho Financeiro: Melhorar fluxo de caixa, reduzir custos e otimizar recursos.
* Segmentação e Comportamento dos Clientes: Insights sobre padrões de consumo e tendências sazonais.
* Soluções de Crédito e Financeiro: Ofertas personalizadas de crédito (antecipação de recebíveis e linhas de capital de giro).
* Eficiência Operacional: Otimizar processos internos e melhorar a produtividade.
* Recomendações Estratégicas: Orientações para crescimento sustentável e expansão de mercado.
* Relacionamento: Melhoria na interação com clientes e fornecedores.
* Marketing e Posicionamento de Mercado: Estratégias para aumentar vendas e atrair novos clientes.
* Educação Financeira Empreendedora: Capacitação e orientações práticas para melhorar a gestão financeira.

Principais Funcionalidades:
* Análise de Comportamento de Clientes: Insights sobre o comportamento de compra (número de clientes recorrentes no PIX, melhor dia de vendas), análise do ticket médio e sugestões para aumentá-lo.
* Apoio em Datas Comemorativas e Sazonalidades: Recomendações específicas para períodos de alta demanda (feriados e datas comemorativas) para auxiliar no preparo para picos de vendas.
* Educação Financeira: Dicas práticas e orientações sobre produtos financeiros para melhorar a compreensão das finanças e a tomada de decisões.
* Integração com o Painel PJ: Recomendações disponibilizadas no Painel PJ (plataforma do Banco do Brasil que centraliza informações de pagamentos e recebimentos) para uma visão unificada das finanças.

Exemplos de Recomendações:
* Antecipação de Recebíveis: Converter vendas futuras em capital imediato.
* Capital de Giro: Alertas para clientes utilizando o limite do cheque especial ou necessitando de crédito adicional.
* Monitoramento de Transações: Análise detalhada de entradas e saídas de PIX, boletos e cartões.
* Datas Comemorativas: Sugestões de marketing e promoções alinhadas a feriados e eventos sazonais.
* Investimentos: Identificação de saldos ociosos que podem ser aplicados em produtos financeiros.
* Alertas de Custos: Notificações sobre aumentos em contas fixas (água e luz) com dicas para economizar.
* Open Finance: Incentivos para concentrar operações financeiras no Banco do Brasil, aproveitando benefícios exclusivos.

Porque o Banco do Brasil criou a ARI:
O Banco do Brasil desenvolveu a ARI para aproximar-se dos pequenos empreendedores, oferecer suporte personalizado e reforçar o compromisso com a educação financeira e a inovação tecnológica, posicionando-se como pioneiro no uso de IA generativa no mercado financeiro brasileiro.

Como acessar a ARI:
As recomendações da ARI estão disponíveis no Painel PJ (plataforma digital gratuita que consolida informações de vendas, recebimentos e fluxos de caixa). O acesso é destinado a clientes pessoa jurídica do Banco do Brasil que utilizam o Painel PJ via BB Digital PJ.

Impactos da ARI:
A ARI impacta a gestão e a tomada de decisões dos pequenos negócios, transformando dados em insights práticos, auxiliando os empreendedores a:
* Melhorar a eficiência operacional.
* Reduzir custos e otimizar recursos.
* Aumentar vendas e atrair novos clientes.
* Fortalecer a saúde financeira do negócio.

A ARI promove mudanças sustentáveis na gestão financeira, contribuindo para o crescimento de longo prazo das micro e pequenas empresas.
"""

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
        # Transcrição inicial com instruções de ARI e base de conhecimento incluso
        self.full_transcript = [
            {"role": "system", "content": f"""Seja uma assistente virtual brasileira chamada ARI (Área de Recomendações Inteligentes). 
Responda de forma clara e concisa, seguindo estas regras:
1. Respostas curtas (até 300 caracteres)
2. Gramática brasileira correta
3. Sem anglicismos ou regionalismos
4. Formatação simples sem caracteres especiais
5. Sem emojis ou notação markdown

Base de Conhecimento da ARI:
{ARI_KNOWLEDGE_BASE}
"""}
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
