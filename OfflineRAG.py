import json
import re
import time
from vosk import Model, KaldiRecognizer  # STT
import pyaudio
import pyttsx3  # TTS
import ollama
from dotenv import load_dotenv
import os

# Configurações iniciais
load_dotenv()

BASE_DE_CONHECIMENTO = """
O que é a ARI:
A ARI Área de Recomendações Inteligentes é uma solução inovadora do Banco do Brasil que utiliza Inteligência Artificial Generativa e Analytics para fornecer recomendações personalizadas e insights valiosos para empresas.
Como funciona a ARI: 
A ARI transforma dados financeiros brutos (extratos, saldos, transações e fluxo de caixa) em dicas práticas e personalizadas, auxiliando empreendedores na gestão financeira e operacional de seus negócios.
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

class OfflineSpeechEngine:
    def __init__(self):
        # Modelo Vosk atualizado para melhor desempenho
        self.model = Model("model/pt-small-model")
        self.recognizer = KaldiRecognizer(self.model, 16000)
        # Configuração de áudio otimizada
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=8192,  # Buffer maior para estabilidade
            input_device_index=None,
            stream_callback=None
        )
        # Configuração TTS original mantida
        self.tts = pyttsx3.init()
        self.configurar_voz()

    def configurar_voz(self):
        """Configura propriedades da voz sintetizada"""
        self.tts.setProperty('rate', 195)  # Velocidade da fala
        self.tts.setProperty('volume', 1.0)  # Volume da fala
        for voz in self.tts.getProperty('voices'):
            if 'portugues' in voz.id.lower():
                self.tts.setProperty('voice', voz.id)
                break

    def processar_pontuacao(self, texto):
        """
        Insere pausas no texto com base nas pontuações.
        - '.' -> pausa longa
        - ',' -> pausa curta
        - '?'/'!' -> pausa média
        - ':'/';' -> pausa média
        """
        # Substitui pontuações por versões com pausas
        texto = re.sub(r'\.', '. <pause>', texto)  # Ponto final
        texto = re.sub(r',', ',<short_pause>', texto)  # Vírgula
        texto = re.sub(r'[?!]', '?<medium_pause>', texto)  # Interrogação ou exclamação
        texto = re.sub(r'[:;]', ':<medium_pause>', texto)  # Dois pontos ou ponto-e-vírgula

        # Substitui as tags de pausa por silêncios
        texto = texto.replace('<pause>', '...')  # Pausa longa
        texto = texto.replace('<short_pause>', ', ')  # Pausa curta
        texto = texto.replace('<medium_pause>', '. ')  # Pausa média

        return texto
    
    def capturar_audio(self):
        """Captura áudio com tratamento de overflow aprimorado"""
        print("\033[1;33mPergunte sobre a ARI...\033[0m")
        while True:
            dados = self.stream.read(4096, exception_on_overflow=False)
            if self.recognizer.AcceptWaveform(dados):
                resultado = json.loads(self.recognizer.Result())
                texto = resultado.get('text', '').strip()
                return self.limpar_transcricao(texto)

    def limpar_transcricao(self, texto):
        """Remove artefatos comuns em transcrições"""
        return re.sub(r'\b(uhm|ah|hum)\b', '', texto, flags=re.IGNORECASE)

    def sintetizar_voz(self, texto):
        """Sintetiza o texto com pausas adicionadas"""
        try:
            # Processa o texto para adicionar pausas
            texto_com_pausas = self.processar_pontuacao(texto)

            # Limpa caracteres especiais desnecessários
            texto_limpo = re.sub(r'[^\w\sà-úÀ-Ú.,!?;:]', '', texto_com_pausas)

            # Sintetiza o texto com pausas
            self.tts.say(texto_limpo)
            self.tts.runAndWait()
        except Exception as e:
            print(f"\033[1;31mErro na síntese: {str(e)}\033[0m")
class AssistenteVirtual:
    def __init__(self):
        self.engine = OfflineSpeechEngine()
        self.historico = [
            {"role": "system", "content": f"""Você é a ARI - Área de Recomendações inteligêntes que apoia pequenos empreendedores na gestão do seu negócio!
            Sua principal função é utilizar a base de conhecimento {BASE_DE_CONHECIMENTO} fornecida para responder às perguntas dos usuários da forma mais clara e concisa possível.
            Responda em português do Brasil, seguindo estas regras:
            1. Responda uma única vez de forma objetiva e com no máximo 300 caracteres.
            2. Linguagem natural, sem termos técnicos.
            3. Evite falar "Segundo a ARI".
            4. Não leia as pontuações das frases ("," "." ":" ";" "?" "!")
            4. Concentre-se em resolver problemas cotidianos dos empreendedores, usando exemplos da base de conhecimento sempre que possível."""}
        ]

    def gerar_resposta(self, pergunta):
        """Gera resposta usando modelo local Ollama"""
        self.historico.append({"role": "user", "content": pergunta})

        try:
            resposta_stream = ollama.chat(
                model="gemma2:2b",
                messages=self.historico,
                stream=True,
            )

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
            print("\033[1;36mSistema inicializado. Aguardando comandos...\033[0m")
            while True:
                self.processar_comando()
                time.sleep(0.5)
    
    def processar_comando(self):  # ★ Método obrigatório
        try:
            pergunta = self.engine.capturar_audio()
            if not pergunta:
                return

            print(f"\033[1;34mUsuário:\033[0m {pergunta}")
            resposta = self.gerar_resposta(pergunta)
            print(f"\033[1;32mARI:\033[0m {resposta}")
            self.engine.sintetizar_voz(resposta)

        except Exception as e:
            erro = f"Desculpe, ocorreu um erro: {str(e)}"
            print(f"\033[1;31mERRO:\033[0m {erro}")
            self.engine.sintetizar_voz(erro)
if __name__ == "__main__":
    assistente = AssistenteVirtual()
    assistente.executar()
