import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


openai_api_key = os.getenv('OPENAI_API_KEY')

model = ChatOpenAI(model='gpt-3.5-turbo')


classificacao_chain = (
    PromptTemplate.from_template(
        '''
        Classifique a pergunta do usuario em um dos seguintes setores da empresa:
        - Gestão de Pessoas
        - Frota
        - Area de saude
        - Area Tecninca
        - Suporte Tecnico


        Pergunta: {pergunta}
        '''
    )
    | model
    | StrOutputParser()
)

gestao_pessoa_chain = (
    PromptTemplate.from_template('''
    Você é um especialista a na area de recursos humanos.
    Sempre responda as perguntas começando com a "Bem vindo ao setor de recursos humanos".
    Responda a pergunta do usuario:
    Pergunta: {pergunta}
    ''')
    | model
    | StrOutputParser()
)

frota_chain = (
    PromptTemplate.from_template('''
    Você é um especialista a na area de forta dos veiculos.
    Sempre responda as perguntas começando com a "Bem vindo ao setor de frotas".
    Responda a pergunta do usuario:
    Pergunta: {pergunta}
    ''')
    | model
    | StrOutputParser()
)

area_saude_chain = (
    PromptTemplate.from_template('''
    Você é um especialista a na area de saude trabalhista.
    Sempre responda as perguntas começando com a "Bem vindo ao setor de saude".
    Responda a pergunta do usuario:
    Pergunta: {pergunta}
    ''')
    | model
    | StrOutputParser()
)

area_tecninca_chain = (
    PromptTemplate.from_template('''
    Você é um especialista a na area de tecnica da empresa.
    Sempre responda as perguntas começando com a "Bem vindo ao setor de Tecnico".
    Responda a pergunta do usuario:
    Pergunta: {pergunta}
    ''')
    | model
    | StrOutputParser()
)

suporte_tecninco_chain = (
    PromptTemplate.from_template('''
    Você é um especialista a na area de TI da empresa.
    Sempre responda as perguntas começando com a "Bem vindo ao setor de Tecnico de TI".
    Responda a pergunta do usuario:
    Pergunta: {pergunta}
    ''')
    | model
    | StrOutputParser()
)

def rota(classificacao):
    classificacao = classificacao.lower()
    if 'Gestão de Pessoas' in classificacao:
        return gestao_pessoa_chain
    elif 'Frota' in classificacao:
        return frota_chain
    elif 'Area de saude' in classificacao:
        return area_saude_chain 
    elif 'Area Tecninca' in classificacao:
        return area_tecninca_chain
    else:
        return suporte_tecninco_chain
    
pergunta = input('Qual a sua pergunta ?')

classificacao = classificacao_chain.invoke(
    {'pergunta': pergunta}
)

response_chain = rota(classificacao=classificacao)

response = response_chain.invoke(
    {'pergunta': pergunta}
)
print(response)


