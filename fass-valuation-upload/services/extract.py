from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import logging as logger
from dotenv import load_dotenv
import os
import time
import base64
import numpy as np
from typing import List, Optional, Dict, Any
from ..models.entites_models import ExtractedEntity, ContractEntity, DocumentExtractionResult
from ..utils.token_counter import TokenCounter

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")


class DocumentEntityExtractor:
    """Extrator de entidades específicas de documentos usando embeddings"""
    
    def __init__(self):
        self.token_counter = TokenCounter()
        self.llm = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
            azure_deployment="gpt-4o",
            api_version="2024-06-01",
            temperature=0.1,
            max_tokens=2000
        )

        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
            azure_deployment="text-embedding-3-large",
            api_version="2024-06-01",
        )
        
    def create_vector_store_from_chunks(self, document_chunks: List[Dict[str, Any]]) -> FAISS:
        """Cria um banco de dados vetorial FAISS a partir dos chunks já processados"""
        logger.info(f"Criando banco vetorial com {len(document_chunks)} chunks pré-processados")
        
        documents = []
        chunk_texts = []
        
        for chunk in document_chunks:
            chunk_content = base64.b64decode(chunk['content']).decode('utf-8')
            chunk_texts.append(chunk_content)
            
            doc = Document(
                page_content=chunk_content,
                metadata={
                    'chunk_id': int(chunk['chunk_id']),
                    'page': int(chunk['page']),
                    'tokens': int(chunk['tokens'])
                }
            )
            documents.append(doc)
        
        self.token_counter.log_embedding_usage(chunk_texts)
        
        vector_store = FAISS.from_documents(documents, self.embeddings)
        
        logger.info(f"Banco vetorial criado com {len(documents)} chunks")
        return vector_store
    
    def extract_entity(self, vector_store: FAISS, entity_type: str, query: str, top_k: int = 5) -> Optional[ExtractedEntity]:
        """Extrai uma entidade específica usando busca vetorial"""
        try:
            start_time = time.time()
            
            docs = vector_store.similarity_search_with_score(query, k=top_k)
            
            if not docs:
                return None
            
            context_parts = []
            page_refs = set()
            
            for doc, score in docs:
                score_float = float(score)
                context_parts.append(f"Chunk {doc.metadata.get('chunk_id', 'N/A')} (página {doc.metadata.get('page', 'N/A')}, score: {score_float:.3f}):\n{doc.page_content}")
                if 'page' in doc.metadata:
                    page_refs.add(int(doc.metadata['page']))
            
            full_context = "\n\n---\n\n".join(context_parts)
            
            prompt = f"""
            Analise o contexto extraído do documento e extraia EXATAMENTE a informação solicitada sobre: {entity_type}
            
            CONTEXTO:
            {full_context}
            
            INSTRUÇÃO: {query}
            
            RESPOSTA:
            - Se encontrar a informação, forneça APENAS o valor específico solicitado (sem explicações adicionais)
            - Se não encontrar, responda apenas: "NÃO ENCONTRADO"
            - Seja preciso e objetivo
            - Se encontrar múltiplas ocorrências, liste todas separadas por vírgula
            
            VALOR EXTRAÍDO:
            """
            
            response = self.llm.invoke(prompt)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            input_tokens = self.token_counter.count_tokens(prompt)
            output_tokens = self.token_counter.count_tokens(response_content)
            self.token_counter.usage.input_tokens += input_tokens
            self.token_counter.usage.output_tokens += output_tokens
            self.token_counter.usage.api_calls += 1
            
            processing_time = time.time() - start_time
            logger.info(f"Entidade '{entity_type}' extraída em {processing_time:.2f}s")
            
            if "NÃO ENCONTRADO" in response_content.upper():
                return None
            
            return ExtractedEntity(
                entity_type=entity_type,
                value=response_content.strip(),
                confidence=self._calculate_confidence(docs),
                page_references=sorted(list(page_refs)),
                context=full_context[:500] + "..." if len(full_context) > 500 else full_context
            )
            
        except Exception as e:
            logger.error(f"Erro ao extrair entidade '{entity_type}': {e}")
            return None
    
    def _calculate_confidence(self, docs_with_scores) -> float:
        """Calcula confiança baseada nos scores de similaridade"""
        if not docs_with_scores:
            return 0.0
        
        scores = [float(score) for _, score in docs_with_scores]
        avg_score = sum(scores) / len(scores)
        
        confidence = max(0.0, min(1.0, 1.0 - (avg_score / 2.0)))
        return round(float(confidence), 3)
    
    def extract_all_entities(self, document_chunks: List[Dict[str, Any]], filename: str) -> DocumentExtractionResult:
        """Extrai todas as entidades de um documento usando chunks pré-processados"""
        start_time = time.time()
        
        try:
            vector_store = self.create_vector_store_from_chunks(document_chunks)
            
            entity_queries = {
                'atualizacao_monetaria': "Identifique se há menção a IPCA, IGPM, SELIC ou índices de atualização monetária e correção",
                'juros_remuneratorios': "Identifique a indexação do contrato: DI, CDI, taxa pré-fixada ou outro indexador de juros",
                'spread_fixo': "Identifique o valor percentual de remuneração, spread ou sobretaxa (ex: 2% em DI+2% ou 14% em pré-fixado)",
                'base_calculo': "Identifique se a fórmula de cálculo usa base 252 ou 365 dias (dias úteis ou corridos) e metodologia de cálculo",
                'data_emissao': "Encontre a data de emissão, subscrição ou início de vigência do contrato",
                'data_vencimento': "Encontre a data de vencimento, resgate ou término da operação",
                'valor_nominal_unitario': "Identifique o valor nominal unitário, valor da cota ou valor principal do contrato",
                'fluxos_pagamento': "Encontre as datas específicas de pagamento de amortização e juros (cronograma de pagamentos)",
                'fluxos_percentuais': "Identifique os valores percentuais ou frações de amortização para cada data de pagamento"
            }
            
            contract_entities = ContractEntity()
            
            for entity_name, query in entity_queries.items():
                logger.info(f"Extraindo entidade: {entity_name}")
                entity = self.extract_entity(vector_store, entity_name, query)
                setattr(contract_entities, entity_name, entity)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            del vector_store
            
            return DocumentExtractionResult(
                filename=filename,
                success=True,
                contract_entities=contract_entities,
                processing_time_ms=processing_time,
                token_summary=self.token_counter.get_summary()
            )
            
        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            logger.error(f"Erro ao extrair entidades do documento '{filename}': {e}")
            
            return DocumentExtractionResult(
                filename=filename,
                success=False,
                contract_entities=ContractEntity(),
                processing_time_ms=processing_time,
                token_summary=self.token_counter.get_summary(),
                error_message=str(e)
            )