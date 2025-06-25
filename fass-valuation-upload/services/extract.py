from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import logging as logger
import os
import time
import base64
import json
from typing import List, Dict, Any
from ..models.entites_models import ExtractedEntity, ContractEntity, DocumentExtractionResult
from ..utils.token_counter import TokenCounter

def get_required_env_var(var_name: str, default_value: str = None) -> str:
    value = os.getenv(var_name, default_value)
    if not value:
        logger.error(f"Variável de ambiente obrigatória não encontrada: {var_name}")
        raise ValueError(f"Variável de ambiente {var_name} não configurada")
    return value

try:
    AZURE_OPENAI_ENDPOINT = get_required_env_var("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_KEY = get_required_env_var("AZURE_OPENAI_KEY") 
    AZURE_DEPLOYMENT_NAME = get_required_env_var("AZURE_DEPLOYMENT_NAME")
    AZURE_OPENAI_API_VERSION = get_required_env_var("AZURE_OPENAI_API_VERSION", "2024-06-01")
    AZURE_EMBEDDING_DEPLOYMENT = get_required_env_var("AZURE_EMBEDDING_DEPLOYMENT")
    
    logger.info("Todas as variáveis de ambiente carregadas com sucesso")
except ValueError as e:
    logger.error(f"Erro na configuração: {e}")
    raise

class DocumentEntityExtractor:
    """Extrator de entidades específicas de documentos usando embeddings"""
    
    def __init__(self):
        self.token_counter = TokenCounter()
        self.llm = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
            azure_deployment=AZURE_DEPLOYMENT_NAME,
            api_version=AZURE_OPENAI_API_VERSION,
            temperature=0.1,
            max_tokens=4000
        )

        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
            azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
            api_version=AZURE_OPENAI_API_VERSION,
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
    
    def extract_all_entities(self, document_chunks: List[Dict[str, Any]], filename: str) -> DocumentExtractionResult:
        """Extrai todas as entidades em uma única chamada otimizada"""
        start_time = time.time()
        
        try:
            vector_store = self.create_vector_store_from_chunks(document_chunks)
            
            combined_query = """
            Contratos financeiros: indexação de juros, spread, taxas, datas de emissão e vencimento, 
            valores nominais, cronogramas de pagamento, atualização monetária IPCA IGPM SELIC, 
            bases de cálculo 252 365 dias, fluxos de amortização, DI CDI pré-fixado
            """
            
            docs = vector_store.similarity_search_with_score(combined_query, k=20)
            
            if not docs:
                return self._create_empty_result(filename, start_time)
            
            context_parts = []
            all_page_refs = set()
            
            for doc, score in docs:
                score_float = float(score)
                context_parts.append(f"Chunk {doc.metadata.get('chunk_id', 'N/A')} (página {doc.metadata.get('page', 'N/A')}):\n{doc.page_content}")
                if 'page' in doc.metadata:
                    all_page_refs.add(int(doc.metadata['page']))
            
            full_context = "\n\n---\n\n".join(context_parts)
            
            prompt = f"""
            Analise o contexto do documento financeiro e extraia as informações específicas solicitadas.

            CONTEXTO DO DOCUMENTO:
            {full_context}

            INSTRUÇÕES IMPORTANTES:
            - Para cada item, extraia EXATAMENTE a informação encontrada no documento
            - Se não encontrar a informação, responda "NÃO ENCONTRADO"
            - Responda sempre com STRING (texto), nunca com arrays ou listas
            - Se encontrar múltiplos valores, junte-os com vírgula em uma única string
            - Seja preciso e objetivo
            - Mantenha o formato JSON estrito

            EXTRAIA AS SEGUINTES INFORMAÇÕES:

            1. ATUALIZAÇÃO MONETÁRIA: Índices de correção (IPCA, IGPM, SELIC, etc.)
            2. JUROS REMUNERATÓRIOS: Indexação principal (DI, CDI, taxa pré-fixada, etc.)
            3. SPREAD FIXO: Percentual de spread ou sobretaxa (ex: 2% a.a., 1.5% ao ano)
            4. BASE DE CÁLCULO: Metodologia de cálculo (252 dias úteis, 365 dias corridos, etc.)
            5. DATA EMISSÃO: Data de emissão, subscrição ou início de vigência
            6. DATA VENCIMENTO: Data de vencimento, resgate ou término da operação
            7. VALOR NOMINAL UNITÁRIO: Valor nominal unitário, valor da cota ou valor principal
            8. FLUXOS PAGAMENTO: Datas específicas de pagamento de juros e amortização
            9. FLUXOS PERCENTUAIS: Percentuais ou frações de amortização para cada data

            RESPONDA EM JSON VÁLIDO COM STRINGS:
            {{
                "atualizacao_monetaria": "valor ou NÃO ENCONTRADO",
                "juros_remuneratorios": "valor ou NÃO ENCONTRADO", 
                "spread_fixo": "valor ou NÃO ENCONTRADO",
                "base_calculo": "valor ou NÃO ENCONTRADO",
                "data_emissao": "valor ou NÃO ENCONTRADO",
                "data_vencimento": "valor ou NÃO ENCONTRADO",
                "valor_nominal_unitario": "valor ou NÃO ENCONTRADO",
                "fluxos_pagamento": "valor ou NÃO ENCONTRADO",
                "fluxos_percentuais": "valor ou NÃO ENCONTRADO"
            }}
            """
            
            response = self.llm.invoke(prompt)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            input_tokens = self.token_counter.count_tokens(prompt)
            output_tokens = self.token_counter.count_tokens(response_content)
            self.token_counter.usage.input_tokens += input_tokens
            self.token_counter.usage.output_tokens += output_tokens
            self.token_counter.usage.api_calls += 1
            
            try:
                json_start = response_content.find('{')
                json_end = response_content.rfind('}') + 1
                if json_start != -1 and json_end != 0:
                    json_content = response_content[json_start:json_end]
                else:
                    json_content = response_content
                
                entities_data = json.loads(json_content)
                logger.info(f"JSON parseado com sucesso: {len(entities_data)} entidades")
                
                for key, value in entities_data.items():
                    logger.info(f"Entidade '{key}': tipo={type(value)}, valor={value}")
                
            except json.JSONDecodeError as e:
                logger.warning(f"Falha no parsing JSON: {e}")
                logger.warning(f"Conteúdo recebido: {response_content[:500]}...")
                return self._extract_entities_fallback(vector_store, filename, start_time, response_content)
            
            contract_entities = ContractEntity()
            entities_found = 0
            
            for entity_name, value in entities_data.items():
                if isinstance(value, list):
                    processed_value = ", ".join(str(item) for item in value if item)
                elif value is None:
                    processed_value = ""
                else:
                    processed_value = str(value)
                
                if processed_value and processed_value.strip() and processed_value.strip().upper() != "NÃO ENCONTRADO":
                    confidence = self._calculate_confidence_from_context(processed_value, full_context)
                    
                    entity = ExtractedEntity(
                        entity_type=entity_name,
                        value=processed_value.strip(),
                        confidence=confidence,
                        page_references=sorted(list(all_page_refs)),
                        context=full_context[:500] + "..." if len(full_context) > 500 else full_context
                    )
                    setattr(contract_entities, entity_name, entity)
                    entities_found += 1
                    logger.info(f"Entidade '{entity_name}' encontrada: {processed_value.strip()[:50]}...")
            
            processing_time = int((time.time() - start_time) * 1000)
            del vector_store
            
            logger.info(f"Extração otimizada concluída: {entities_found}/9 entidades encontradas em {processing_time}ms com 1 chamada à API")
            
            return DocumentExtractionResult(
                filename=filename,
                success=True,
                contract_entities=contract_entities,
                processing_time_ms=processing_time,
                token_summary=self.token_counter.get_summary()
            )
            
        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            logger.error(f"Erro na extração otimizada: {e}")
            
            return DocumentExtractionResult(
                filename=filename,
                success=False,
                contract_entities=ContractEntity(),
                processing_time_ms=processing_time,
                token_summary=self.token_counter.get_summary(),
                error_message=str(e)
            )

    def _extract_entities_fallback(self, vector_store, filename, start_time, response_content):
        """Fallback quando JSON parsing falha - tenta extrair informações do texto"""
        logger.info("Executando fallback para parsing manual")
        
        contract_entities = ContractEntity()
        
        entity_patterns = {
            'atualizacao_monetaria': ['IPCA', 'IGPM', 'SELIC', 'correção', 'atualização'],
            'juros_remuneratorios': ['DI', 'CDI', 'pré-fixado', 'fixo', '%'],
            'spread_fixo': ['%', 'spread', 'sobretaxa', 'a.a.', 'ao ano'],
            'base_calculo': ['252', '365', 'dias', 'úteis', 'corridos'],
            'data_emissao': ['/', '-', 'emissão', 'início'],
            'data_vencimento': ['/', '-', 'vencimento', 'término'],
            'valor_nominal_unitario': ['R$', 'valor', 'nominal', 'unitário'],
            'fluxos_pagamento': ['/', '-', 'pagamento', 'cronograma'],
            'fluxos_percentuais': ['%', 'amortização', 'parcela']
        }
        
        for entity_name, patterns in entity_patterns.items():
            for pattern in patterns:
                if pattern.lower() in response_content.lower() and "NÃO ENCONTRADO" not in response_content:
                    lines = response_content.split('\n')
                    for line in lines:
                        if pattern.lower() in line.lower() and len(line.strip()) > 5:
                            entity = ExtractedEntity(
                                entity_type=entity_name,
                                value=line.strip()[:100],
                                confidence=0.5,
                                page_references=[],
                                context="Extraído via fallback"
                            )
                            setattr(contract_entities, entity_name, entity)
                            break
                    break
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return DocumentExtractionResult(
            filename=filename,
            success=True,
            contract_entities=contract_entities,
            processing_time_ms=processing_time,
            token_summary=self.token_counter.get_summary()
        )

    def _calculate_confidence_from_context(self, value: str, context: str) -> float:
        """Calcula confiança baseada na presença do valor no contexto"""
        if not value or not context:
            return 0.5
        
        if value.lower() in context.lower():
            return 0.9
        
        keywords = value.lower().split()
        found_keywords = sum(1 for keyword in keywords if keyword in context.lower())
        
        if found_keywords > 0:
            confidence = min(0.8, 0.5 + (found_keywords / len(keywords)) * 0.3)
            return round(confidence, 3)
        
        return 0.5

    def _create_empty_result(self, filename: str, start_time: float) -> DocumentExtractionResult:
        """Cria um resultado vazio quando nenhum chunk relevante é encontrado"""
        processing_time = int((time.time() - start_time) * 1000)
        
        return DocumentExtractionResult(
            filename=filename,
            success=False,
            contract_entities=ContractEntity(),
            processing_time_ms=processing_time,
            token_summary=self.token_counter.get_summary(),
            error_message="Nenhum chunk relevante encontrado no documento"
        )