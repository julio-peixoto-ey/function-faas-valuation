import logging
import time
import tempfile
import os
from typing import List
import fitz
import easyocr
from .model import DocumentModel

class OCRService:
    """Serviço especializado para OCR de documentos PDF"""
    
    def __init__(self):
        self._ocr_reader = None
        self.MAX_PAGES_PER_BATCH = 20
        self.MAX_TOTAL_PAGES_OCR = 100
        self.MAX_FILE_SIZE_MB = 50
        self.OCR_TIMEOUT_PER_PAGE = 15

    def _get_ocr_reader(self):
        """Inicializa o EasyOCR reader apenas quando necessário"""
        if self._ocr_reader is None:
            try:
                logging.info("Inicializando EasyOCR...")
                
                self._ocr_reader = easyocr.Reader(
                    ['pt', 'en'], 
                    gpu=False,
                    verbose=False,
                    download_enabled=True,
                    detector=True,
                    recognizer=True
                )
                
                logging.info("EasyOCR inicializado com sucesso")
                
            except Exception as e:
                logging.error(f"Erro ao inicializar EasyOCR: {str(e)}")
                raise Exception(f"Falha na inicialização do EasyOCR: {str(e)}")
        
        return self._ocr_reader

    def is_pdf_text_extractable(self, file_content: bytes, filename: str) -> bool:
        """Detecta se um PDF contém texto extraível ou se é uma imagem escaneada"""
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name

            doc = fitz.open(temp_path)
            
            total_text_length = 0
            total_chars_analyzed = 0
            pages_with_text = 0
            pages_with_images = 0
            
            pages_to_analyze = min(3, len(doc))
            
            for page_num in range(pages_to_analyze):
                page = doc.load_page(page_num)
                
                text = page.get_text().strip()
                if text:
                    total_text_length += len(text)
                    pages_with_text += 1
                    
                    non_space_chars = len([c for c in text if not c.isspace()])
                    total_chars_analyzed += non_space_chars
                
                image_list = page.get_images()
                if image_list:
                    pages_with_images += 1
            
            doc.close()
            
            if total_chars_analyzed == 0:
                logging.info(f"{filename}: Nenhum texto extraível encontrado - PDF escaneado")
                return False
            
            avg_chars_per_page = total_chars_analyzed / pages_to_analyze if pages_to_analyze > 0 else 0
            if avg_chars_per_page < 100:
                logging.info(f"{filename}: Texto insuficiente ({avg_chars_per_page:.1f} chars/página) - PDF escaneado")
                return False
            
            if pages_with_images > 0 and avg_chars_per_page < 300:
                logging.info(f"{filename}: PDF com imagens e pouco texto - usando OCR para garantir")
                return False
            
            logging.info(f"{filename}: PDF contém texto extraível ({avg_chars_per_page:.1f} chars/página)")
            return True
            
        except Exception as e:
            logging.error(f"Erro ao verificar se PDF é text-extractable {filename}: {str(e)}")
            return True
            
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logging.warning(f"Erro ao remover arquivo temporário: {str(e)}")

    def _should_use_sampling_strategy(self, total_pages: int, file_size_mb: float) -> bool:
        """Determina se deve usar estratégia de amostragem para documentos grandes"""
        return total_pages > self.MAX_TOTAL_PAGES_OCR or file_size_mb > self.MAX_FILE_SIZE_MB

    def _get_strategic_pages(self, total_pages: int, max_pages: int = 50) -> List[int]:
        """Seleciona páginas estratégicas de documentos grandes"""
        if total_pages <= max_pages:
            return list(range(total_pages))
        
        strategic_pages = []
        
        strategic_pages.extend(range(min(10, total_pages)))
        
        if total_pages > 20:
            middle_start = total_pages // 3
            middle_end = 2 * total_pages // 3
            middle_sample = max_pages - 20
            
            if middle_sample > 0:
                middle_pages = range(middle_start, middle_end)
                step = max(1, len(middle_pages) // middle_sample)
                strategic_pages.extend(middle_pages[::step][:middle_sample])
        
        if total_pages > 10:
            last_pages_start = max(total_pages - 10, max(strategic_pages) + 1)
            strategic_pages.extend(range(last_pages_start, total_pages))
        
        return sorted(list(set(strategic_pages)))

    def _create_progress_bar(self, percent: float, width: int = 20) -> str:
        """Cria uma barra de progresso visual"""
        filled = int(width * percent / 100)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}]"

    def _estimate_tokens(self, text: str) -> int:
        """Estima número de tokens no texto"""
        return len(text) // 4

    def extract_text_with_ocr(self, file_content: bytes, filename: str) -> List[DocumentModel]:
        """Extrai texto de PDF usando EasyOCR com estratégia otimizada para documentos grandes"""
        documents = []
        temp_path = None
        
        try:
            file_size_mb = len(file_content) / (1024 * 1024)
            logging.info(f"Iniciando OCR: {filename} ({file_size_mb:.1f}MB)")
            
            if file_size_mb > 100:
                logging.warning(f"Arquivo muito grande ({file_size_mb:.1f}MB), pode haver problemas de memória")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name

            doc = fitz.open(temp_path)
            total_pages = len(doc)
            
            use_sampling = self._should_use_sampling_strategy(total_pages, file_size_mb)
            
            if use_sampling:
                pages_to_process = self._get_strategic_pages(total_pages, self.MAX_TOTAL_PAGES_OCR)
                logging.info(f"Estratégia: {len(pages_to_process)} páginas estratégicas de {total_pages} total")
            else:
                pages_to_process = list(range(min(self.MAX_TOTAL_PAGES_OCR, total_pages)))
                logging.info(f"Processando: {len(pages_to_process)} de {total_pages} páginas")
            
            try:
                reader = self._get_ocr_reader()
            except Exception as e:
                logging.error(f"Erro ao inicializar OCR: {str(e)}")
                doc.close()
                raise Exception(f"Falha na inicialização do OCR: {str(e)}")
            
            total_processed = 0
            start_time = time.time()
            
            for batch_start in range(0, len(pages_to_process), self.MAX_PAGES_PER_BATCH):
                batch_end = min(batch_start + self.MAX_PAGES_PER_BATCH, len(pages_to_process))
                batch_pages = pages_to_process[batch_start:batch_end]
                batch_num = batch_start // self.MAX_PAGES_PER_BATCH + 1
                total_batches = (len(pages_to_process) + self.MAX_PAGES_PER_BATCH - 1) // self.MAX_PAGES_PER_BATCH
                
                logging.info(f"Lote {batch_num}/{total_batches}: Páginas {batch_pages[0]+1}-{batch_pages[-1]+1}")
                
                for page_num in batch_pages:
                    try:
                        page_start_time = time.time()
                        
                        page = doc.load_page(page_num)
                        
                        if total_pages > 200:
                            mat = fitz.Matrix(1.5, 1.5)
                        else:
                            mat = fitz.Matrix(2.0, 2.0)
                        
                        pix = page.get_pixmap(matrix=mat)
                        img_data = pix.tobytes("png")
                        
                        if len(img_data) > 5 * 1024 * 1024:
                            mat = fitz.Matrix(1.0, 1.0)
                            pix = page.get_pixmap(matrix=mat)
                            img_data = pix.tobytes("png")
                        
                        try:
                            results = reader.readtext(img_data, detail=0, width_ths=0.7, height_ths=0.7)
                            
                            page_processing_time = time.time() - page_start_time
                            if page_processing_time > self.OCR_TIMEOUT_PER_PAGE:
                                logging.warning(f"Página {page_num + 1} demorou {page_processing_time:.1f}s")
                            
                        except Exception as ocr_error:
                            logging.error(f"OCR falhou na página {page_num + 1}: {str(ocr_error)}")
                            continue
                        
                        if results:
                            filtered_results = [text.strip() for text in results if text.strip() and len(text.strip()) > 2]
                            
                            if filtered_results:
                                page_text = '\n'.join(filtered_results)
                                
                                document = DocumentModel(
                                    page_content=page_text,
                                    page_number=page_num + 1,
                                    source=filename,
                                    file_type="pdf_ocr_strategic" if use_sampling else "pdf_ocr",
                                    character_count=len(page_text),
                                    token_count=self._estimate_tokens(page_text),
                                )
                                documents.append(document)
                                total_processed += 1
                                
                                if total_processed % 10 == 0 or total_processed % 5 == 0:
                                    elapsed_time = time.time() - start_time
                                    avg_time_per_page = elapsed_time / total_processed
                                    estimated_remaining = avg_time_per_page * (len(pages_to_process) - total_processed)
                                    
                                    progress_percent = (total_processed / len(pages_to_process)) * 100
                                    progress_bar = self._create_progress_bar(progress_percent)
                                    
                                    logging.info(f"Progresso: {progress_bar} {total_processed}/{len(pages_to_process)} páginas " +
                                               f"({progress_percent:.1f}%) - {avg_time_per_page:.1f}s/página - " +
                                               f"ETA: {estimated_remaining/60:.1f}min")
                    
                    except Exception as e:
                        logging.error(f"Erro na página {page_num + 1}: {str(e)}")
                        continue
                
                if batch_end < len(pages_to_process):
                    time.sleep(0.5)
            
            doc.close()
            
            total_time = time.time() - start_time
            total_chars = sum(len(doc.page_content) for doc in documents)
            
            logging.info(f"OCR Completo: {len(documents)} páginas, {total_chars:,} caracteres, " +
                        f"{total_time/60:.1f}min total")
            
            if not documents:
                logging.warning(f"Nenhum texto extraído via OCR")
                raise Exception("Nenhum texto extraído via OCR")
            
            return documents
            
        except Exception as e:
            logging.error(f"Erro geral na extração OCR: {str(e)}")
            raise
            
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logging.warning(f"Erro ao remover arquivo temporário: {str(e)}") 