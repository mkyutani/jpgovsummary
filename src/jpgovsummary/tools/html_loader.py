import requests
import re
from io import BytesIO
from typing import List, Dict
from langchain_core.tools import tool
from lxml import etree, html
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc.document import DoclingDocument
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer

from .. import logger
from ..utils import is_local_file, get_local_file_path, validate_local_file


class HyperlinkExtractor:
    """Helper class for extracting hyperlink information from HTML"""
    
    @staticmethod
    def extract_hyperlinks_from_html_cell(html_content: str) -> tuple[str, List[Dict]]:
        """Extract hyperlinks from HTML cell content"""
        link_pattern = r'<a\s+href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>'
        matches = list(re.finditer(link_pattern, html_content))
        
        # Get plain text
        plain_text = re.sub(r'<[^>]+>', '', html_content)
        
        hyperlinks = []
        offset = 0
        
        for match in matches:
            url = match.group(1)
            link_text = match.group(2)
            
            # Find position in plain text
            start_pos = plain_text.find(link_text, offset)
            if start_pos != -1:
                end_pos = start_pos + len(link_text)
                hyperlinks.append({
                    'start': start_pos,
                    'end': end_pos,
                    'url': url,
                    'text': link_text
                })
                offset = end_pos
        
        return plain_text, hyperlinks
    
    @staticmethod
    def extract_cell_hyperlinks_from_html(original_html: str) -> Dict:
        """Extract hyperlink information from HTML and return as dictionary"""
        cell_hyperlinks = {}
        
        # Extract table cells from HTML
        cell_pattern = r'<td[^>]*>(.*?)</td>'
        html_cells = re.findall(cell_pattern, original_html, re.DOTALL)
        
        for html_cell in html_cells:
            plain_text, hyperlinks = HyperlinkExtractor.extract_hyperlinks_from_html_cell(html_cell)
            if hyperlinks:
                cell_hyperlinks[plain_text] = hyperlinks
        
        return cell_hyperlinks


class CustomMarkdownSerializer(MarkdownDocSerializer):
    """Custom Markdown exporter that handles hyperlinks in table cells"""
    
    def __init__(self, doc: DoclingDocument, original_html: str = None):
        super().__init__(doc=doc)
        # Use composition for hyperlink extraction
        self._hyperlink_extractor = HyperlinkExtractor()
        # Store in __dict__ to avoid Pydantic validation
        self.__dict__['cell_hyperlinks'] = (
            self._hyperlink_extractor.extract_cell_hyperlinks_from_html(original_html) 
            if original_html else {}
        )
    
    def serialize_hyperlink(self, text: str, hyperlink=None, **kwargs) -> str:
        """Serialize hyperlink in Markdown format - following TextItem pattern"""
        # Handle both old (url) and new (hyperlink) parameter formats
        if hyperlink is not None:
            url = str(hyperlink)
        else:
            url = kwargs.get('url', '')
        return f"[{text}]({url})"
    
    def serialize(self, *, item=None, **kwargs):
        """Override serialize to process table cells with hyperlinks"""
        # Call parent serialize method
        result = super().serialize(item=item, **kwargs)
        
        # Post-process the text to add hyperlinks in table cells
        if hasattr(self, 'cell_hyperlinks') and self.cell_hyperlinks:
            text = result.text
            
            # Parse the markdown table and process each cell
            lines = text.split('\n')
            processed_lines = []
            
            for line in lines:
                if '|' in line and line.strip() and not line.strip().replace('|', '').replace('-', '').strip() == '':
                    # This is a table content row (not separator)
                    parts = line.split('|')
                    processed_parts = []
                    
                    for part in parts:
                        cell_content = part.strip()
                        if cell_content and cell_content in self.cell_hyperlinks:
                            # Process this cell's hyperlinks
                            hyperlinks = self.cell_hyperlinks[cell_content]
                            processed_cell = cell_content
                            
                            for link in sorted(hyperlinks, key=lambda x: x.get('start', 0), reverse=True):
                                start = link.get('start', 0)
                                end = link.get('end', len(link.get('text', '')))
                                url = link.get('url', '')
                                link_text = link.get('text', '')
                                
                                if start < len(processed_cell) and end <= len(processed_cell):
                                    formatted_link = f"[{link_text}]({url})"
                                    processed_cell = processed_cell[:start] + formatted_link + processed_cell[end:]
                            
                            # Preserve spacing
                            processed_parts.append(f" {processed_cell} " if part.strip() else part)
                        else:
                            processed_parts.append(part)
                    
                    processed_lines.append('|'.join(processed_parts))
                else:
                    processed_lines.append(line)
            
            text = '\n'.join(processed_lines)
            
            # Update result with processed text
            from docling_core.transforms.serializer.base import SerializationResult
            result = SerializationResult(text=text, spans=result.spans)
        
        return result


def _normalize_and_convert_html(html_content: str | bytes) -> str:
    """
    Normalize HTML with lxml and convert to markdown using Docling.
    
    Args:
        html_content: HTML content as string or bytes
        
    Returns:
        str: markdown content
    """
    # lxmlでHTMLを正規化
    doc = html.fromstring(html_content)
    normalized_html = etree.tostring(doc, encoding='unicode', method='html')
    
    # Use Docling with custom serializer for hyperlink support
    converter = DocumentConverter()
    result = converter.convert_string(normalized_html, InputFormat.HTML, 'converted.html')
    
    # Use custom serializer that preserves table hyperlinks
    custom_serializer = CustomMarkdownSerializer(doc=result.document, original_html=normalized_html)
    enhanced_result = custom_serializer.serialize()
    
    return enhanced_result.text


def load_html_as_markdown(url: str) -> str:
    """
    Load HTML page into markdown string from URL or local file.
    
    This function handles HTML normalization using lxml before converting
    to markdown using Docling with hyperlink preservation in table cells.

    Args:
        url (str): URL of the page with ending .html or local file path

    Returns:
        str: markdown of the page
    """
    if is_local_file(url):
        # Handle local file
        file_path = get_local_file_path(url)
        validate_local_file(file_path)
        logger.info(f"{file_path} (HTML)を読み込みます")

        # Read local HTML file content
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        return _normalize_and_convert_html(html_content)
    else:
        # Handle remote URL
        logger.info(f"{url} (HTML)を読み込みます")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30, verify=True)
        response.raise_for_status()
        
        return _normalize_and_convert_html(response.content)
