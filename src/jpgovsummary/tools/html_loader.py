import requests
import re
import chardet
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
        # Get plain text
        plain_text = re.sub(r'<[^>]+>', '', html_content)
        
        hyperlinks = []
        offset = 0
        
        # Extract the full anchor tags with their content
        full_link_pattern = r'<a[^>]*href=(?:"([^"]*)"|\'([^\']*)\')[^>]*>(.*?)</a>'
        full_matches = list(re.finditer(full_link_pattern, html_content, re.DOTALL))
        
        for match in full_matches:
            # URL is in group 1 (double quotes) or group 2 (single quotes)
            url = match.group(1) if match.group(1) else match.group(2)
            link_content = match.group(3)
            
            # Extract text content, removing any HTML tags (like <img>)
            link_text = re.sub(r'<[^>]+>', '', link_content).strip()
            
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
                        # Normalize whitespace for comparison
                        normalized_cell_content = re.sub(r'\s+', ' ', cell_content)
                        
                        # Find matching cell hyperlinks by normalized content
                        matching_hyperlinks = None
                        for cell_key, hyperlinks in self.cell_hyperlinks.items():
                            normalized_key = re.sub(r'\s+', ' ', cell_key.strip())
                            if normalized_cell_content == normalized_key:
                                matching_hyperlinks = hyperlinks
                                break
                        
                        if cell_content and matching_hyperlinks:
                            # Process this cell's hyperlinks
                            hyperlinks = matching_hyperlinks
                            processed_cell = cell_content
                            
                            # Process hyperlinks by replacing text with markdown links
                            for link in hyperlinks:
                                url = link.get('url', '')
                                link_text = link.get('text', '')
                                
                                # Replace the link text with markdown format
                                if link_text in processed_cell:
                                    formatted_link = f"[{link_text}]({url})"
                                    processed_cell = processed_cell.replace(link_text, formatted_link, 1)
                            
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


def _detect_encoding(content: bytes, headers: dict = None) -> str:
    """
    Detect encoding from HTTP headers, HTML meta tags, or using chardet.
    
    Args:
        content: Raw bytes content
        headers: HTTP response headers
        
    Returns:
        str: detected encoding
    """
    # 1. Check HTTP Content-Type header
    if headers:
        content_type = headers.get('content-type', '').lower()
        if 'charset=' in content_type:
            charset = content_type.split('charset=')[1].split(';')[0].strip()
            logger.info(f"Encoding detected from HTTP header: {charset}")
            return charset
    
    # 2. Check HTML meta tags
    try:
        # Try to decode with common encodings to find meta tags
        for try_encoding in ['utf-8', 'shift_jis', 'cp932', 'euc-jp', 'iso-2022-jp']:
            try:
                html_text = content.decode(try_encoding)
                # Look for charset in meta tags
                charset_match = re.search(r'<meta[^>]+charset[="\s]+([^">\s]+)', html_text, re.IGNORECASE)
                if charset_match:
                    detected_charset = charset_match.group(1).lower()
                    logger.info(f"Encoding detected from HTML meta tag: {detected_charset}")
                    return detected_charset
                break
            except UnicodeDecodeError:
                continue
    except Exception:
        pass
    
    # 3. Use chardet as fallback
    detection = chardet.detect(content)
    if detection and detection['encoding']:
        logger.info(f"Encoding detected by chardet: {detection['encoding']} (confidence: {detection['confidence']})")
        return detection['encoding']
    
    # Default fallback
    logger.warning("Could not detect encoding, defaulting to utf-8")
    return 'utf-8'


def _decode_content(content: bytes, headers: dict = None) -> str:
    """
    Decode bytes content to string with proper encoding detection.
    
    Args:
        content: Raw bytes content
        headers: HTTP response headers
        
    Returns:
        str: decoded string content
    """
    detected_encoding = _detect_encoding(content, headers)
    
    # Try detected encoding first
    try:
        return content.decode(detected_encoding)
    except UnicodeDecodeError:
        logger.warning(f"Failed to decode with detected encoding {detected_encoding}")
    
    # Try common Japanese encodings
    for encoding in ['shift_jis', 'cp932', 'euc-jp', 'iso-2022-jp', 'utf-8']:
        try:
            decoded = content.decode(encoding)
            logger.info(f"Successfully decoded with {encoding}")
            return decoded
        except UnicodeDecodeError:
            continue
    
    # Last resort: decode with errors='replace'
    logger.error("Could not decode with any encoding, using utf-8 with error replacement")
    return content.decode('utf-8', errors='replace')


def _clean_html_for_lxml(html_content: str) -> str:
    """
    Clean HTML content to make it compatible with lxml.
    
    Args:
        html_content: HTML content as string
        
    Returns:
        str: cleaned HTML content
    """
    # Remove XML declaration if present
    html_content = re.sub(r'<\?xml[^>]*\?>', '', html_content, flags=re.IGNORECASE)
    
    # Remove encoding declaration from meta tags that might conflict
    html_content = re.sub(
        r'<meta\s+[^>]*http-equiv\s*=\s*["\']content-type["\'][^>]*>', 
        '', 
        html_content, 
        flags=re.IGNORECASE
    )
    
    # Remove standalone encoding meta tags that might cause issues
    html_content = re.sub(
        r'<meta\s+charset\s*=\s*["\'][^"\']*["\'][^>]*>', 
        '', 
        html_content, 
        flags=re.IGNORECASE
    )
    
    return html_content.strip()


def _normalize_and_convert_html(html_content: str | bytes, headers: dict = None) -> str:
    """
    Normalize HTML with lxml and convert to markdown using Docling.
    
    Args:
        html_content: HTML content as string or bytes
        headers: HTTP response headers for encoding detection
        
    Returns:
        str: markdown content
    """
    # Convert bytes to string if needed
    if isinstance(html_content, bytes):
        html_content = _decode_content(html_content, headers)
    
    # Clean HTML content for lxml compatibility
    cleaned_html = _clean_html_for_lxml(html_content)
    
    # lxmlでHTMLを正規化
    try:
        doc = html.fromstring(cleaned_html)
        normalized_html = etree.tostring(doc, encoding='unicode', method='html')
    except Exception as e:
        logger.warning(f"lxml parsing failed: {e}, trying with fragment parsing")
        # Try parsing as fragment if full document parsing fails
        try:
            doc = html.fragment_fromstring(cleaned_html)
            if hasattr(doc, 'tag'):
                # Single element fragment
                normalized_html = etree.tostring(doc, encoding='unicode', method='html')
            else:
                # Multiple fragments, wrap in div
                wrapper = html.Element('div')
                if isinstance(doc, list):
                    for fragment in doc:
                        wrapper.append(fragment)
                else:
                    wrapper.append(doc)
                normalized_html = etree.tostring(wrapper, encoding='unicode', method='html')
        except Exception as e2:
            logger.error(f"Fragment parsing also failed: {e2}, using original content")
            normalized_html = cleaned_html
    
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
    It automatically detects and handles various encodings including Shift_JIS.

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

        # Read local HTML file as bytes for encoding detection
        with open(file_path, 'rb') as f:
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
        
        return _normalize_and_convert_html(response.content, response.headers)
