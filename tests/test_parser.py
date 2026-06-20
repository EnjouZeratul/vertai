"""Unit tests for DocParser module."""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vertai.data import DocParser
from vertai.data.parser import (
    ExcelParser,
    PDFParser,
    PPTParser,
    WordParser,
)


class TestBaseParser:
    """Tests for BaseParser abstract class."""

    def test_base_parser_cannot_instantiate(self):
        """Test that BaseParser cannot be instantiated directly."""
        from vertai.data.parser import BaseParser

        # BaseParser is abstract, so it should raise TypeError
        with pytest.raises(TypeError):
            BaseParser()

    def test_base_parser_create_result(self):
        """Test _create_result static method."""
        from vertai.data.parser import BaseParser

        # Test calling the static method directly
        result = BaseParser._create_result(
            text="Hello World",
            metadata={"key": "value"},
            chunks=[{"chunk": 1}]
        )

        assert result["text"] == "Hello World"
        assert result["metadata"]["key"] == "value"
        assert len(result["chunks"]) == 1

    def test_base_parser_create_result_no_chunks(self):
        """Test _create_result static method without chunks."""
        from vertai.data.parser import BaseParser

        result = BaseParser._create_result(
            text="Test",
            metadata={}
        )

        assert result["text"] == "Test"
        assert result["chunks"] == []

    def test_base_parser_abstract_parse_called(self, tmp_path: Path):
        """Test that abstract parse method can be called via subclass super()."""
        from vertai.data.parser import BaseParser

        # Create a concrete implementation that calls super().parse()
        class ConcreteParser(BaseParser):
            def parse(self, file_path: Path) -> dict[str, Any]:
                # Call super to hit the abstract pass statement
                super().parse(file_path)
                return self._create_result("test", {})

        parser = ConcreteParser()
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        result = parser.parse(test_file)
        assert result["text"] == "test"


class TestMarkdownParser:
    """Tests for Markdown parsing."""

    def test_parse_simple_markdown(self, tmp_path: Path):
        """Test parsing a simple markdown file."""
        md_file = tmp_path / "test.md"
        md_file.write_text(
            "# Title\n\nThis is a paragraph.\n\n## Section 1\n\nContent here.\n"
        )

        parser = DocParser()
        result = parser.parse(str(md_file))

        assert "text" in result
        assert "metadata" in result
        assert "chunks" in result
        assert "Title" in result["text"]
        assert result["metadata"]["format"] == "MD"
        assert result["metadata"]["header_count"] == 2

    def test_parse_markdown_with_code_blocks(self, tmp_path: Path):
        """Test parsing markdown with code blocks."""
        md_file = tmp_path / "code.md"
        md_file.write_text(
            "# Code Example\n\n```python\nprint('hello')\n```\n\nMore text.\n"
        )

        parser = DocParser()
        result = parser.parse(md_file)

        assert result["metadata"]["code_blocks"] == 1
        assert "print('hello')" in result["text"]

    def test_chunks_by_section(self, tmp_path: Path):
        """Test that chunks are created by section."""
        md_file = tmp_path / "sections.md"
        md_file.write_text(
            "# Main\n\nIntro text.\n\n## First\n\nFirst section.\n\n## Second\n\nSecond section.\n"
        )

        parser = DocParser()
        result = parser.parse(md_file)

        assert len(result["chunks"]) == 3
        sections = [c["section"] for c in result["chunks"]]
        assert "Main" in sections


class TestPDFParser:
    """Tests for PDF parsing."""

    def test_parse_pdf_file_not_found(self):
        """Test error when PDF file doesn't exist."""
        parser = DocParser()
        with pytest.raises(FileNotFoundError):
            parser.parse("nonexistent.pdf")

    def test_parse_pdf_mock(self, tmp_path: Path):
        """Test PDF parsing with mocked PyMuPDF."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake pdf content")

        mock_fitz = MagicMock()
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=3)
        mock_doc.metadata = {
            "title": "Test Document",
            "author": "Test Author",
        }
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Page content"
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page, mock_page, mock_page]))
        mock_fitz.open.return_value = mock_doc

        with patch.dict(sys.modules, {"fitz": mock_fitz}):
            parser = DocParser()
            # Clear cached parser to force re-creation with mock
            parser._parsers.clear()
            result = parser.parse(str(pdf_file))

        assert result["metadata"]["pages"] == 3
        assert result["metadata"]["title"] == "Test Document"
        assert result["metadata"]["author"] == "Test Author"
        assert len(result["chunks"]) == 3


class TestWordParser:
    """Tests for Word document parsing."""

    def test_parse_word_mock(self, tmp_path: Path):
        """Test Word parsing with mocked python-docx."""
        docx_file = tmp_path / "test.docx"
        docx_file.write_bytes(b"fake docx content")

        mock_docx = MagicMock()
        mock_doc = MagicMock()
        mock_doc.paragraphs = [
            MagicMock(text="Paragraph 1", style=MagicMock(name="Normal")),
            MagicMock(text="Paragraph 2", style=MagicMock(name="Normal")),
        ]
        mock_doc.tables = []

        mock_core_props = MagicMock()
        mock_core_props.title = "Test Title"
        mock_core_props.author = "Test Author"
        mock_core_props.subject = ""
        mock_core_props.keywords = ""
        mock_core_props.created = None
        mock_core_props.modified = None
        mock_doc.core_properties = mock_core_props

        mock_docx.Document.return_value = mock_doc

        with patch.dict(sys.modules, {"docx": mock_docx}):
            parser = DocParser()
            parser._parsers.clear()
            result = parser.parse(str(docx_file))

        assert result["metadata"]["format"] == "DOCX"
        assert result["metadata"]["title"] == "Test Title"
        assert "Paragraph 1" in result["text"]

    def test_parse_word_with_tables(self, tmp_path: Path):
        """Test Word parsing with tables."""
        docx_file = tmp_path / "test.docx"
        docx_file.write_bytes(b"fake docx content")

        mock_docx = MagicMock()

        # Create mock paragraphs
        mock_para1 = MagicMock()
        mock_para1.text = "Paragraph 1"
        mock_para1.style = MagicMock(name="Normal")

        mock_para2 = MagicMock()
        mock_para2.text = ""  # Empty paragraph (should be skipped)
        mock_para2.style = MagicMock(name="Normal")

        mock_para3 = MagicMock()
        mock_para3.text = "   "  # Whitespace-only paragraph (should be skipped)
        mock_para3.style = MagicMock(name="Normal")

        # Create mock table
        mock_cell1 = MagicMock()
        mock_cell1.text = "Cell 1"
        mock_cell2 = MagicMock()
        mock_cell2.text = "Cell 2"
        mock_cell3 = MagicMock()
        mock_cell3.text = "Cell 3"
        mock_cell4 = MagicMock()
        mock_cell4.text = "Cell 4"

        mock_row1 = MagicMock()
        mock_row1.cells = [mock_cell1, mock_cell2]

        mock_row2 = MagicMock()
        mock_row2.cells = [mock_cell3, mock_cell4]

        mock_table = MagicMock()
        mock_table.rows = [mock_row1, mock_row2]

        mock_doc = MagicMock()
        mock_doc.paragraphs = [mock_para1, mock_para2, mock_para3]
        mock_doc.tables = [mock_table]

        mock_core_props = MagicMock()
        mock_core_props.title = ""
        mock_core_props.author = ""
        mock_core_props.subject = ""
        mock_core_props.keywords = ""
        mock_core_props.created = None
        mock_core_props.modified = None
        mock_doc.core_properties = mock_core_props

        mock_docx.Document.return_value = mock_doc

        with patch.dict(sys.modules, {"docx": mock_docx}):
            parser = DocParser()
            parser._parsers.clear()
            result = parser.parse(str(docx_file))

        assert result["metadata"]["format"] == "DOCX"
        assert "Paragraph 1" in result["text"]
        # Table content should be included
        assert "Cell 1" in result["text"]
        assert "Cell 2" in result["text"]
        # Check that we have table chunk
        table_chunks = [c for c in result["chunks"] if c.get("type") == "table"]
        assert len(table_chunks) == 1


class TestExcelParser:
    """Tests for Excel document parsing."""

    def test_parse_excel_mock(self, tmp_path: Path):
        """Test Excel parsing with mocked openpyxl."""
        xlsx_file = tmp_path / "test.xlsx"
        xlsx_file.write_bytes(b"fake xlsx content")

        mock_openpyxl = MagicMock()
        mock_ws = MagicMock()
        mock_ws.iter_rows.return_value = [
            (1, 2, 3),
            ("a", "b", "c"),
        ]
        mock_ws.max_row = 2
        mock_ws.max_column = 3

        mock_wb = MagicMock()
        mock_wb.sheetnames = ["Sheet1", "Sheet2"]
        mock_wb.__getitem__ = MagicMock(return_value=mock_ws)
        mock_wb.close = MagicMock()
        mock_openpyxl.load_workbook.return_value = mock_wb

        with patch.dict(sys.modules, {"openpyxl": mock_openpyxl}):
            parser = DocParser()
            parser._parsers.clear()
            result = parser.parse(str(xlsx_file))

        assert result["metadata"]["format"] == "XLSX"
        assert result["metadata"]["sheets"] == 2
        assert result["metadata"]["sheet_names"] == ["Sheet1", "Sheet2"]

    def test_parse_excel_with_empty_sheet(self, tmp_path: Path):
        """Test Excel parsing with empty sheet content."""
        xlsx_file = tmp_path / "test.xlsx"
        xlsx_file.write_bytes(b"fake xlsx content")

        mock_openpyxl = MagicMock()

        # Empty sheet
        mock_ws_empty = MagicMock()
        mock_ws_empty.iter_rows.return_value = []  # No rows
        mock_ws_empty.max_row = 0
        mock_ws_empty.max_column = 0

        # Sheet with all empty/None values
        mock_ws_none_values = MagicMock()
        mock_ws_none_values.iter_rows.return_value = [
            (None, None, None),
            (None, None, None),
        ]
        mock_ws_none_values.max_row = 2
        mock_ws_none_values.max_column = 3

        mock_wb = MagicMock()
        mock_wb.sheetnames = ["EmptySheet", "NoneValuesSheet"]
        mock_wb.__getitem__ = MagicMock(side_effect=lambda name:
            mock_ws_empty if name == "EmptySheet" else mock_ws_none_values)
        mock_wb.close = MagicMock()
        mock_openpyxl.load_workbook.return_value = mock_wb

        with patch.dict(sys.modules, {"openpyxl": mock_openpyxl}):
            parser = DocParser()
            parser._parsers.clear()
            result = parser.parse(str(xlsx_file))

        assert result["metadata"]["format"] == "XLSX"
        # Empty sheets should not add chunks
        assert len(result["chunks"]) == 0


class TestPPTParser:
    """Tests for PowerPoint document parsing."""

    def test_parse_ppt_mock(self, tmp_path: Path):
        """Test PowerPoint parsing with mocked python-pptx."""
        pptx_file = tmp_path / "test.pptx"
        pptx_file.write_bytes(b"fake pptx content")

        mock_pptx = MagicMock()
        mock_shape = MagicMock()
        mock_shape.text = "Slide content"

        mock_slide = MagicMock()
        mock_slide.shapes = [mock_shape]

        mock_prs = MagicMock()
        mock_prs.slides = [mock_slide, mock_slide]

        mock_core_props = MagicMock()
        mock_core_props.title = "Presentation Title"
        mock_core_props.author = "Presenter"
        mock_core_props.subject = ""
        mock_core_props.created = None
        mock_core_props.modified = None
        mock_prs.core_properties = mock_core_props

        mock_pptx.Presentation.return_value = mock_prs

        with patch.dict(sys.modules, {"pptx": mock_pptx}):
            parser = DocParser()
            parser._parsers.clear()
            result = parser.parse(str(pptx_file))

        assert result["metadata"]["format"] == "PPTX"
        assert result["metadata"]["slides"] == 2
        assert "Slide content" in result["text"]

    def test_parse_ppt_empty_slide(self, tmp_path: Path):
        """Test PowerPoint parsing with empty slides."""
        pptx_file = tmp_path / "test.pptx"
        pptx_file.write_bytes(b"fake pptx content")

        mock_pptx = MagicMock()

        # Shape with no text attribute
        mock_shape_no_text = MagicMock(spec=[])  # No 'text' attribute

        # Shape with empty text
        mock_shape_empty = MagicMock()
        mock_shape_empty.text = ""

        # Shape with whitespace text
        mock_shape_whitespace = MagicMock()
        mock_shape_whitespace.text = "   "

        # Shape with actual content
        mock_shape_content = MagicMock()
        mock_shape_content.text = "Real content"

        mock_slide_empty = MagicMock()
        mock_slide_empty.shapes = [mock_shape_no_text, mock_shape_empty, mock_shape_whitespace]

        mock_slide_with_content = MagicMock()
        mock_slide_with_content.shapes = [mock_shape_content]

        mock_prs = MagicMock()
        mock_prs.slides = [mock_slide_empty, mock_slide_with_content]

        mock_core_props = MagicMock()
        mock_core_props.title = ""
        mock_core_props.author = ""
        mock_core_props.subject = ""
        mock_core_props.created = None
        mock_core_props.modified = None
        mock_prs.core_properties = mock_core_props

        mock_pptx.Presentation.return_value = mock_prs

        with patch.dict(sys.modules, {"pptx": mock_pptx}):
            parser = DocParser()
            parser._parsers.clear()
            result = parser.parse(str(pptx_file))

        assert result["metadata"]["format"] == "PPTX"
        # Only one chunk should be created (for the slide with content)
        assert len(result["chunks"]) == 1
        assert "Real content" in result["text"]


class TestDocParserInterface:
    """Tests for DocParser public interface."""

    def test_supported_formats(self):
        """Test that supported formats are listed."""
        parser = DocParser()
        formats = parser.supported_formats

        assert ".pdf" in formats
        assert ".docx" in formats
        assert ".xlsx" in formats
        assert ".pptx" in formats
        assert ".md" in formats

    def test_supports_method(self, tmp_path: Path):
        """Test the supports() method."""
        parser = DocParser()

        assert parser.supports("test.pdf") is True
        assert parser.supports("test.docx") is True
        assert parser.supports("test.txt") is False
        assert parser.supports("test.xyz") is False

    def test_parse_nonexistent_file(self):
        """Test error for non-existent file."""
        parser = DocParser()
        with pytest.raises(FileNotFoundError):
            parser.parse("nonexistent.pdf")

    def test_parse_unsupported_format(self, tmp_path: Path):
        """Test error for unsupported file format."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("plain text")

        parser = DocParser()
        with pytest.raises(ValueError, match="Unsupported file format"):
            parser.parse(str(txt_file))

    def test_parse_directory_raises_error(self, tmp_path: Path):
        """Test that parsing a directory raises error."""
        parser = DocParser()
        with pytest.raises(ValueError, match="Not a file"):
            parser.parse(str(tmp_path))

    def test_result_structure(self, tmp_path: Path):
        """Test that result has expected structure."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Test\n\nContent.")

        parser = DocParser()
        result = parser.parse(str(md_file))

        assert "text" in result
        assert "metadata" in result
        assert "chunks" in result
        assert isinstance(result["text"], str)
        assert isinstance(result["metadata"], dict)
        assert isinstance(result["chunks"], list)

    def test_metadata_includes_file_info(self, tmp_path: Path):
        """Test that metadata includes file information."""
        md_file = tmp_path / "document.md"
        md_file.write_text("# Test")

        parser = DocParser()
        result = parser.parse(str(md_file))

        assert "file_path" in result["metadata"]
        assert "file_name" in result["metadata"]
        assert result["metadata"]["file_name"] == "document.md"


class TestImportErrors:
    """Test handling of missing dependencies."""

    def test_pdf_missing_pymupdf(self, tmp_path: Path):
        """Test helpful error when PyMuPDF is not installed."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake pdf")

        # Create a mock that raises ImportError when open is called
        mock_fitz = MagicMock()
        mock_fitz.open.side_effect = ModuleNotFoundError("No module named 'fitz'")

        parser = DocParser()
        parser._parsers.clear()

        with patch.dict(sys.modules, {"fitz": mock_fitz}):
            with pytest.raises((ImportError, ModuleNotFoundError)):
                parser.parse(str(pdf_file))

    def test_pdf_import_error(self, tmp_path: Path):
        """Test ImportError is raised when fitz module is not available."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake pdf")

        # Use PDFParser directly to test the import error path
        parser = PDFParser()

        # Mock the import to raise ImportError
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "fitz":
                raise ImportError("No module named 'fitz'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="PyMuPDF is required"):
                parser.parse(pdf_file)

    def test_word_import_error(self, tmp_path: Path):
        """Test ImportError is raised when docx module is not available."""
        docx_file = tmp_path / "test.docx"
        docx_file.write_bytes(b"fake docx")

        parser = WordParser()

        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "docx":
                raise ImportError("No module named 'docx'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="python-docx is required"):
                parser.parse(docx_file)

    def test_excel_import_error(self, tmp_path: Path):
        """Test ImportError is raised when openpyxl module is not available."""
        xlsx_file = tmp_path / "test.xlsx"
        xlsx_file.write_bytes(b"fake xlsx")

        parser = ExcelParser()

        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "openpyxl":
                raise ImportError("No module named 'openpyxl'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="openpyxl is required"):
                parser.parse(xlsx_file)

    def test_ppt_import_error(self, tmp_path: Path):
        """Test ImportError is raised when pptx module is not available."""
        pptx_file = tmp_path / "test.pptx"
        pptx_file.write_bytes(b"fake pptx")

        parser = PPTParser()

        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pptx":
                raise ImportError("No module named 'pptx'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="python-pptx is required"):
                parser.parse(pptx_file)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_markdown(self, tmp_path: Path):
        """Test parsing empty markdown file."""
        md_file = tmp_path / "empty.md"
        md_file.write_text("")

        parser = DocParser()
        result = parser.parse(str(md_file))

        assert result["text"] == ""
        assert result["metadata"]["header_count"] == 0

    def test_markdown_unicode(self, tmp_path: Path):
        """Test parsing markdown with unicode characters."""
        md_file = tmp_path / "unicode.md"
        md_file.write_text("# 中文标题\n\n内容测试 🎉", encoding="utf-8")

        parser = DocParser()
        result = parser.parse(str(md_file))

        assert "中文标题" in result["text"]
        assert "🎉" in result["text"]

    def test_path_object_input(self, tmp_path: Path):
        """Test that Path objects are accepted."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Test")

        parser = DocParser()
        result = parser.parse(md_file)

        assert "Test" in result["text"]