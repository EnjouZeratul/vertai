"""Unit tests for DocGen module."""

import pytest
from datetime import datetime

from vertai.output.docgen import (
    DocGen,
    OutputFormat,
    TemplateType,
    Template,
)


class TestOutputFormat:
    """Tests for OutputFormat enum."""

    def test_output_formats_exist(self):
        assert OutputFormat.MARKDOWN.value == "markdown"
        assert OutputFormat.HTML.value == "html"
        assert OutputFormat.PDF.value == "pdf"


class TestTemplateType:
    """Tests for TemplateType enum."""

    def test_template_types_exist(self):
        assert TemplateType.REPORT.value == "report"
        assert TemplateType.RESUME.value == "resume"
        assert TemplateType.PAPER.value == "paper"
        assert TemplateType.LETTER.value == "letter"
        assert TemplateType.PROPOSAL.value == "proposal"


class TestTemplate:
    """Tests for Template dataclass."""

    def test_template_creation(self):
        template = Template(
            name="custom",
            sections=["title", "body"],
            title_format="# {title}",
        )
        assert template.name == "custom"
        assert template.sections == ["title", "body"]
        assert template.title_format == "# {title}"

    def test_template_default_metadata(self):
        template = Template(name="test", sections=[])
        assert template.metadata == {}


class TestDocGenInit:
    """Tests for DocGen initialization."""

    def test_init_with_string_template(self):
        doc = DocGen(template="report")
        assert doc.template.name == "report"

    def test_init_with_enum_template(self):
        doc = DocGen(template=TemplateType.RESUME)
        assert doc.template.name == "resume"

    def test_init_with_template_instance(self):
        custom = Template(name="custom", sections=["title"])
        doc = DocGen(template=custom)
        assert doc.template.name == "custom"

    def test_init_with_string_format(self):
        doc = DocGen(format="html")
        assert doc.format == OutputFormat.HTML

    def test_init_with_enum_format(self):
        doc = DocGen(format=OutputFormat.PDF)
        assert doc.format == OutputFormat.PDF

    def test_init_default_format(self):
        doc = DocGen()
        assert doc.format == OutputFormat.MARKDOWN

    def test_init_invalid_template_raises(self):
        with pytest.raises(ValueError, match="Unknown template"):
            DocGen(template="nonexistent")


class TestDocGenGenerate:
    """Tests for DocGen.generate method."""

    def test_generate_markdown_report(self):
        doc = DocGen(template="report", format="markdown")
        data = {
            "title": "Test Report",
            "author": "John Doe",
            "summary": "This is a test summary.",
        }
        result = doc.generate(data)

        assert "# Test Report" in result
        assert "John Doe" in result
        assert "This is a test summary." in result

    def test_generate_markdown_with_list(self):
        doc = DocGen(template="report", format="markdown")
        data = {
            "title": "Test",
            "findings": ["Finding 1", "Finding 2", "Finding 3"],
        }
        result = doc.generate(data)

        assert "- Finding 1" in result
        assert "- Finding 2" in result
        assert "- Finding 3" in result

    def test_generate_markdown_with_dict_items(self):
        """Test generating with dict items in list (uses _process_item)."""
        doc = DocGen(template="report", format="markdown")
        data = {
            "title": "Test",
            "findings": [
                {"name": "Finding A", "value": "Important"},
                {"key": "Finding B", "score": 90},
            ],
        }
        result = doc.generate(data)

        # Dict items should be formatted as "key: value | key: value"
        assert "name: Important" in result or "name: Finding A" in result
        assert "value: Important" in result

    def test_generate_html_report(self):
        doc = DocGen(template="report", format="html")
        data = {
            "title": "Test Report",
            "author": "Jane Doe",
        }
        result = doc.generate(data)

        assert "<!DOCTYPE html>" in result
        assert "<title>Test Report</title>" in result
        assert "<h1>Test Report</h1>" in result

    def test_generate_html_with_list(self):
        """Test HTML generation with list items."""
        doc = DocGen(template="report", format="html")
        data = {
            "title": "Test",
            "findings": ["Finding 1", "Finding 2"],
        }
        result = doc.generate(data)

        assert "<ul>" in result
        assert "<li>Finding 1</li>" in result
        assert "<li>Finding 2</li>" in result
        assert "</ul>" in result

    def test_generate_html_escapes_content(self):
        doc = DocGen(template="report", format="html")
        data = {
            "title": "<script>alert('xss')</script>",
        }
        result = doc.generate(data)

        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_generate_pdf_raises_not_implemented(self):
        doc = DocGen(template="report", format="pdf")
        data = {"title": "Test"}

        with pytest.raises(NotImplementedError, match="PDF generation requires"):
            doc.generate(data)

    def test_generate_unsupported_format_raises(self):
        """Test generate with unsupported format raises ValueError."""
        doc = DocGen(template="report", format=OutputFormat.MARKDOWN)

        # Manually set an invalid format to trigger the error
        doc._format = "invalid_format"  # Bypass enum validation

        data = {"title": "Test"}

        with pytest.raises(ValueError, match="Unsupported format"):
            doc.generate(data)

    def test_generate_with_auto_date(self):
        doc = DocGen(template="report")
        data = {"title": "Test"}
        result = doc.generate(data)

        today = datetime.now().strftime("%Y-%m-%d")
        assert today in result

    def test_generate_with_custom_date(self):
        doc = DocGen(template="report")
        data = {
            "title": "Test",
            "date": "2024-01-15",
        }
        result = doc.generate(data)

        assert "2024-01-15" in result


class TestDocGenTemplates:
    """Tests for built-in templates."""

    def test_resume_template(self):
        doc = DocGen(template="resume")
        data = {
            "name": "John Doe",
            "contact": "john@example.com",
            "summary": "Experienced developer",
        }
        result = doc.generate(data)

        assert "# John Doe" in result
        assert "john@example.com" in result

    def test_paper_template(self):
        doc = DocGen(template="paper")
        data = {
            "title": "Research Paper",
            "authors": ["Author 1", "Author 2"],
            "abstract": "This is the abstract.",
        }
        result = doc.generate(data)

        assert "# Research Paper" in result
        assert "Abstract" in result

    def test_letter_template(self):
        doc = DocGen(template="letter")
        data = {
            "sender": "John Doe",
            "recipient": "Jane Smith",
            "body": "Dear Jane, ...",
        }
        result = doc.generate(data)

        assert "John Doe" in result
        assert "Jane Smith" in result

    def test_proposal_template(self):
        doc = DocGen(template="proposal")
        data = {
            "title": "Project Proposal",
            "client": "Acme Corp",
        }
        result = doc.generate(data)

        assert "# Project Proposal" in result
        assert "Acme Corp" in result


class TestDocGenMethods:
    """Tests for DocGen helper methods."""

    def test_add_section(self):
        doc = DocGen(template="report")
        doc.add_section("custom_section")

        assert "custom_section" in doc.template.sections

    def test_add_section_after(self):
        doc = DocGen(template="report")
        doc.add_section("new_section", after="title")

        idx = doc.template.sections.index("new_section")
        title_idx = doc.template.sections.index("title")
        assert idx == title_idx + 1

    def test_add_section_after_nonexistent(self):
        """Test add_section with non-existent 'after' section appends to end."""
        doc = DocGen(template="report")
        original_len = len(doc.template.sections)

        # Adding section after non-existent section should append
        doc.add_section("new_section", after="nonexistent_section")

        assert "new_section" in doc.template.sections
        assert doc.template.sections[-1] == "new_section"
        assert len(doc.template.sections) == original_len + 1

    def test_remove_section(self):
        doc = DocGen(template="report")
        original_len = len(doc.template.sections)
        doc.remove_section("summary")

        assert "summary" not in doc.template.sections
        assert len(doc.template.sections) == original_len - 1

    def test_set_format(self):
        doc = DocGen()
        doc.set_format("html")

        assert doc.format == OutputFormat.HTML

    def test_to_markdown(self):
        doc = DocGen(format="html")
        result = doc.to_markdown()

        assert result is doc
        assert doc.format == OutputFormat.MARKDOWN

    def test_to_html(self):
        doc = DocGen()
        result = doc.to_html()

        assert result is doc
        assert doc.format == OutputFormat.HTML

    def test_to_pdf(self):
        doc = DocGen()
        result = doc.to_pdf()

        assert result is doc
        assert doc.format == OutputFormat.PDF

    def test_method_chaining(self):
        doc = DocGen()
        result = doc.to_html().add_section("custom").remove_section("summary")

        assert result is doc


class TestDocGenSave:
    """Tests for DocGen.save method."""

    def test_save_markdown(self, tmp_path):
        doc = DocGen(template="report", format="markdown")
        data = {"title": "Saved Document"}
        filepath = str(tmp_path / "output.md")

        doc.save(data, filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        assert "# Saved Document" in content

    def test_save_html(self, tmp_path):
        doc = DocGen(template="report", format="html")
        data = {"title": "Saved Document"}
        filepath = str(tmp_path / "output.html")

        doc.save(data, filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        assert "<!DOCTYPE html>" in content


class TestDocGenClassMethods:
    """Tests for DocGen class methods."""

    def test_list_templates(self):
        templates = DocGen.list_templates()

        assert "report" in templates
        assert "resume" in templates
        assert "paper" in templates
        assert "letter" in templates
        assert "proposal" in templates

    def test_list_formats(self):
        formats = DocGen.list_formats()

        assert "markdown" in formats
        assert "html" in formats
        assert "pdf" in formats


class TestDocGenIntegration:
    """Integration tests for DocGen."""

    def test_full_report_workflow(self):
        doc = DocGen(template="report", format="markdown")
        data = {
            "title": "Q4 2024 Report",
            "author": "Analytics Team",
            "summary": "Quarterly performance review.",
            "introduction": "This report covers Q4 2024.",
            "findings": [
                "Revenue increased by 15%",
                "Customer satisfaction improved",
                "New markets entered",
            ],
            "conclusion": "Strong quarter overall.",
        }
        result = doc.generate(data)

        assert "# Q4 2024 Report" in result
        assert "Analytics Team" in result
        assert "Quarterly performance review" in result
        assert "- Revenue increased by 15%" in result

    def test_resume_workflow(self):
        doc = DocGen(template="resume")
        data = {
            "name": "John Smith",
            "contact": "john.smith@email.com | (555) 123-4567",
            "summary": "Senior Software Engineer with 10 years experience",
            "experience": [
                "Senior Engineer at TechCorp (2020-Present)",
                "Engineer at StartupInc (2015-2020)",
            ],
            "education": ["BS Computer Science, MIT"],
            "skills": ["Python", "JavaScript", "SQL", "AWS"],
        }
        result = doc.generate(data)

        assert "# John Smith" in result
        assert "Senior Software Engineer" in result
        assert "- Python" in result

    def test_html_report_has_toc(self):
        doc = DocGen(template="report", format="html")
        data = {
            "title": "Report with TOC",
            "introduction": "Intro",
            "findings": "Findings",
            "conclusion": "Conclusion",
        }
        result = doc.generate(data)

        assert "Table of Contents" in result
        assert 'href="#introduction"' in result
        assert 'href="#findings"' in result

    def test_resume_no_toc(self):
        doc = DocGen(template="resume", format="html")
        data = {
            "name": "Test User",
            "summary": "Test",
        }
        result = doc.generate(data)

        assert "Table of Contents" not in result
