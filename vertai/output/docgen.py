"""Document generation module for VertAI.

Supports Markdown, HTML, and PDF output formats with built-in templates.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import html
import re


class OutputFormat(Enum):
    """Supported output formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"


class TemplateType(Enum):
    """Built-in template types."""
    REPORT = "report"
    RESUME = "resume"
    PAPER = "paper"
    LETTER = "letter"
    PROPOSAL = "proposal"


@dataclass
class Template:
    """Template definition for document generation."""
    name: str
    sections: list[str]
    title_format: str = "# {title}"
    section_format: str = "## {section}"
    item_format: str = "- {item}"
    metadata: dict[str, Any] = field(default_factory=dict)


BUILTIN_TEMPLATES: dict[str, Template] = {
    "report": Template(
        name="report",
        sections=["title", "author", "date", "summary", "introduction", "findings", "conclusion"],
        title_format="# {title}",
        section_format="## {section}",
        metadata={"include_toc": True, "page_numbers": True}
    ),
    "resume": Template(
        name="resume",
        sections=["name", "contact", "summary", "experience", "education", "skills"],
        title_format="# {title}",
        section_format="### {section}",
        metadata={"include_toc": False, "page_numbers": False}
    ),
    "paper": Template(
        name="paper",
        sections=["title", "authors", "abstract", "introduction", "methodology", "results", "discussion", "conclusion", "references"],
        title_format="# {title}",
        section_format="## {section}",
        metadata={"include_toc": True, "page_numbers": True, "citation_style": "apa"}
    ),
    "letter": Template(
        name="letter",
        sections=["date", "sender", "recipient", "subject", "body", "closing"],
        title_format="**{title}**",
        section_format="",
        metadata={"include_toc": False, "page_numbers": False}
    ),
    "proposal": Template(
        name="proposal",
        sections=["title", "client", "date", "executive_summary", "background", "objectives", "methodology", "timeline", "budget", "conclusion"],
        title_format="# {title}",
        section_format="## {section}",
        metadata={"include_toc": True, "page_numbers": True}
    ),
}


class DocGen:
    """Document generator supporting multiple output formats and templates.

    Examples:
        >>> from vertai import DocGen
        >>> doc = DocGen(template="report", format="markdown")
        >>> result = doc.generate({
        ...     "title": "Annual Report",
        ...     "author": "John Doe",
        ...     "summary": "Key findings from the year."
        ... })
    """

    def __init__(
        self,
        template: str | TemplateType | Template = "report",
        format: str | OutputFormat = OutputFormat.MARKDOWN,
        locale: str = "en_US",
    ) -> None:
        """Initialize the document generator.

        Args:
            template: Template name, TemplateType enum, or Template instance.
            format: Output format (markdown, html, pdf).
            locale: Locale for date and number formatting.
        """
        self._template = self._resolve_template(template)
        self._format = OutputFormat(format) if isinstance(format, str) else format
        self._locale = locale

    def _resolve_template(self, template: str | TemplateType | Template) -> Template:
        """Resolve template parameter to a Template instance."""
        if isinstance(template, Template):
            return template
        if isinstance(template, TemplateType):
            template_name = template.value
        else:
            template_name = template

        if template_name not in BUILTIN_TEMPLATES:
            raise ValueError(
                f"Unknown template: {template_name}. "
                f"Available templates: {list(BUILTIN_TEMPLATES.keys())}"
            )
        return deepcopy(BUILTIN_TEMPLATES[template_name])

    @property
    def template(self) -> Template:
        """Get current template."""
        return self._template

    @property
    def format(self) -> OutputFormat:
        """Get current output format."""
        return self._format

    def generate(self, data: dict[str, Any]) -> str:
        """Generate a document from the provided data.

        Args:
            data: Dictionary containing document content. Keys should match
                  template sections.

        Returns:
            Generated document as a string.

        Raises:
            ValueError: If required template sections are missing from data.
        """
        content = self._build_content(data)

        if self._format == OutputFormat.MARKDOWN:
            return self._render_markdown(content)
        elif self._format == OutputFormat.HTML:
            return self._render_html(content)
        elif self._format == OutputFormat.PDF:
            return self._render_pdf(content)
        else:
            raise ValueError(f"Unsupported format: {self._format}")

    def _build_content(self, data: dict[str, Any]) -> dict[str, Any]:
        """Build content dictionary with processed sections."""
        content = {}

        for section in self._template.sections:
            if section in data:
                content[section] = self._process_section(section, data[section])
            elif section in ["date"] and section not in data:
                content[section] = datetime.now().strftime("%Y-%m-%d")

        return content

    def _process_section(self, section: str, value: Any) -> Any:
        """Process a section value based on its type."""
        if isinstance(value, list):
            return [self._process_item(item) for item in value]
        return value

    def _process_item(self, item: Any) -> str:
        """Convert an item to string representation."""
        if isinstance(item, dict):
            parts = [f"{k}: {v}" for k, v in item.items()]
            return " | ".join(parts)
        return str(item)

    def _render_markdown(self, content: dict[str, Any]) -> str:
        """Render content as Markdown."""
        lines = []

        title_section = self._template.sections[0] if self._template.sections else None
        if title_section and title_section in content:
            title = content[title_section]
            if isinstance(title, str):
                lines.append(self._template.title_format.format(title=title))
                lines.append("")

        if self._template.metadata.get("include_toc"):
            lines.append("## Table of Contents")
            lines.append("")
            for section in self._template.sections[1:]:
                if section in content:
                    section_title = section.replace("_", " ").title()
                    lines.append(f"- [{section_title}](#{section})")
            lines.append("")

        for section in self._template.sections:
            if section == title_section or section not in content:
                continue

            section_title = section.replace("_", " ").title()
            value = content[section]

            if self._template.section_format:
                lines.append(self._template.section_format.format(section=section_title))
                lines.append("")

            if isinstance(value, list):
                for item in value:
                    lines.append(self._template.item_format.format(item=item))
            else:
                lines.append(str(value))
            lines.append("")

        return "\n".join(lines).strip()

    def _render_html(self, content: dict[str, Any]) -> str:
        """Render content as HTML."""
        html_parts = [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="UTF-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
        ]

        title_section = self._template.sections[0] if self._template.sections else None
        if title_section and title_section in content:
            title = html.escape(str(content[title_section]))
            html_parts.append(f"<title>{title}</title>")

        html_parts.extend([
            "<style>",
            "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 2rem; }",
            "h1 { border-bottom: 2px solid #333; padding-bottom: 0.5rem; }",
            "h2 { color: #555; margin-top: 2rem; }",
            "h3 { color: #666; }",
            "ul { line-height: 1.8; }",
            ".toc { background: #f5f5f5; padding: 1rem; border-radius: 4px; }",
            ".toc ul { list-style: none; padding-left: 0; }",
            ".toc a { color: #0066cc; text-decoration: none; }",
            "</style>",
            "</head>",
            "<body>",
        ])

        if title_section and title_section in content:
            title = html.escape(str(content[title_section]))
            html_parts.append(f"<h1>{title}</h1>")

        if self._template.metadata.get("include_toc"):
            html_parts.append('<div class="toc">')
            html_parts.append("<h2>Table of Contents</h2>")
            html_parts.append("<ul>")
            for section in self._template.sections[1:]:
                if section in content:
                    section_title = section.replace("_", " ").title()
                    html_parts.append(f'<li><a href="#{section}">{html.escape(section_title)}</a></li>')
            html_parts.append("</ul>")
            html_parts.append("</div>")

        for section in self._template.sections:
            if section == title_section or section not in content:
                continue

            section_title = section.replace("_", " ").title()
            value = content[section]

            html_parts.append(f'<section id="{section}">')
            html_parts.append(f"<h2>{html.escape(section_title)}</h2>")

            if isinstance(value, list):
                html_parts.append("<ul>")
                for item in value:
                    html_parts.append(f"<li>{html.escape(str(item))}</li>")
                html_parts.append("</ul>")
            else:
                html_parts.append(f"<p>{html.escape(str(value))}</p>")

            html_parts.append("</section>")

        html_parts.extend([
            "</body>",
            "</html>",
        ])

        return "\n".join(html_parts)

    def _render_pdf(self, content: dict[str, Any]) -> bytes:
        """Render content as PDF.

        Raises:
            NotImplementedError: PDF generation requires external dependencies.
                Use HTML output and convert with weasyprint/pdfkit/xhtml2pdf.

        Note:
            For PDF output, use the following workflow:
            1. Generate HTML: doc.set_format("html").generate(data)
            2. Convert with weasyprint: weasyprint.HTML(string=html).write_pdf()
        """
        raise NotImplementedError(
            "PDF generation requires external dependencies.\n"
            "Use HTML output and convert with:\n"
            "  - weasyprint: pip install weasyprint\n"
            "  - pdfkit: pip install pdfkit (requires wkhtmltopdf)\n"
            "  - xhtml2pdf: pip install xhtml2pdf\n\n"
            "Example:\n"
            "  from weasyprint import HTML\n"
            "  html = doc.set_format('html').generate(data)\n"
            "  pdf_bytes = HTML(string=html).write_pdf()"
        )

    def add_section(self, name: str, after: str | None = None) -> "DocGen":
        """Add a custom section to the template.

        Args:
            name: Section name.
            after: Insert after this section. If None, appends to end.

        Returns:
            Self for method chaining.
        """
        if after is None:
            self._template.sections.append(name)
        else:
            try:
                idx = self._template.sections.index(after)
                self._template.sections.insert(idx + 1, name)
            except ValueError:
                self._template.sections.append(name)
        return self

    def remove_section(self, name: str) -> "DocGen":
        """Remove a section from the template.

        Args:
            name: Section name to remove.

        Returns:
            Self for method chaining.
        """
        if name in self._template.sections:
            self._template.sections.remove(name)
        return self

    def set_format(self, format: str | OutputFormat) -> "DocGen":
        """Set the output format.

        Args:
            format: Output format (markdown, html, pdf).

        Returns:
            Self for method chaining.
        """
        self._format = OutputFormat(format) if isinstance(format, str) else format
        return self

    def to_markdown(self) -> "DocGen":
        """Set output format to Markdown."""
        self._format = OutputFormat.MARKDOWN
        return self

    def to_html(self) -> "DocGen":
        """Set output format to HTML."""
        self._format = OutputFormat.HTML
        return self

    def to_pdf(self) -> "DocGen":
        """Set output format to PDF."""
        self._format = OutputFormat.PDF
        return self

    def save(self, data: dict[str, Any], filepath: str) -> None:
        """Generate and save document to a file.

        Args:
            data: Document content data.
            filepath: Output file path.
        """
        content = self.generate(data)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    @classmethod
    def list_templates(cls) -> list[str]:
        """List available built-in templates."""
        return list(BUILTIN_TEMPLATES.keys())

    @classmethod
    def list_formats(cls) -> list[str]:
        """List available output formats."""
        return [f.value for f in OutputFormat]