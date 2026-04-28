from __future__ import annotations

import os
from pathlib import Path

import markdown


class Reconstructor:
    def markdown_to_html(self, markdown_text: str, title: str | None = None, output_mode: str = "readable") -> str:
        body_html = markdown.markdown(
            markdown_text,
            extensions=["tables", "fenced_code", "md_in_html"],
        )
        title_text = title or "Translated Document"
        mode_css = self._mode_css(output_mode)
        return f"""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>{title_text}</title>
  <style>
    :root {{
      --font-body: "Noto Serif", "Times New Roman", serif;
      --font-head: "Noto Sans", "Helvetica", sans-serif;
    }}
    @page {{
      size: A4;
      margin: 18mm 16mm 18mm 16mm;
      @bottom-center {{ content: counter(page); font-size: 10pt; color: #444; }}
    }}
    body {{ font-family: var(--font-body); line-height: 1.45; color: #111; font-size: 11pt; }}
    main {{ {mode_css} }}
    h1,h2,h3,h4,h5 {{ font-family: var(--font-head); margin-top: 1.1em; margin-bottom: .4em; }}
    h1,h2 {{ break-after: avoid; column-span: all; }}
    h3,h4,h5 {{ break-after: avoid; }}
    p {{ margin: .35em 0; text-align: justify; }}
    table {{ width: 100%; border-collapse: collapse; margin: .8em 0; font-size: 9.5pt; break-inside: auto; column-span: all; }}
    thead {{ display: table-header-group; }}
    tr {{ break-inside: avoid; }}
    th,td {{ border: 1px solid #777; padding: 4px 6px; vertical-align: top; }}
    th {{ background: #f0f0f0; }}
    .table-block {{ column-span: all; border: 1px solid #777; padding: 6px 8px; margin: .8em 0; font-size: 9.5pt; white-space: pre-wrap; break-inside: avoid; }}
    img {{ max-width: 100%; page-break-inside: avoid; break-inside: avoid; }}
    figure, blockquote, pre {{ break-inside: avoid; }}
    em {{ color: #2d2d2d; }}
    small {{ color: #666; font-size: 9pt; }}
    .page-marker {{ column-span: all; break-before: page; height: 0; overflow: hidden; }}
  </style>
</head>
<body>
<main>
{body_html}
</main>
</body>
</html>
"""

    def _mode_css(self, output_mode: str) -> str:
        if output_mode == "faithful":
            return "column-count: 2; column-gap: 8mm;"
        if output_mode == "debug_markdown":
            return "max-width: 185mm; margin: 0 auto; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;"
        if output_mode == "readable":
            return "max-width: 172mm; margin: 0 auto;"
        return "max-width: 172mm; margin: 0 auto;"

    def html_to_pdf(self, html_text: str, pdf_path: Path) -> None:
        self._configure_macos_homebrew_libs()
        from weasyprint import CSS, HTML

        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        HTML(string=html_text).write_pdf(str(pdf_path), stylesheets=[CSS(string="")])

    def _configure_macos_homebrew_libs(self) -> None:
        homebrew_lib = Path("/opt/homebrew/lib")
        if homebrew_lib.exists():
            current = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
            paths = [p for p in current.split(":") if p]
            if str(homebrew_lib) not in paths:
                os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join([str(homebrew_lib), *paths])
