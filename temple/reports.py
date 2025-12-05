"""
Temple Reports - Shareable Diagnostic Reports
===============================================

Generate beautiful, shareable reports in multiple formats:
- HTML (rich, interactive, single-file - MOST SHAREABLE)
- Markdown (GitHub/Discord friendly)
- JSON (machine readable)

Report Types:
- Problem Report: "Help me debug this"
- Achievement Report: "Look what happened!"
- Campaign Summary: Full training run analysis
- Checkpoint Analysis: Specific checkpoint deep-dive

The key insight: People share reports when:
1. Things go WRONG (need help)
2. Things go RIGHT (want to show off)
3. Working with teams (need to communicate)

Usage:
    from temple.reports import ReportGenerator, ReportType

    # Generate a problem report
    report = ReportGenerator.problem_report(
        diagnoses=diagnoses,
        config=config,
        title="NaN Loss Help Needed",
    )

    # Export to different formats
    report.to_html("problem_report.html")
    report.to_markdown("problem_report.md")
    report.to_json("problem_report.json")

    # Or get strings directly
    html_str = report.html
    md_str = report.markdown
"""

from __future__ import annotations

import json
import html as html_lib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from temple.diagnostics.severity import Diagnosis, DiagnosticSeverity, DiagnosisCategory


class ReportType(Enum):
    """Type of report to generate."""
    PROBLEM = "problem"           # Help me debug this
    ACHIEVEMENT = "achievement"   # Look what happened!
    CAMPAIGN = "campaign"         # Full training summary
    CHECKPOINT = "checkpoint"     # Specific checkpoint analysis
    PREFLIGHT = "preflight"       # Pre-training checks
    HEALTH = "health"             # Current health status


# =============================================================================
# HTML TEMPLATES
# =============================================================================

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --text-muted: #6e7681;
            --border-color: #30363d;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-yellow: #d29922;
            --accent-orange: #db6d28;
            --accent-red: #f85149;
            --accent-purple: #a371f7;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
            max-width: 900px;
            margin: 0 auto;
        }}

        h1, h2, h3 {{
            color: var(--text-primary);
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.3em;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }}

        h1 {{ font-size: 1.8em; }}
        h2 {{ font-size: 1.4em; }}
        h3 {{ font-size: 1.1em; border-bottom: none; }}

        .header {{
            background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 20px;
        }}

        .header h1 {{
            margin: 0 0 10px 0;
            border: none;
            padding: 0;
        }}

        .header .meta {{
            color: var(--text-secondary);
            font-size: 0.9em;
        }}

        .header .meta span {{
            margin-right: 15px;
        }}

        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: 600;
            text-transform: uppercase;
        }}

        .badge-critical {{ background: var(--accent-red); color: white; }}
        .badge-error {{ background: var(--accent-orange); color: white; }}
        .badge-warn {{ background: var(--accent-yellow); color: black; }}
        .badge-info {{ background: var(--accent-blue); color: white; }}
        .badge-ok {{ background: var(--accent-green); color: white; }}

        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}

        .card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 15px;
        }}

        .card-title {{
            color: var(--text-secondary);
            font-size: 0.8em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }}

        .card-value {{
            font-size: 1.8em;
            font-weight: 600;
        }}

        .card-subtitle {{
            color: var(--text-muted);
            font-size: 0.85em;
        }}

        .health-bar {{
            height: 8px;
            background: var(--bg-tertiary);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }}

        .health-bar-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }}

        .health-excellent {{ background: var(--accent-green); }}
        .health-good {{ background: var(--accent-blue); }}
        .health-warning {{ background: var(--accent-yellow); }}
        .health-danger {{ background: var(--accent-orange); }}
        .health-critical {{ background: var(--accent-red); }}

        .diagnosis-list {{
            margin: 20px 0;
        }}

        .diagnosis {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            margin-bottom: 10px;
            overflow: hidden;
        }}

        .diagnosis-header {{
            padding: 12px 15px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .diagnosis-header:hover {{
            background: var(--bg-tertiary);
        }}

        .diagnosis-icon {{
            font-size: 1.2em;
        }}

        .diagnosis-title {{
            flex: 1;
            font-weight: 500;
        }}

        .diagnosis-details {{
            padding: 15px;
            background: var(--bg-tertiary);
            border-top: 1px solid var(--border-color);
            display: none;
        }}

        .diagnosis.expanded .diagnosis-details {{
            display: block;
        }}

        .remediation {{
            background: rgba(88, 166, 255, 0.1);
            border-left: 3px solid var(--accent-blue);
            padding: 10px 15px;
            margin-top: 10px;
            font-size: 0.9em;
        }}

        .remediation strong {{
            color: var(--accent-blue);
        }}

        pre, code {{
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
            font-size: 0.85em;
        }}

        pre {{
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 15px;
            overflow-x: auto;
            margin: 10px 0;
        }}

        code {{
            background: var(--bg-tertiary);
            padding: 2px 6px;
            border-radius: 3px;
        }}

        pre code {{
            background: none;
            padding: 0;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}

        th, td {{
            text-align: left;
            padding: 10px 12px;
            border-bottom: 1px solid var(--border-color);
        }}

        th {{
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 0.85em;
            text-transform: uppercase;
        }}

        tr:hover {{
            background: var(--bg-secondary);
        }}

        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
            color: var(--text-muted);
            font-size: 0.85em;
            text-align: center;
        }}

        .footer a {{
            color: var(--accent-blue);
            text-decoration: none;
        }}

        .section {{
            margin: 30px 0;
        }}

        .expandable {{
            cursor: pointer;
        }}

        .expandable::before {{
            content: '▶ ';
            font-size: 0.8em;
            color: var(--text-muted);
        }}

        .expandable.expanded::before {{
            content: '▼ ';
        }}

        @media (max-width: 600px) {{
            body {{ padding: 10px; }}
            .summary-cards {{ grid-template-columns: 1fr; }}
            .card-value {{ font-size: 1.4em; }}
        }}
    </style>
</head>
<body>
    {content}

    <div class="footer">
        Generated by <a href="https://github.com/anthropics/claude-code">Temple Diagnostics</a> • {timestamp}
    </div>

    <script>
        // Toggle diagnosis details
        document.querySelectorAll('.diagnosis-header').forEach(header => {{
            header.addEventListener('click', () => {{
                header.parentElement.classList.toggle('expanded');
            }});
        }});

        // Toggle expandable sections
        document.querySelectorAll('.expandable').forEach(el => {{
            el.addEventListener('click', () => {{
                el.classList.toggle('expanded');
                const content = el.nextElementSibling;
                if (content) content.style.display = content.style.display === 'none' ? 'block' : 'none';
            }});
        }});
    </script>
</body>
</html>'''


# =============================================================================
# REPORT DATA CLASSES
# =============================================================================

@dataclass
class ReportSection:
    """A section of the report."""
    title: str
    content: str  # HTML content
    priority: int = 5  # Lower = higher priority (shows first)


@dataclass
class TempleReport:
    """A complete Temple diagnostic report."""
    title: str
    report_type: ReportType
    timestamp: str = ""
    sections: List[ReportSection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    diagnoses: List[Diagnosis] = field(default_factory=list)
    health_scores: Dict[str, float] = field(default_factory=dict)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    @property
    def html(self) -> str:
        """Generate HTML report."""
        return self._generate_html()

    @property
    def markdown(self) -> str:
        """Generate Markdown report."""
        return self._generate_markdown()

    @property
    def json_data(self) -> Dict[str, Any]:
        """Generate JSON-serializable data."""
        return self._generate_json()

    def to_html(self, path: str | Path):
        """Save HTML report to file."""
        Path(path).write_text(self.html)

    def to_markdown(self, path: str | Path):
        """Save Markdown report to file."""
        Path(path).write_text(self.markdown)

    def to_json(self, path: str | Path):
        """Save JSON report to file."""
        Path(path).write_text(json.dumps(self.json_data, indent=2))

    def _generate_html(self) -> str:
        """Generate full HTML report."""
        content_parts = []

        # Header
        content_parts.append(self._html_header())

        # Summary cards
        if self.health_scores:
            content_parts.append(self._html_summary_cards())

        # Diagnoses
        if self.diagnoses:
            content_parts.append(self._html_diagnoses())

        # Additional sections
        sorted_sections = sorted(self.sections, key=lambda s: s.priority)
        for section in sorted_sections:
            content_parts.append(f'<div class="section"><h2>{html_lib.escape(section.title)}</h2>{section.content}</div>')

        # Config snapshot
        if self.config_snapshot:
            content_parts.append(self._html_config())

        content = "\n".join(content_parts)
        return HTML_TEMPLATE.format(
            title=html_lib.escape(self.title),
            content=content,
            timestamp=self.timestamp,
        )

    def _html_header(self) -> str:
        """Generate HTML header."""
        report_badge = {
            ReportType.PROBLEM: ("Problem Report", "badge-critical"),
            ReportType.ACHIEVEMENT: ("Achievement", "badge-ok"),
            ReportType.CAMPAIGN: ("Campaign Summary", "badge-info"),
            ReportType.CHECKPOINT: ("Checkpoint Analysis", "badge-info"),
            ReportType.PREFLIGHT: ("Pre-flight Check", "badge-info"),
            ReportType.HEALTH: ("Health Report", "badge-info"),
        }.get(self.report_type, ("Report", "badge-info"))

        meta_items = []
        if self.metadata.get("model"):
            meta_items.append(f'<span>🤖 {html_lib.escape(str(self.metadata["model"]))}</span>')
        if self.metadata.get("step"):
            meta_items.append(f'<span>📊 Step {self.metadata["step"]:,}</span>')
        if self.metadata.get("campaign"):
            meta_items.append(f'<span>🎯 {html_lib.escape(str(self.metadata["campaign"]))}</span>')

        return f'''
        <div class="header">
            <span class="badge {report_badge[1]}">{report_badge[0]}</span>
            <h1>{html_lib.escape(self.title)}</h1>
            <div class="meta">
                <span>🕐 {self.timestamp}</span>
                {" ".join(meta_items)}
            </div>
        </div>
        '''

    def _html_summary_cards(self) -> str:
        """Generate summary cards HTML."""
        cards = []

        # Overall health
        if "overall" in self.health_scores:
            health = self.health_scores["overall"]
            health_class = self._health_class(health)
            cards.append(f'''
            <div class="card">
                <div class="card-title">Overall Health</div>
                <div class="card-value" style="color: var(--accent-{health_class.split('-')[1]})">{health:.0%}</div>
                <div class="health-bar">
                    <div class="health-bar-fill {health_class}" style="width: {health*100}%"></div>
                </div>
            </div>
            ''')

        # Other health scores
        score_names = {
            "gradient": ("Energy Flow", "🌊"),
            "memory": ("Memory", "💾"),
            "lr": ("Learning Rate", "⚡"),
            "data": ("Data Quality", "📦"),
            "loss": ("Loss Health", "📉"),
        }

        for key, (name, icon) in score_names.items():
            if key in self.health_scores:
                health = self.health_scores[key]
                health_class = self._health_class(health)
                cards.append(f'''
                <div class="card">
                    <div class="card-title">{icon} {name}</div>
                    <div class="card-value">{health:.0%}</div>
                    <div class="health-bar">
                        <div class="health-bar-fill {health_class}" style="width: {health*100}%"></div>
                    </div>
                </div>
                ''')

        # Issue counts
        if self.diagnoses:
            critical = sum(1 for d in self.diagnoses if d.severity == DiagnosticSeverity.CRITICAL)
            errors = sum(1 for d in self.diagnoses if d.severity == DiagnosticSeverity.ERROR)
            warnings = sum(1 for d in self.diagnoses if d.severity == DiagnosticSeverity.WARN)

            if critical or errors or warnings:
                cards.append(f'''
                <div class="card">
                    <div class="card-title">Issues Found</div>
                    <div class="card-value">{critical + errors + warnings}</div>
                    <div class="card-subtitle">
                        {f'🚨 {critical} critical' if critical else ''}
                        {f'❌ {errors} errors' if errors else ''}
                        {f'⚠️ {warnings} warnings' if warnings else ''}
                    </div>
                </div>
                ''')

        if not cards:
            return ""

        return f'<div class="summary-cards">{"".join(cards)}</div>'

    def _html_diagnoses(self) -> str:
        """Generate diagnoses HTML."""
        sorted_diagnoses = sorted(self.diagnoses, key=lambda d: d.severity, reverse=True)

        items = []
        for d in sorted_diagnoses:
            severity_class = {
                DiagnosticSeverity.CRITICAL: "badge-critical",
                DiagnosticSeverity.ERROR: "badge-error",
                DiagnosticSeverity.WARN: "badge-warn",
                DiagnosticSeverity.INFO: "badge-info",
            }.get(d.severity, "badge-info")

            remediation_html = ""
            if d.remediation:
                remediation_html = f'''
                <div class="remediation">
                    <strong>💡 Fix:</strong>
                    <pre>{html_lib.escape(d.remediation)}</pre>
                </div>
                '''

            evidence_html = ""
            if d.evidence:
                evidence_html = f'''
                <details>
                    <summary>Evidence</summary>
                    <pre><code>{html_lib.escape(json.dumps(d.evidence, indent=2))}</code></pre>
                </details>
                '''

            items.append(f'''
            <div class="diagnosis">
                <div class="diagnosis-header">
                    <span class="diagnosis-icon">{d.category.icon}</span>
                    <span class="badge {severity_class}">{d.severity.value}</span>
                    <span class="diagnosis-title">{html_lib.escape(d.summary)}</span>
                </div>
                <div class="diagnosis-details">
                    <p>{html_lib.escape(d.details)}</p>
                    {remediation_html}
                    {evidence_html}
                </div>
            </div>
            ''')

        return f'''
        <div class="section">
            <h2>Diagnoses ({len(self.diagnoses)})</h2>
            <div class="diagnosis-list">{"".join(items)}</div>
        </div>
        '''

    def _html_config(self) -> str:
        """Generate config snapshot HTML."""
        return f'''
        <div class="section">
            <h3 class="expandable">Configuration Snapshot</h3>
            <div style="display: none;">
                <pre><code>{html_lib.escape(json.dumps(self.config_snapshot, indent=2))}</code></pre>
            </div>
        </div>
        '''

    def _health_class(self, health: float) -> str:
        """Get CSS class for health value."""
        if health >= 0.9:
            return "health-excellent"
        elif health >= 0.7:
            return "health-good"
        elif health >= 0.5:
            return "health-warning"
        elif health >= 0.3:
            return "health-danger"
        else:
            return "health-critical"

    def _generate_markdown(self) -> str:
        """Generate Markdown report."""
        lines = []

        # Header
        lines.append(f"# {self.title}")
        lines.append("")
        lines.append(f"**Type:** {self.report_type.value.title()} Report")
        lines.append(f"**Generated:** {self.timestamp}")
        if self.metadata.get("model"):
            lines.append(f"**Model:** {self.metadata['model']}")
        if self.metadata.get("step"):
            lines.append(f"**Step:** {self.metadata['step']:,}")
        lines.append("")

        # Health scores
        if self.health_scores:
            lines.append("## Health Scores")
            lines.append("")
            lines.append("| Metric | Score |")
            lines.append("|--------|-------|")
            for key, value in self.health_scores.items():
                emoji = "🟢" if value >= 0.8 else "🟡" if value >= 0.5 else "🔴"
                lines.append(f"| {key.title()} | {emoji} {value:.0%} |")
            lines.append("")

        # Diagnoses
        if self.diagnoses:
            lines.append("## Diagnoses")
            lines.append("")
            for d in sorted(self.diagnoses, key=lambda x: x.severity, reverse=True):
                emoji = {"critical": "🚨", "error": "❌", "warn": "⚠️", "info": "ℹ️"}.get(d.severity.value, "📋")
                lines.append(f"### {emoji} {d.summary}")
                lines.append("")
                lines.append(f"**Severity:** {d.severity.value.upper()}")
                lines.append(f"**Category:** {d.category.value}")
                lines.append("")
                lines.append(d.details)
                lines.append("")
                if d.remediation:
                    lines.append("**Remediation:**")
                    lines.append("```")
                    lines.append(d.remediation)
                    lines.append("```")
                    lines.append("")

        # Sections
        for section in sorted(self.sections, key=lambda s: s.priority):
            lines.append(f"## {section.title}")
            lines.append("")
            # Convert HTML to plain text (basic)
            import re
            text = re.sub(r'<[^>]+>', '', section.content)
            lines.append(text)
            lines.append("")

        # Config
        if self.config_snapshot:
            lines.append("<details>")
            lines.append("<summary>Configuration Snapshot</summary>")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(self.config_snapshot, indent=2))
            lines.append("```")
            lines.append("</details>")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append("*Generated by Temple Diagnostics*")

        return "\n".join(lines)

    def _generate_json(self) -> Dict[str, Any]:
        """Generate JSON data."""
        return {
            "title": self.title,
            "report_type": self.report_type.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "health_scores": self.health_scores,
            "diagnoses": [d.to_dict() for d in self.diagnoses],
            "config_snapshot": self.config_snapshot,
            "sections": [
                {"title": s.title, "content": s.content, "priority": s.priority}
                for s in self.sections
            ],
        }


# =============================================================================
# REPORT GENERATOR
# =============================================================================

class ReportGenerator:
    """Factory for creating different types of reports."""

    @staticmethod
    def problem_report(
        diagnoses: List[Diagnosis],
        title: str = "Training Issue Report",
        config: Optional[Dict[str, Any]] = None,
        health_scores: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        description: str = "",
    ) -> TempleReport:
        """
        Generate a problem report for sharing when things go wrong.

        This is optimized for asking for help - includes all context needed.
        """
        sections = []

        if description:
            sections.append(ReportSection(
                title="Problem Description",
                content=f"<p>{html_lib.escape(description)}</p>",
                priority=1,
            ))

        # Add "What I've Tried" section placeholder
        sections.append(ReportSection(
            title="What I've Tried",
            content="<p><em>Add your debugging attempts here before sharing...</em></p>",
            priority=10,
        ))

        return TempleReport(
            title=title,
            report_type=ReportType.PROBLEM,
            diagnoses=diagnoses,
            health_scores=health_scores or {},
            config_snapshot=config or {},
            metadata=metadata or {},
            sections=sections,
        )

    @staticmethod
    def health_report(
        diagnoses: List[Diagnosis],
        health_scores: Dict[str, float],
        title: str = "Training Health Report",
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TempleReport:
        """Generate a general health status report."""
        return TempleReport(
            title=title,
            report_type=ReportType.HEALTH,
            diagnoses=diagnoses,
            health_scores=health_scores,
            config_snapshot=config or {},
            metadata=metadata or {},
        )

    @staticmethod
    def campaign_report(
        campaign_name: str,
        start_step: int,
        end_step: int,
        diagnoses: List[Diagnosis],
        health_scores: Dict[str, float],
        metrics: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> TempleReport:
        """Generate a full campaign summary report."""
        sections = []

        # Training progress
        sections.append(ReportSection(
            title="Training Progress",
            content=f'''
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Start Step</td><td>{start_step:,}</td></tr>
                <tr><td>End Step</td><td>{end_step:,}</td></tr>
                <tr><td>Total Steps</td><td>{end_step - start_step:,}</td></tr>
            </table>
            ''',
            priority=2,
        ))

        # Metrics table if provided
        if metrics:
            rows = "".join(f"<tr><td>{html_lib.escape(str(k))}</td><td>{v}</td></tr>" for k, v in metrics.items())
            sections.append(ReportSection(
                title="Key Metrics",
                content=f"<table><tr><th>Metric</th><th>Value</th></tr>{rows}</table>",
                priority=3,
            ))

        return TempleReport(
            title=f"Campaign Report: {campaign_name}",
            report_type=ReportType.CAMPAIGN,
            diagnoses=diagnoses,
            health_scores=health_scores,
            config_snapshot=config or {},
            metadata={"campaign": campaign_name, "start_step": start_step, "end_step": end_step},
            sections=sections,
        )

    @staticmethod
    def preflight_report(
        preflight_result: "PreflightReport",  # noqa: F821
        title: str = "Pre-flight Check Report",
    ) -> TempleReport:
        """Generate report from preflight check results."""
        diagnoses = []

        # Convert preflight checks to diagnoses
        for check in preflight_result.checks:
            if not check.passed:
                diagnoses.append(Diagnosis(
                    id=check.id,
                    category=DiagnosisCategory.SYSTEM,
                    severity=check.severity,
                    summary=check.name,
                    details=check.message,
                    remediation=check.recommendation or "",
                    evidence=check.details,
                ))

        sections = []

        # Summary table
        rows = []
        for check in preflight_result.checks:
            status = "✅" if check.passed else "❌"
            rows.append(f"<tr><td>{status}</td><td>{html_lib.escape(check.name)}</td><td>{html_lib.escape(check.message)}</td></tr>")

        sections.append(ReportSection(
            title="Check Results",
            content=f'''
            <table>
                <tr><th>Status</th><th>Check</th><th>Message</th></tr>
                {"".join(rows)}
            </table>
            ''',
            priority=2,
        ))

        # VRAM estimate
        if preflight_result.predicted_vram_gb:
            sections.append(ReportSection(
                title="VRAM Estimate",
                content=f'''
                <div class="card">
                    <div class="card-title">Predicted VRAM Usage</div>
                    <div class="card-value">{preflight_result.predicted_vram_gb:.1f} GB</div>
                    <div class="card-subtitle">Available: {preflight_result.available_vram_gb:.1f} GB</div>
                </div>
                ''',
                priority=3,
            ))

        health = {"overall": 1.0 if preflight_result.ready_to_train else 0.3}

        return TempleReport(
            title=title,
            report_type=ReportType.PREFLIGHT,
            diagnoses=diagnoses,
            health_scores=health,
            config_snapshot=preflight_result.config,
            metadata={"ready_to_train": preflight_result.ready_to_train},
            sections=sections,
        )

    @staticmethod
    def from_diagnostic_report(
        diagnostic_report: "DiagnosticReport",  # noqa: F821
        title: str = "Diagnostic Report",
    ) -> TempleReport:
        """Convert a DiagnosticReport to a TempleReport."""
        # Determine report type based on content
        if diagnostic_report.has_critical:
            report_type = ReportType.PROBLEM
        else:
            report_type = ReportType.HEALTH

        health = {
            "overall": diagnostic_report.overall_health,
            "gradient": diagnostic_report.gradient_health,
            "memory": diagnostic_report.memory_health,
            "lr": diagnostic_report.lr_health,
            "data": diagnostic_report.data_health,
            "loss": diagnostic_report.loss_health,
        }

        sections = []

        # Predictions
        if diagnostic_report.predicted_oom_steps or diagnostic_report.predicted_nan_steps:
            predictions = []
            if diagnostic_report.predicted_oom_steps:
                predictions.append(f"<li>💾 OOM predicted in ~{diagnostic_report.predicted_oom_steps:,} steps</li>")
            if diagnostic_report.predicted_nan_steps:
                predictions.append(f"<li>📉 NaN predicted in ~{diagnostic_report.predicted_nan_steps:,} steps</li>")

            sections.append(ReportSection(
                title="⚠️ Predictions",
                content=f"<ul>{''.join(predictions)}</ul>",
                priority=1,
            ))

        return TempleReport(
            title=title,
            report_type=report_type,
            diagnoses=diagnostic_report.diagnoses,
            health_scores=health,
            metadata={"step": diagnostic_report.step},
            sections=sections,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_problem_report(
    diagnoses: List[Diagnosis],
    title: str = "Help Needed: Training Issue",
    description: str = "",
    output_path: Optional[str] = None,
    format: str = "html",
) -> str:
    """
    Quick function to generate a shareable problem report.

    Returns the report content as a string, optionally saves to file.
    """
    report = ReportGenerator.problem_report(
        diagnoses=diagnoses,
        title=title,
        description=description,
    )

    if format == "html":
        content = report.html
        ext = ".html"
    elif format == "markdown":
        content = report.markdown
        ext = ".md"
    else:
        content = json.dumps(report.json_data, indent=2)
        ext = ".json"

    if output_path:
        path = Path(output_path)
        if not path.suffix:
            path = path.with_suffix(ext)
        path.write_text(content)

    return content


def generate_health_report(
    health_scores: Dict[str, float],
    diagnoses: Optional[List[Diagnosis]] = None,
    output_path: Optional[str] = None,
) -> str:
    """Quick function to generate a health status report."""
    report = ReportGenerator.health_report(
        diagnoses=diagnoses or [],
        health_scores=health_scores,
    )

    content = report.html

    if output_path:
        Path(output_path).write_text(content)

    return content
