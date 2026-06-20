"""Dashboard visualization module for VertAI.

Supports charts (line, bar, pie), dashboard layouts, and HTML report export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Union
import html
import json


class ChartType(Enum):
    """Supported chart types."""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"


class DashboardTheme(Enum):
    """Dashboard color themes."""
    LIGHT = "light"
    DARK = "dark"


# Default configuration constants
DEFAULT_CHART_WIDTH = 600
DEFAULT_CHART_HEIGHT = 400
DEFAULT_MAX_LABEL_LENGTH = 30
DEFAULT_DECIMAL_PLACES = 2
DEFAULT_COLORS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
]


@dataclass
class ChartConfig:
    """Configuration for chart rendering.

    Args:
        width: Chart width in pixels.
        height: Chart height in pixels.
        colors: List of color codes for data series.
        show_legend: Whether to display legend.
        show_grid: Whether to display grid lines.
        decimal_places: Number of decimal places for values.
        max_label_length: Maximum length for axis labels.
    """
    width: int = DEFAULT_CHART_WIDTH
    height: int = DEFAULT_CHART_HEIGHT
    colors: list[str] = field(default_factory=lambda: DEFAULT_COLORS.copy())
    show_legend: bool = True
    show_grid: bool = True
    decimal_places: int = DEFAULT_DECIMAL_PLACES
    max_label_length: int = DEFAULT_MAX_LABEL_LENGTH


@dataclass
class Metric:
    """Dashboard metric display.

    Args:
        name: Metric name/label.
        value: Metric value (numeric or string).
        unit: Optional unit suffix.
        format_spec: Optional format specifier (e.g., '.2%').
        threshold: Optional threshold for color coding.
    """
    name: str
    value: Union[int, float, str]
    unit: str = ""
    format_spec: str = ""
    threshold: Union[int, float, None] = None

    def __post_init__(self) -> None:
        """Validate metric parameters."""
        if not self.name or not self.name.strip():
            raise ValueError("Metric name cannot be empty")
        if isinstance(self.value, str) and not self.value.strip():
            raise ValueError("Metric value cannot be empty string")

    def format_value(self) -> str:
        """Format the metric value for display.

        Returns:
            Formatted string representation of the value.
        """
        if isinstance(self.value, str):
            return self.value
        if self.format_spec:
            return format(self.value, self.format_spec)
        if isinstance(self.value, float):
            return f"{self.value:.{self.decimal_places}f}"
        return str(self.value)

    @property
    def decimal_places(self) -> int:
        """Get decimal places for formatting."""
        return DEFAULT_DECIMAL_PLACES


@dataclass
class Chart:
    """Chart data container.

    Args:
        title: Chart title.
        chart_type: Type of chart (line, bar, pie).
        data: Chart data (list of values or dict of series).
        labels: Optional labels for data points.
        config: Optional chart configuration.
    """
    title: str
    chart_type: ChartType
    data: Union[list[Union[int, float]], dict[str, list[Union[int, float]]]]
    labels: list[str] = field(default_factory=list)
    config: ChartConfig = field(default_factory=ChartConfig)

    def __post_init__(self) -> None:
        """Validate chart parameters."""
        if not self.title or not self.title.strip():
            raise ValueError("Chart title cannot be empty")
        self._validate_data()

    def _validate_data(self) -> None:
        """Validate chart data structure."""
        if isinstance(self.data, list):
            if not self.data:
                raise ValueError("Chart data cannot be empty")
            for i, val in enumerate(self.data):
                if not isinstance(val, (int, float)):
                    raise TypeError(
                        f"Data value at index {i} must be numeric, "
                        f"got {type(val).__name__}"
                    )
        elif isinstance(self.data, dict):
            if not self.data:
                raise ValueError("Chart data cannot be empty")
            for series_name, series_data in self.data.items():
                if not series_data:
                    raise ValueError(f"Series '{series_name}' cannot be empty")
                for i, val in enumerate(series_data):
                    if not isinstance(val, (int, float)):
                        raise TypeError(
                            f"Value at index {i} in series '{series_name}' "
                            f"must be numeric, got {type(val).__name__}"
                        )


class Dashboard:
    """Dashboard for visualizing metrics and charts with HTML export.

    Examples:
        >>> from vertai.viz.dashboard import Dashboard
        >>> dash = Dashboard(title="Performance Dashboard")
        >>> dash.add_metric("任务完成率", 85, unit="%")
        >>> dash.add_chart("趋势", [10, 20, 30, 40], chart_type="line")
        >>> dash.show()  # Display dashboard
        >>> dash.export("report.html")  # Export to HTML file

    Note:
        Dashboard lives in the optional ``vertai[viz]`` extra and is not
        imported by ``import vertai``. Install/use the explicit path above.
    """

    def __init__(
        self,
        title: str = "Dashboard",
        theme: Union[str, DashboardTheme] = DashboardTheme.LIGHT,
        config: ChartConfig | None = None,
    ) -> None:
        """Initialize the dashboard.

        Args:
            title: Dashboard title.
            theme: Color theme (light or dark).
            config: Default chart configuration.
        """
        if not title or not title.strip():
            raise ValueError("Dashboard title cannot be empty")
        self._title = title.strip()
        self._theme = self._resolve_theme(theme)
        self._config = config or ChartConfig()
        self._metrics: list[Metric] = []
        self._charts: list[Chart] = []
        self._created_at = datetime.now()

    def _resolve_theme(self, theme: Union[str, DashboardTheme]) -> DashboardTheme:
        """Resolve theme parameter to DashboardTheme enum.

        Args:
            theme: Theme name or enum value.

        Returns:
            DashboardTheme enum value.

        Raises:
            ValueError: If theme is invalid.
        """
        if isinstance(theme, DashboardTheme):
            return theme
        theme_lower = theme.lower()
        valid_themes = [t.value for t in DashboardTheme]
        if theme_lower not in valid_themes:
            raise ValueError(
                f"Invalid theme: {theme}. Valid themes: {valid_themes}"
            )
        return DashboardTheme(theme_lower)

    @property
    def title(self) -> str:
        """Get dashboard title."""
        return self._title

    @property
    def theme(self) -> DashboardTheme:
        """Get dashboard theme."""
        return self._theme

    @property
    def metrics(self) -> list[Metric]:
        """Get list of metrics."""
        return self._metrics.copy()

    @property
    def charts(self) -> list[Chart]:
        """Get list of charts."""
        return self._charts.copy()

    def add_metric(
        self,
        name: str,
        value: Union[int, float, str],
        unit: str = "",
        format_spec: str = "",
        threshold: Union[int, float, None] = None,
    ) -> "Dashboard":
        """Add a metric to the dashboard.

        Args:
            name: Metric name/label.
            value: Metric value.
            unit: Optional unit suffix.
            format_spec: Optional format specifier.
            threshold: Optional threshold for color coding.

        Returns:
            Self for method chaining.
        """
        metric = Metric(
            name=name,
            value=value,
            unit=unit,
            format_spec=format_spec,
            threshold=threshold,
        )
        self._metrics.append(metric)
        return self

    def add_chart(
        self,
        title: str,
        data: Union[list[Union[int, float]], dict[str, list[Union[int, float]]]],
        chart_type: Union[str, ChartType] = ChartType.BAR,
        labels: list[str] | None = None,
        config: ChartConfig | None = None,
    ) -> "Dashboard":
        """Add a chart to the dashboard.

        Args:
            title: Chart title.
            data: Chart data (list of values or dict of series).
            chart_type: Type of chart (line, bar, pie).
            labels: Optional labels for data points.
            config: Optional chart configuration override.

        Returns:
            Self for method chaining.
        """
        if isinstance(chart_type, str):
            chart_type = self._resolve_chart_type(chart_type)

        chart = Chart(
            title=title,
            chart_type=chart_type,
            data=data,
            labels=labels or [],
            config=config or self._config,
        )
        self._charts.append(chart)
        return self

    def _resolve_chart_type(self, chart_type: str) -> ChartType:
        """Resolve chart type string to enum.

        Args:
            chart_type: Chart type string.

        Returns:
            ChartType enum value.

        Raises:
            ValueError: If chart type is invalid.
        """
        chart_type_lower = chart_type.lower()
        valid_types = [t.value for t in ChartType]
        if chart_type_lower not in valid_types:
            raise ValueError(
                f"Invalid chart type: {chart_type}. Valid types: {valid_types}"
            )
        return ChartType(chart_type_lower)

    def show(self) -> str:
        """Generate and return the dashboard HTML.

        Returns:
            HTML string representation of the dashboard.
        """
        return self._render_html()

    def export(self, filepath: str) -> None:
        """Export dashboard to an HTML file.

        Args:
            filepath: Output file path.

        Raises:
            IOError: If file cannot be written.
        """
        content = self._render_html()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    def to_json(self) -> str:
        """Export dashboard configuration as JSON.

        Returns:
            JSON string of dashboard data.
        """
        data = {
            "title": self._title,
            "theme": self._theme.value,
            "created_at": self._created_at.isoformat(),
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "unit": m.unit,
                    "format_spec": m.format_spec,
                    "threshold": m.threshold,
                }
                for m in self._metrics
            ],
            "charts": [
                {
                    "title": c.title,
                    "type": c.chart_type.value,
                    "data": c.data,
                    "labels": c.labels,
                }
                for c in self._charts
            ],
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    def _render_html(self) -> str:
        """Render complete dashboard as HTML.

        Returns:
            Complete HTML document string.
        """
        is_dark = self._theme == DashboardTheme.DARK
        bg_color = "#1a1a2e" if is_dark else "#ffffff"
        text_color = "#e0e0e0" if is_dark else "#333333"
        card_bg = "#16213e" if is_dark else "#f8f9fa"
        border_color = "#0f3460" if is_dark else "#dee2e6"

        html_parts = [
            "<!DOCTYPE html>",
            '<html lang="zh-CN">',
            "<head>",
            '<meta charset="UTF-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
            f"<title>{html.escape(self._title)}</title>",
            "<style>",
            self._get_css_styles(is_dark, bg_color, text_color, card_bg, border_color),
            "</style>",
            "</head>",
            f"<body style='background-color: {bg_color}; color: {text_color};'>",
            "<div class='container'>",
            f"<h1 class='dashboard-title'>{html.escape(self._title)}</h1>",
            f"<p class='timestamp'>生成时间: {self._created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>",
        ]

        if self._metrics:
            html_parts.append("<div class='metrics-grid'>")
            for metric in self._metrics:
                html_parts.append(self._render_metric_card(metric, card_bg, is_dark))
            html_parts.append("</div>")

        if self._charts:
            html_parts.append("<div class='charts-grid'>")
            for chart in self._charts:
                html_parts.append(self._render_chart(chart, card_bg, is_dark))
            html_parts.append("</div>")

        html_parts.extend([
            "</div>",
            "<script>",
            self._get_chart_script(),
            "</script>",
            "</body>",
            "</html>",
        ])

        return "\n".join(html_parts)

    def _get_css_styles(
        self,
        is_dark: bool,
        bg_color: str,
        text_color: str,
        card_bg: str,
        border_color: str,
    ) -> str:
        """Generate CSS styles for the dashboard.

        Args:
            is_dark: Whether dark theme is active.
            bg_color: Background color.
            text_color: Text color.
            card_bg: Card background color.
            border_color: Border color.

        Returns:
            CSS stylesheet string.
        """
        metric_positive = "#28a745" if not is_dark else "#4ade80"
        metric_negative = "#dc3545" if not is_dark else "#f87171"

        return f"""
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                min-height: 100vh;
                padding: 2rem;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .dashboard-title {{
                font-size: 2rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
                border-bottom: 3px solid {border_color};
                padding-bottom: 1rem;
            }}
            .timestamp {{
                color: {'#888' if is_dark else '#666'};
                margin-bottom: 2rem;
                font-size: 0.875rem;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1.5rem;
                margin-bottom: 2rem;
            }}
            .metric-card {{
                background: {card_bg};
                border-radius: 8px;
                padding: 1.5rem;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border: 1px solid {border_color};
            }}
            .metric-name {{
                font-size: 0.875rem;
                color: {'#aaa' if is_dark else '#666'};
                margin-bottom: 0.5rem;
            }}
            .metric-value {{
                font-size: 2rem;
                font-weight: 700;
            }}
            .metric-unit {{
                font-size: 1rem;
                font-weight: 400;
                margin-left: 0.25rem;
            }}
            .metric-positive {{ color: {metric_positive}; }}
            .metric-negative {{ color: {metric_negative}; }}
            .charts-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 1.5rem;
            }}
            .chart-card {{
                background: {card_bg};
                border-radius: 8px;
                padding: 1.5rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border: 1px solid {border_color};
            }}
            .chart-title {{
                font-size: 1.125rem;
                font-weight: 600;
                margin-bottom: 1rem;
            }}
            .chart-container {{
                position: relative;
                width: 100%;
            }}
            canvas {{
                max-width: 100%;
                height: auto;
            }}
        """

    def _render_metric_card(
        self, metric: Metric, card_bg: str, is_dark: bool
    ) -> str:
        """Render a single metric card HTML.

        Args:
            metric: Metric to render.
            card_bg: Card background color.
            is_dark: Whether dark theme is active.

        Returns:
            HTML string for the metric card.
        """
        value_str = metric.format_value()
        threshold_class = ""

        if metric.threshold is not None and isinstance(metric.value, (int, float)):
            if metric.value >= metric.threshold:
                threshold_class = "metric-positive"
            else:
                threshold_class = "metric-negative"

        return f"""
            <div class="metric-card">
                <div class="metric-name">{html.escape(metric.name)}</div>
                <div class="metric-value {threshold_class}">
                    {html.escape(value_str)}<span class="metric-unit">{html.escape(metric.unit)}</span>
                </div>
            </div>
        """

    def _render_chart(self, chart: Chart, card_bg: str, is_dark: bool) -> str:
        """Render a single chart card HTML.

        Args:
            chart: Chart to render.
            card_bg: Card background color.
            is_dark: Whether dark theme is active.

        Returns:
            HTML string for the chart card.
        """
        chart_id = f"chart_{id(chart)}"
        config_json = json.dumps({
            "type": chart.chart_type.value,
            "data": chart.data,
            "labels": chart.labels,
            "colors": chart.config.colors,
            "showLegend": chart.config.show_legend,
            "showGrid": chart.config.show_grid,
            "decimalPlaces": chart.config.decimal_places,
            "isDark": is_dark,
        })

        return f"""
            <div class="chart-card">
                <div class="chart-title">{html.escape(chart.title)}</div>
                <div class="chart-container">
                    <canvas id="{chart_id}" width="{chart.config.width}" height="{chart.config.height}"></canvas>
                </div>
                <script>
                    renderChart('{chart_id}', {config_json});
                </script>
            </div>
        """

    def _get_chart_script(self) -> str:
        """Get JavaScript for chart rendering.

        Returns:
            JavaScript code string for chart canvas rendering.
        """
        return """
            function renderChart(canvasId, config) {
                const canvas = document.getElementById(canvasId);
                const ctx = canvas.getContext('2d');
                const width = canvas.width;
                const height = canvas.height;
                const padding = 50;
                const colors = config.colors;

                ctx.clearRect(0, 0, width, height);

                const textColor = config.isDark ? '#e0e0e0' : '#333333';
                const gridColor = config.isDark ? '#333333' : '#e0e0e0';

                if (config.type === 'pie') {
                    renderPieChart(ctx, config, width, height, colors, textColor);
                } else {
                    renderAxisChart(ctx, config, width, height, padding, colors, textColor, gridColor);
                }
            }

            function renderPieChart(ctx, config, width, height, colors, textColor) {
                const centerX = width / 2;
                const centerY = height / 2;
                const radius = Math.min(width, height) / 2 - 50;

                const data = Array.isArray(config.data) ? config.data : Object.values(config.data)[0];
                const total = data.reduce((a, b) => a + b, 0);

                let startAngle = -Math.PI / 2;
                const labels = config.labels.length > 0 ? config.labels : data.map((_, i) => 'Slice ' + (i + 1));

                data.forEach((value, i) => {
                    const sliceAngle = (value / total) * 2 * Math.PI;
                    ctx.beginPath();
                    ctx.moveTo(centerX, centerY);
                    ctx.arc(centerX, centerY, radius, startAngle, startAngle + sliceAngle);
                    ctx.closePath();
                    ctx.fillStyle = colors[i % colors.length];
                    ctx.fill();

                    const midAngle = startAngle + sliceAngle / 2;
                    const labelX = centerX + Math.cos(midAngle) * (radius * 0.7);
                    const labelY = centerY + Math.sin(midAngle) * (radius * 0.7);

                    ctx.fillStyle = '#ffffff';
                    ctx.font = '12px sans-serif';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    const pct = (value / total * 100).toFixed(config.decimalPlaces);
                    ctx.fillText(pct + '%', labelX, labelY);

                    startAngle += sliceAngle;
                });

                if (config.showLegend) {
                    const legendX = width - 100;
                    let legendY = 20;
                    labels.forEach((label, i) => {
                        ctx.fillStyle = colors[i % colors.length];
                        ctx.fillRect(legendX, legendY, 15, 15);
                        ctx.fillStyle = textColor;
                        ctx.textAlign = 'left';
                        ctx.textBaseline = 'middle';
                        const displayLabel = label.length > 15 ? label.substring(0, 12) + '...' : label;
                        ctx.fillText(displayLabel, legendX + 20, legendY + 7);
                        legendY += 20;
                    });
                }
            }

            function renderAxisChart(ctx, config, width, height, padding, colors, textColor, gridColor) {
                const chartWidth = width - padding * 2;
                const chartHeight = height - padding * 2;

                let allData = [];
                let seriesNames = [];

                if (Array.isArray(config.data)) {
                    allData = [config.data];
                    seriesNames = ['Series 1'];
                } else {
                    seriesNames = Object.keys(config.data);
                    allData = Object.values(config.data);
                }

                const maxValue = Math.max(...allData.flat());
                const minValue = Math.min(0, ...allData.flat());
                const valueRange = maxValue - minValue || 1;

                const numPoints = allData[0].length;
                const labels = config.labels.length > 0 ? config.labels : allData[0].map((_, i) => String(i + 1));

                if (config.showGrid) {
                    ctx.strokeStyle = gridColor;
                    ctx.lineWidth = 1;
                    for (let i = 0; i <= 5; i++) {
                        const y = padding + (chartHeight / 5) * i;
                        ctx.beginPath();
                        ctx.moveTo(padding, y);
                        ctx.lineTo(width - padding, y);
                        ctx.stroke();

                        const value = maxValue - (valueRange / 5) * i;
                        ctx.fillStyle = textColor;
                        ctx.font = '11px sans-serif';
                        ctx.textAlign = 'right';
                        ctx.textBaseline = 'middle';
                        ctx.fillText(value.toFixed(config.decimalPlaces), padding - 5, y);
                    }
                }

                allData.forEach((series, seriesIndex) => {
                    const color = colors[seriesIndex % colors.length];

                    if (config.type === 'bar') {
                        const barWidth = chartWidth / numPoints / (allData.length + 1);
                        series.forEach((value, i) => {
                            const x = padding + (chartWidth / numPoints) * i + barWidth * seriesIndex + barWidth / 2;
                            const barHeight = (value / valueRange) * chartHeight;
                            const y = height - padding - barHeight;

                            ctx.fillStyle = color;
                            ctx.fillRect(x, y, barWidth * 0.8, barHeight);
                        });
                    } else if (config.type === 'line') {
                        ctx.beginPath();
                        ctx.strokeStyle = color;
                        ctx.lineWidth = 2;
                        series.forEach((value, i) => {
                            const x = padding + (chartWidth / (numPoints - 1 || 1)) * i;
                            const y = height - padding - ((value - minValue) / valueRange) * chartHeight;
                            if (i === 0) ctx.moveTo(x, y);
                            else ctx.lineTo(x, y);
                        });
                        ctx.stroke();

                        series.forEach((value, i) => {
                            const x = padding + (chartWidth / (numPoints - 1 || 1)) * i;
                            const y = height - padding - ((value - minValue) / valueRange) * chartHeight;
                            ctx.beginPath();
                            ctx.arc(x, y, 4, 0, Math.PI * 2);
                            ctx.fillStyle = color;
                            ctx.fill();
                        });
                    }
                });

                labels.forEach((label, i) => {
                    const x = padding + (chartWidth / (numPoints - 1 || 1)) * i;
                    ctx.fillStyle = textColor;
                    ctx.font = '11px sans-serif';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'top';
                    const displayLabel = label.length > 10 ? label.substring(0, 8) + '...' : label;
                    ctx.fillText(displayLabel, x, height - padding + 10);
                });

                if (config.showLegend && seriesNames.length > 1) {
                    const legendX = width - padding - 100;
                    let legendY = padding;
                    seriesNames.forEach((name, i) => {
                        ctx.fillStyle = colors[i % colors.length];
                        ctx.fillRect(legendX, legendY, 15, 15);
                        ctx.fillStyle = textColor;
                        ctx.textAlign = 'left';
                        ctx.textBaseline = 'middle';
                        ctx.fillText(name, legendX + 20, legendY + 7);
                        legendY += 20;
                    });
                }
            }
        """

    @classmethod
    def from_json(cls, json_str: str) -> "Dashboard":
        """Create dashboard from JSON string.

        Args:
            json_str: JSON string of dashboard data.

        Returns:
            New Dashboard instance.

        Raises:
            ValueError: If JSON is invalid or missing required fields.
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e

        if "title" not in data:
            raise ValueError("JSON must contain 'title' field")

        dashboard = cls(
            title=data["title"],
            theme=data.get("theme", DashboardTheme.LIGHT.value),
        )

        for metric_data in data.get("metrics", []):
            dashboard.add_metric(
                name=metric_data["name"],
                value=metric_data["value"],
                unit=metric_data.get("unit", ""),
                format_spec=metric_data.get("format_spec", ""),
                threshold=metric_data.get("threshold"),
            )

        for chart_data in data.get("charts", []):
            dashboard.add_chart(
                title=chart_data["title"],
                data=chart_data["data"],
                chart_type=chart_data.get("type", "bar"),
                labels=chart_data.get("labels", []),
            )

        return dashboard

    def clear(self) -> "Dashboard":
        """Clear all metrics and charts from the dashboard.

        Returns:
            Self for method chaining.
        """
        self._metrics.clear()
        self._charts.clear()
        return self
