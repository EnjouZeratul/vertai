"""Unit tests for Dashboard visualization module."""

import json
import pytest
from datetime import datetime

from vertai.viz.dashboard import (
    Chart,
    ChartConfig,
    ChartType,
    Dashboard,
    DashboardTheme,
    Metric,
)


class TestMetric:
    """Tests for Metric dataclass."""

    def test_metric_creation_basic(self):
        metric = Metric(name="完成率", value=85)
        assert metric.name == "完成率"
        assert metric.value == 85
        assert metric.unit == ""
        assert metric.format_spec == ""
        assert metric.threshold is None

    def test_metric_creation_with_all_params(self):
        metric = Metric(
            name="销售额",
            value=12345.67,
            unit="万元",
            format_spec=".2f",
            threshold=10000,
        )
        assert metric.name == "销售额"
        assert metric.value == 12345.67
        assert metric.unit == "万元"
        assert metric.format_spec == ".2f"
        assert metric.threshold == 10000

    def test_metric_format_value_integer(self):
        metric = Metric(name="Count", value=42)
        assert metric.format_value() == "42"

    def test_metric_format_value_float(self):
        metric = Metric(name="Rate", value=0.856)
        assert metric.format_value() == "0.86"

    def test_metric_format_value_with_format_spec(self):
        metric = Metric(name="Percent", value=0.856, format_spec=".2%")
        assert metric.format_value() == "85.60%"

    def test_metric_format_value_string(self):
        metric = Metric(name="Status", value="活跃")
        assert metric.format_value() == "活跃"

    def test_metric_empty_name_raises(self):
        with pytest.raises(ValueError, match="name cannot be empty"):
            Metric(name="", value=100)

    def test_metric_whitespace_name_raises(self):
        with pytest.raises(ValueError, match="name cannot be empty"):
            Metric(name="   ", value=100)

    def test_metric_empty_string_value_raises(self):
        with pytest.raises(ValueError, match="value cannot be empty string"):
            Metric(name="Test", value="   ")


class TestChart:
    """Tests for Chart dataclass."""

    def test_chart_creation_bar(self):
        chart = Chart(
            title="销售数据",
            chart_type=ChartType.BAR,
            data=[10, 20, 30, 40],
        )
        assert chart.title == "销售数据"
        assert chart.chart_type == ChartType.BAR
        assert chart.data == [10, 20, 30, 40]

    def test_chart_creation_with_series(self):
        chart = Chart(
            title="对比图",
            chart_type=ChartType.LINE,
            data={"系列A": [1, 2, 3], "系列B": [4, 5, 6]},
            labels=["Q1", "Q2", "Q3"],
        )
        assert chart.title == "对比图"
        assert chart.chart_type == ChartType.LINE
        assert "系列A" in chart.data
        assert chart.labels == ["Q1", "Q2", "Q3"]

    def test_chart_default_config(self):
        chart = Chart(title="Test", chart_type=ChartType.PIE, data=[1, 2, 3])
        assert chart.config.width == 600
        assert chart.config.height == 400

    def test_chart_custom_config(self):
        config = ChartConfig(width=800, height=600, show_legend=False)
        chart = Chart(
            title="Custom",
            chart_type=ChartType.BAR,
            data=[1, 2, 3],
            config=config,
        )
        assert chart.config.width == 800
        assert chart.config.height == 600
        assert chart.config.show_legend is False

    def test_chart_empty_title_raises(self):
        with pytest.raises(ValueError, match="title cannot be empty"):
            Chart(title="", chart_type=ChartType.BAR, data=[1, 2, 3])

    def test_chart_empty_data_raises(self):
        with pytest.raises(ValueError, match="data cannot be empty"):
            Chart(title="Test", chart_type=ChartType.BAR, data=[])

    def test_chart_empty_series_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            Chart(title="Test", chart_type=ChartType.BAR, data={"A": []})

    def test_chart_empty_dict_data_raises(self):
        """Test that empty dict data raises error."""
        with pytest.raises(ValueError, match="Chart data cannot be empty"):
            Chart(title="Test", chart_type=ChartType.BAR, data={})

    def test_chart_non_numeric_data_raises(self):
        with pytest.raises(TypeError, match="must be numeric"):
            Chart(title="Test", chart_type=ChartType.BAR, data=[1, "two", 3])

    def test_chart_non_numeric_series_raises(self):
        with pytest.raises(TypeError, match="must be numeric"):
            Chart(
                title="Test",
                chart_type=ChartType.LINE,
                data={"A": [1, 2, "three"]},
            )


class TestChartConfig:
    """Tests for ChartConfig dataclass."""

    def test_default_config(self):
        config = ChartConfig()
        assert config.width == 600
        assert config.height == 400
        assert config.show_legend is True
        assert config.show_grid is True
        assert config.decimal_places == 2

    def test_custom_config(self):
        config = ChartConfig(
            width=1000,
            height=800,
            show_legend=False,
            show_grid=False,
            decimal_places=3,
        )
        assert config.width == 1000
        assert config.height == 800
        assert config.show_legend is False
        assert config.show_grid is False
        assert config.decimal_places == 3

    def test_default_colors(self):
        config = ChartConfig()
        assert len(config.colors) == 10
        assert config.colors[0] == "#4E79A7"


class TestDashboardInit:
    """Tests for Dashboard initialization."""

    def test_init_default(self):
        dash = Dashboard()
        assert dash.title == "Dashboard"
        assert dash.theme == DashboardTheme.LIGHT
        assert dash.metrics == []
        assert dash.charts == []

    def test_init_with_title(self):
        dash = Dashboard(title="性能监控")
        assert dash.title == "性能监控"

    def test_init_with_theme_string(self):
        dash = Dashboard(theme="dark")
        assert dash.theme == DashboardTheme.DARK

    def test_init_with_theme_enum(self):
        dash = Dashboard(theme=DashboardTheme.DARK)
        assert dash.theme == DashboardTheme.DARK

    def test_init_with_config(self):
        config = ChartConfig(width=800, height=600)
        dash = Dashboard(config=config)
        assert dash.charts == []

    def test_init_empty_title_raises(self):
        with pytest.raises(ValueError, match="title cannot be empty"):
            Dashboard(title="")

    def test_init_invalid_theme_raises(self):
        with pytest.raises(ValueError, match="Invalid theme"):
            Dashboard(theme="invalid")


class TestDashboardMetrics:
    """Tests for Dashboard metric operations."""

    def test_add_metric_basic(self):
        dash = Dashboard()
        result = dash.add_metric("任务完成率", 85)
        assert result is dash
        assert len(dash.metrics) == 1
        assert dash.metrics[0].name == "任务完成率"
        assert dash.metrics[0].value == 85

    def test_add_metric_with_unit(self):
        dash = Dashboard()
        dash.add_metric("响应时间", 150, unit="ms")
        assert dash.metrics[0].unit == "ms"

    def test_add_metric_with_format_spec(self):
        dash = Dashboard()
        dash.add_metric("转化率", 0.1234, format_spec=".2%")
        assert dash.metrics[0].format_spec == ".2%"

    def test_add_metric_with_threshold(self):
        dash = Dashboard()
        dash.add_metric("完成数", 85, threshold=80)
        assert dash.metrics[0].threshold == 80

    def test_add_multiple_metrics(self):
        dash = Dashboard()
        dash.add_metric("M1", 10).add_metric("M2", 20).add_metric("M3", 30)
        assert len(dash.metrics) == 3

    def test_metric_copy_isolation(self):
        dash = Dashboard()
        dash.add_metric("Test", 100)
        metrics = dash.metrics
        assert metrics == dash._metrics
        metrics_copy = dash.metrics
        assert metrics_copy is not dash._metrics


class TestDashboardCharts:
    """Tests for Dashboard chart operations."""

    def test_add_chart_basic(self):
        dash = Dashboard()
        result = dash.add_chart("销售趋势", [10, 20, 30], chart_type="line")
        assert result is dash
        assert len(dash.charts) == 1
        assert dash.charts[0].title == "销售趋势"

    def test_add_chart_with_enum_type(self):
        dash = Dashboard()
        dash.add_chart("分布", [30, 40, 30], chart_type=ChartType.PIE)
        assert dash.charts[0].chart_type == ChartType.PIE

    def test_add_chart_with_series(self):
        dash = Dashboard()
        dash.add_chart(
            "对比",
            {"A": [1, 2], "B": [3, 4]},
            chart_type="bar",
            labels=["Q1", "Q2"],
        )
        assert "A" in dash.charts[0].data
        assert dash.charts[0].labels == ["Q1", "Q2"]

    def test_add_chart_with_config(self):
        config = ChartConfig(width=800, height=600)
        dash = Dashboard()
        dash.add_chart("Custom", [1, 2, 3], config=config)
        assert dash.charts[0].config.width == 800

    def test_add_chart_invalid_type_raises(self):
        dash = Dashboard()
        with pytest.raises(ValueError, match="Invalid chart type"):
            dash.add_chart("Test", [1, 2, 3], chart_type="invalid")

    def test_charts_copy_isolation(self):
        dash = Dashboard()
        dash.add_chart("Test", [1, 2, 3])
        charts = dash.charts
        assert charts is not dash._charts


class TestDashboardRendering:
    """Tests for Dashboard HTML rendering."""

    def test_show_returns_html(self):
        dash = Dashboard(title="测试仪表盘")
        html_output = dash.show()
        assert "<!DOCTYPE html>" in html_output
        assert "测试仪表盘" in html_output

    def test_show_includes_timestamp(self):
        dash = Dashboard()
        html_output = dash.show()
        assert "生成时间:" in html_output

    def test_show_includes_metrics(self):
        dash = Dashboard()
        dash.add_metric("完成率", 85, unit="%")
        html_output = dash.show()
        assert "完成率" in html_output
        assert "85" in html_output

    def test_show_includes_charts(self):
        dash = Dashboard()
        dash.add_chart("趋势", [10, 20, 30])
        html_output = dash.show()
        assert "趋势" in html_output
        assert "renderChart" in html_output

    def test_show_escapes_html(self):
        dash = Dashboard(title="<script>alert('xss')</script>")
        dash.add_metric("<b>Test</b>", 100)
        html_output = dash.show()
        assert "<script>alert" not in html_output
        assert "&lt;script&gt;" in html_output

    def test_dark_theme(self):
        dash = Dashboard(theme="dark")
        html_output = dash.show()
        assert "#1a1a2e" in html_output

    def test_light_theme(self):
        dash = Dashboard(theme="light")
        html_output = dash.show()
        assert "#ffffff" in html_output


class TestDashboardExport:
    """Tests for Dashboard export functionality."""

    def test_export_creates_file(self, tmp_path):
        dash = Dashboard(title="Export Test")
        dash.add_metric("M1", 100)
        filepath = str(tmp_path / "dashboard.html")

        dash.export(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        assert "<!DOCTYPE html>" in content
        assert "Export Test" in content

    def test_export_with_chinese_path(self, tmp_path):
        dash = Dashboard(title="中文测试")
        dash.add_chart("数据", [1, 2, 3])
        filepath = str(tmp_path / "仪表盘_测试.html")

        dash.export(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        assert "中文测试" in content


class TestDashboardJson:
    """Tests for Dashboard JSON serialization."""

    def test_to_json_basic(self):
        dash = Dashboard(title="JSON Test")
        json_str = dash.to_json()
        data = json.loads(json_str)

        assert data["title"] == "JSON Test"
        assert data["theme"] == "light"
        assert data["metrics"] == []
        assert data["charts"] == []

    def test_to_json_with_metrics(self):
        dash = Dashboard()
        dash.add_metric("Rate", 85, unit="%", threshold=80)
        json_str = dash.to_json()
        data = json.loads(json_str)

        assert len(data["metrics"]) == 1
        assert data["metrics"][0]["name"] == "Rate"
        assert data["metrics"][0]["value"] == 85
        assert data["metrics"][0]["unit"] == "%"
        assert data["metrics"][0]["threshold"] == 80

    def test_to_json_with_charts(self):
        dash = Dashboard()
        dash.add_chart("Trend", [1, 2, 3], chart_type="line", labels=["A", "B", "C"])
        json_str = dash.to_json()
        data = json.loads(json_str)

        assert len(data["charts"]) == 1
        assert data["charts"][0]["title"] == "Trend"
        assert data["charts"][0]["type"] == "line"
        assert data["charts"][0]["data"] == [1, 2, 3]
        assert data["charts"][0]["labels"] == ["A", "B", "C"]

    def test_from_json_basic(self):
        json_str = json.dumps({
            "title": "Restored Dashboard",
            "theme": "dark",
            "metrics": [],
            "charts": [],
        })

        dash = Dashboard.from_json(json_str)

        assert dash.title == "Restored Dashboard"
        assert dash.theme == DashboardTheme.DARK

    def test_from_json_with_metrics(self):
        json_str = json.dumps({
            "title": "Test",
            "metrics": [
                {"name": "M1", "value": 100, "unit": "ms"},
                {"name": "M2", "value": 85.5, "format_spec": ".1f"},
            ],
            "charts": [],
        })

        dash = Dashboard.from_json(json_str)

        assert len(dash.metrics) == 2
        assert dash.metrics[0].name == "M1"
        assert dash.metrics[0].unit == "ms"
        assert dash.metrics[1].format_spec == ".1f"

    def test_from_json_with_charts(self):
        json_str = json.dumps({
            "title": "Test",
            "metrics": [],
            "charts": [
                {
                    "title": "Chart1",
                    "type": "bar",
                    "data": [10, 20, 30],
                    "labels": ["X", "Y", "Z"],
                },
            ],
        })

        dash = Dashboard.from_json(json_str)

        assert len(dash.charts) == 1
        assert dash.charts[0].title == "Chart1"
        assert dash.charts[0].chart_type == ChartType.BAR

    def test_from_json_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            Dashboard.from_json("not valid json")

    def test_from_json_missing_title_raises(self):
        json_str = json.dumps({"metrics": [], "charts": []})
        with pytest.raises(ValueError, match="must contain 'title'"):
            Dashboard.from_json(json_str)

    def test_json_roundtrip(self):
        original = Dashboard(title="Roundtrip", theme="dark")
        original.add_metric("Rate", 95, unit="%", threshold=90)
        original.add_chart("Data", [1, 2, 3], chart_type="line", labels=["A", "B", "C"])

        json_str = original.to_json()
        restored = Dashboard.from_json(json_str)

        assert restored.title == original.title
        assert restored.theme == original.theme
        assert len(restored.metrics) == len(original.metrics)
        assert len(restored.charts) == len(original.charts)


class TestDashboardClear:
    """Tests for Dashboard clear functionality."""

    def test_clear_metrics_and_charts(self):
        dash = Dashboard()
        dash.add_metric("M1", 100).add_metric("M2", 200)
        dash.add_chart("C1", [1, 2, 3]).add_chart("C2", [4, 5, 6])

        result = dash.clear()

        assert result is dash
        assert len(dash.metrics) == 0
        assert len(dash.charts) == 0

    def test_clear_empty_dashboard(self):
        dash = Dashboard()
        result = dash.clear()

        assert result is dash
        assert dash.metrics == []
        assert dash.charts == []


class TestDashboardIntegration:
    """Integration tests for Dashboard."""

    def test_full_workflow(self, tmp_path):
        dash = Dashboard(title="性能监控仪表盘", theme="dark")
        dash.add_metric("任务完成率", 85, unit="%", threshold=80)
        dash.add_metric("平均响应时间", 150, unit="ms")
        dash.add_metric("错误率", 0.02, format_spec=".2%")

        dash.add_chart(
            "请求趋势",
            {"成功": [100, 120, 130], "失败": [5, 3, 2]},
            chart_type="line",
            labels=["周一", "周二", "周三"],
        )
        dash.add_chart(
            "状态分布",
            [70, 20, 10],
            chart_type="pie",
            labels=["成功", "警告", "失败"],
        )

        html_output = dash.show()
        assert "性能监控仪表盘" in html_output
        assert "任务完成率" in html_output
        assert "请求趋势" in html_output
        assert "状态分布" in html_output

        filepath = str(tmp_path / "performance.html")
        dash.export(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        assert "性能监控仪表盘" in content

    def test_multiple_series_bar_chart(self):
        dash = Dashboard()
        dash.add_chart(
            "季度对比",
            {
                "2023年": [100, 120, 110, 130],
                "2024年": [110, 140, 125, 150],
            },
            chart_type="bar",
            labels=["Q1", "Q2", "Q3", "Q4"],
        )

        html_output = dash.show()
        assert "季度对比" in html_output
        assert "renderChart" in html_output

    def test_metric_threshold_coloring(self):
        dash = Dashboard()
        dash.add_metric("达标", 90, threshold=80)
        dash.add_metric("未达标", 70, threshold=80)

        html_output = dash.show()
        assert "metric-positive" in html_output
        assert "metric-negative" in html_output
