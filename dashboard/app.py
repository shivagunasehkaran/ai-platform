"""Streamlit dashboard for AI platform metrics visualization."""

import json
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Page config
st.set_page_config(
    page_title="AI Platform Dashboard",
    page_icon="📊",
    layout="wide",
)

# Auto-refresh every 30 seconds
st.markdown(
    """
    <meta http-equiv="refresh" content="30">
    """,
    unsafe_allow_html=True,
)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
METRICS_FILE = PROJECT_ROOT / "metrics.json"
RAG_DATA_DIR = PROJECT_ROOT / "rag" / "data"


@st.cache_data(ttl=30)
def load_metrics() -> dict:
    """Load metrics from JSON file with 30s cache."""
    if not METRICS_FILE.exists():
        return {
            "chat_metrics": [],
            "retrieval_metrics": [],
            "agent_metrics": [],
        }
    
    with open(METRICS_FILE, "r") as f:
        return json.load(f)


@st.cache_data(ttl=30)
def load_index_stats() -> dict:
    """Load FAISS index statistics."""
    metadata_file = RAG_DATA_DIR / "metadata.json"
    index_file = RAG_DATA_DIR / "index.faiss"
    
    stats = {
        "indexed": False,
        "num_chunks": 0,
        "num_files": 0,
        "index_size_mb": 0,
    }
    
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        stats["indexed"] = True
        stats["num_chunks"] = len(metadata)
        stats["num_files"] = len(set(m["source"] for m in metadata))
    
    if index_file.exists():
        stats["index_size_mb"] = index_file.stat().st_size / (1024 * 1024)
    
    return stats


def render_header():
    """Render dashboard header."""
    st.title("📊 AI Platform Evaluation Dashboard")
    st.markdown("Real-time metrics for Chat, RAG, and Agent components • Auto-refreshes every 30s")
    st.divider()


def render_sidebar(metrics: dict) -> tuple:
    """Render sidebar with filters and controls.
    
    Returns:
        tuple: (date_range, selected_metrics)
    """
    with st.sidebar:
        st.header("⚙️ Controls")
        
        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        
        # Date range filter
        st.header("📅 Date Range")
        
        # Get min/max dates from all metrics
        all_timestamps = []
        for key in ["chat_metrics", "retrieval_metrics", "agent_metrics"]:
            for m in metrics.get(key, []):
                if "timestamp" in m:
                    try:
                        all_timestamps.append(datetime.fromisoformat(m["timestamp"]))
                    except:
                        pass
        
        if all_timestamps:
            min_date = min(all_timestamps).date()
            max_date = max(all_timestamps).date()
        else:
            min_date = datetime.now().date() - timedelta(days=7)
            max_date = datetime.now().date()
        
        date_range = st.date_input(
            "Select range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        
        st.divider()
        
        # Metric type filter
        st.header("📊 Metric Types")
        selected_metrics = st.multiselect(
            "Show metrics",
            options=["Chat", "RAG", "Agent"],
            default=["Chat", "RAG", "Agent"],
        )
        
        st.divider()
        
        st.header("📁 Data Sources")
        st.markdown(f"**Metrics:** `metrics.json`")
        st.markdown(f"**RAG data:** `rag/data/`")
        
        st.divider()
        
        # Status indicator
        st.header("📡 Status")
        chat_count = len(metrics.get("chat_metrics", []))
        rag_count = len(metrics.get("retrieval_metrics", []))
        agent_count = len(metrics.get("agent_metrics", []))
        
        st.markdown(f"- Chat records: **{chat_count}**")
        st.markdown(f"- RAG records: **{rag_count}**")
        st.markdown(f"- Agent records: **{agent_count}**")
    
    return date_range, selected_metrics


def filter_by_date(df: pd.DataFrame, date_range: tuple) -> pd.DataFrame:
    """Filter DataFrame by date range."""
    if df.empty or "timestamp" not in df.columns:
        return df
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df[
            (df["timestamp"].dt.date >= start_date) &
            (df["timestamp"].dt.date <= end_date)
        ]
    
    return df


def render_overview_metrics(metrics: dict, index_stats: dict):
    """Render overview KPI cards."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Chat metrics summary
    chat_metrics = metrics.get("chat_metrics", [])
    total_cost = sum(m.get("cost", 0) for m in chat_metrics)
    total_chats = len(chat_metrics)
    avg_latency = (
        sum(m.get("latency_ms", 0) for m in chat_metrics) / total_chats
        if total_chats > 0 else 0
    )
    
    # RAG metrics
    retrieval_metrics = metrics.get("retrieval_metrics", [])
    avg_recall = (
        sum(m.get("recall", 0) for m in retrieval_metrics) / len(retrieval_metrics)
        if retrieval_metrics else 0
    )
    
    # Agent metrics
    agent_metrics = metrics.get("agent_metrics", [])
    success_rate = (
        sum(1 for m in agent_metrics if m.get("success")) / len(agent_metrics)
        if agent_metrics else 0
    )
    
    with col1:
        st.metric("Chat Sessions", total_chats)
    
    with col2:
        st.metric("Total Cost", f"${total_cost:.4f}")
    
    with col3:
        st.metric("Avg Latency", f"{avg_latency:.0f}ms")
    
    with col4:
        st.metric("RAG Recall@5", f"{avg_recall:.1%}")
    
    with col5:
        st.metric("Agent Success", f"{success_rate:.1%}")


def render_chat_metrics(metrics: dict, date_range: tuple):
    """Render chat metrics section."""
    st.subheader("💬 Chat Metrics")
    
    chat_metrics = metrics.get("chat_metrics", [])
    
    if not chat_metrics:
        st.info("No chat metrics recorded yet. Use the chat CLI to generate data.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(chat_metrics)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = filter_by_date(df, date_range)
    
    if df.empty:
        st.warning("No data in selected date range.")
        return
    
    df = df.sort_values("timestamp")
    df["cumulative_cost"] = df["cost"].cumsum()
    
    # Calculate average tokens per message
    avg_prompt_tokens = df["prompt_tokens"].mean()
    avg_completion_tokens = df["completion_tokens"].mean()
    avg_total_tokens = avg_prompt_tokens + avg_completion_tokens
    
    # KPI row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Prompt Tokens", f"{avg_prompt_tokens:.0f}")
    with col2:
        st.metric("Avg Completion Tokens", f"{avg_completion_tokens:.0f}")
    with col3:
        st.metric("Avg Total Tokens/Message", f"{avg_total_tokens:.0f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Latency over time
        fig = px.line(
            df, x="timestamp", y="latency_ms",
            title="Latency Over Time",
            markers=True,
        )
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Latency (ms)",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cumulative cost
        fig = px.area(
            df, x="timestamp", y="cumulative_cost",
            title="Cumulative Cost Over Time",
        )
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Cost ($)",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Token distribution
    fig = go.Figure(go.Pie(
        labels=["Prompt Tokens", "Completion Tokens"],
        values=[df["prompt_tokens"].sum(), df["completion_tokens"].sum()],
        hole=0.4,
        marker=dict(colors=["#636EFA", "#EF553B"]),
    ))
    fig.update_layout(title="Token Distribution", height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    with st.expander("📋 Detailed Chat Metrics"):
        st.dataframe(
            df[["timestamp", "prompt_tokens", "completion_tokens", "cost", "latency_ms"]],
            use_container_width=True,
        )


def render_rag_metrics(metrics: dict, date_range: tuple, index_stats: dict):
    """Render RAG metrics section."""
    st.subheader("🎯 RAG Retrieval Metrics")
    
    # Index stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Indexed Chunks", f"{index_stats['num_chunks']:,}")
    with col2:
        st.metric("Source Files", index_stats['num_files'])
    with col3:
        st.metric("Index Size", f"{index_stats['index_size_mb']:.2f} MB")
    
    retrieval_metrics = metrics.get("retrieval_metrics", [])
    
    if not retrieval_metrics:
        st.info("No retrieval metrics recorded yet. Run `python -m rag.evaluate` to generate data.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(retrieval_metrics)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = filter_by_date(df, date_range)
    
    if df.empty:
        st.warning("No data in selected date range.")
        return
    
    df = df.sort_values("timestamp")
    
    # Gauge metrics for Recall@5 and MRR@5
    avg_recall = df["recall"].mean()
    avg_mrr = df["mrr"].mean()
    avg_latency = df["latency_ms"].mean()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Recall@5 Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_recall * 100,
            title={"text": "Recall@5"},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#636EFA"},
                "steps": [
                    {"range": [0, 50], "color": "#FED7D7"},
                    {"range": [50, 70], "color": "#FEEBC8"},
                    {"range": [70, 100], "color": "#C6F6D5"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # MRR@5 Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_mrr * 100,
            title={"text": "MRR@5"},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#EF553B"},
                "steps": [
                    {"range": [0, 30], "color": "#FED7D7"},
                    {"range": [30, 50], "color": "#FEEBC8"},
                    {"range": [50, 100], "color": "#C6F6D5"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Latency metric
        fig = go.Figure(go.Indicator(
            mode="number+delta",
            value=avg_latency,
            title={"text": "Avg Retrieval Latency"},
            number={"suffix": " ms"},
            delta={"reference": 300, "relative": False, "increasing": {"color": "red"}, "decreasing": {"color": "green"}},
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    # Latency histogram
    fig = px.histogram(
        df, x="latency_ms",
        title="Retrieval Latency Distribution",
        nbins=20,
        color_discrete_sequence=["#636EFA"],
    )
    fig.update_layout(
        xaxis_title="Latency (ms)",
        yaxis_title="Count",
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    with st.expander("📋 Detailed Retrieval Metrics"):
        st.dataframe(df, use_container_width=True)


def render_agent_metrics(metrics: dict, date_range: tuple):
    """Render agent metrics section."""
    st.subheader("🤖 Agent Metrics")
    
    agent_metrics = metrics.get("agent_metrics", [])
    
    if not agent_metrics:
        st.info("No agent metrics recorded yet. Run agent tasks to generate data.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(agent_metrics)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = filter_by_date(df, date_range)
    
    if df.empty:
        st.warning("No data in selected date range.")
        return
    
    # Success rate
    success_count = df["success"].sum()
    failure_count = len(df) - success_count
    success_rate = success_count / len(df) if len(df) > 0 else 0
    
    # KPI row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Success Rate", f"{success_rate:.1%}")
    with col2:
        st.metric("Avg Tool Calls", f"{df['tool_calls'].mean():.1f}")
    with col3:
        st.metric("Total Cost", f"${df['cost'].sum():.4f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Success/Failure pie chart
        fig = go.Figure(go.Pie(
            labels=["Success", "Failure"],
            values=[success_count, failure_count],
            marker=dict(colors=["#00CC96", "#EF553B"]),
            hole=0.4,
        ))
        fig.update_layout(
            title="Task Completion Rate",
            height=350,
            annotations=[dict(
                text=f"{success_count}/{len(df)}",
                x=0.5, y=0.5,
                font_size=20,
                showarrow=False,
            )],
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cost breakdown by task
        fig = px.bar(
            df,
            x=df["task"].str[:25],
            y="cost",
            color="success",
            color_discrete_map={True: "#00CC96", False: "#EF553B"},
            title="Cost Breakdown by Task",
        )
        fig.update_layout(
            xaxis_title="Task",
            yaxis_title="Cost ($)",
            height=350,
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tool calls distribution
    fig = px.bar(
        df,
        x=df["task"].str[:25],
        y="tool_calls",
        color="success",
        color_discrete_map={True: "#00CC96", False: "#EF553B"},
        title="Tool Calls per Task",
    )
    fig.update_layout(
        xaxis_title="Task",
        yaxis_title="Tool Calls",
        height=300,
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    with st.expander("📋 Detailed Agent Metrics"):
        st.dataframe(df, use_container_width=True)


def main():
    """Main dashboard entry point."""
    # Load data first for sidebar
    metrics = load_metrics()
    index_stats = load_index_stats()
    
    # Render sidebar and get filters
    date_range, selected_metrics = render_sidebar(metrics)
    
    # Header
    render_header()
    
    # Overview metrics
    render_overview_metrics(metrics, index_stats)
    
    st.divider()
    
    # Render selected metric sections
    if "Chat" in selected_metrics:
        render_chat_metrics(metrics, date_range)
        st.divider()
    
    if "RAG" in selected_metrics:
        render_rag_metrics(metrics, date_range, index_stats)
        st.divider()
    
    if "Agent" in selected_metrics:
        render_agent_metrics(metrics, date_range)


if __name__ == "__main__":
    main()
