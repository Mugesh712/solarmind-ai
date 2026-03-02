export default function KPIMetrics({ kpis }) {
    if (!kpis) return null

    const metrics = [
        { label: 'Precision', value: `${(kpis.precision * 100).toFixed(1)}%`, fill: kpis.precision * 100, color: 'var(--accent-green)' },
        { label: 'Recall', value: `${(kpis.recall * 100).toFixed(1)}%`, fill: kpis.recall * 100, color: 'var(--accent-blue)' },
        { label: 'F1-Score', value: `${(kpis.f1_score * 100).toFixed(1)}%`, fill: kpis.f1_score * 100, color: 'var(--accent-cyan)' },
        { label: 'mAP@0.5', value: `${(kpis.mAP * 100).toFixed(1)}%`, fill: kpis.mAP * 100, color: 'var(--accent-purple)' },
        { label: 'Edge Latency', value: `${kpis.inference_latency_ms}ms`, fill: Math.max(0, 100 - kpis.inference_latency_ms * 2), color: 'var(--accent-yellow)' },
        { label: 'Edge Uptime', value: `${kpis.edge_uptime_pct}%`, fill: kpis.edge_uptime_pct, color: 'var(--accent-green)' },
    ]

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">🤖 AI Model Performance</span>
                <span className="card-badge green">ViT-Base/16</span>
            </div>
            <div className="metrics-grid">
                {metrics.map((m, i) => (
                    <div key={i} className="metric-item">
                        <div className="metric-value" style={{ color: m.color }}>{m.value}</div>
                        <div className="metric-label">{m.label}</div>
                        <div className="metric-bar">
                            <div className="metric-fill" style={{ width: `${m.fill}%`, background: m.color }} />
                        </div>
                    </div>
                ))}
            </div>
        </div>
    )
}
