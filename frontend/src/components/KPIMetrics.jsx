import { useState, useEffect } from 'react';

const API_URL = 'http://localhost:8000';

export default function KPIMetrics({ kpis }) {
    const [compData, setCompData] = useState(null);

    useEffect(() => {
        fetch(`${API_URL}/api/model/comparison`)
            .then(res => res.ok ? res.json() : null)
            .then(data => { if (data) setCompData(data); })
            .catch(() => {});
    }, []);

    if (!kpis) return null;

    const models = compData?.models || [];
    const bestModel = compData?.best_model || '';

    const getColor = (name) => {
        if (name.includes('Ensemble')) return '#f59e0b';
        if (name.includes('Swin')) return '#a855f7';
        if (name.includes('ViT')) return '#3b82f6';
        if (name.includes('ResNet')) return '#f97316';
        if (name.includes('Efficient')) return '#22c55e';
        return '#94a3b8';
    };

    const getIcon = (name) => {
        if (name.includes('Ensemble')) return '🧬';
        if (name.includes('Swin')) return '🔷';
        if (name.includes('ViT')) return '🔮';
        if (name.includes('ResNet')) return '🏗️';
        if (name.includes('Efficient')) return '⚡';
        return '🤖';
    };

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">🤖 AI Model Performance</span>
                <span className="card-badge green">
                    {models.length > 0 ? `${models.length} Models` : 'ViT-Small/16'}
                </span>
            </div>

            {models.length > 0 ? (
                <>
                    {/* Model comparison mini bars */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 4 }}>
                        {[...models]
                            .sort((a, b) => b.test_accuracy - a.test_accuracy)
                            .map((m) => (
                            <div key={m.model_name} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                                <span style={{ fontSize: '0.92rem', width: 22, textAlign: 'center', flexShrink: 0 }}>
                                    {getIcon(m.model_name)}
                                </span>
                                <span style={{
                                    fontSize: '0.74rem', fontWeight: 600, color: 'var(--text-secondary)',
                                    width: 110, flexShrink: 0, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                                }}>
                                    {m.model_name.length > 14 ? m.model_name.split(' ').slice(0, 2).join(' ') + '…' : m.model_name}
                                </span>
                                <div style={{
                                    flex: 1, height: 16, background: 'var(--bg-elevated)',
                                    borderRadius: 100, overflow: 'hidden', position: 'relative',
                                }}>
                                    <div style={{
                                        height: '100%', width: `${m.test_accuracy}%`,
                                        background: getColor(m.model_name),
                                        borderRadius: 100, transition: 'width 0.8s ease',
                                    }} />
                                </div>
                                <span style={{
                                    fontSize: '0.8rem', fontWeight: 800, fontFamily: 'var(--font-mono)',
                                    color: getColor(m.model_name), width: 50, textAlign: 'right', flexShrink: 0,
                                }}>
                                    {m.test_accuracy}%
                                </span>
                                {m.model_name === bestModel && (
                                    <span style={{
                                        fontSize: '0.58rem', padding: '1px 5px', borderRadius: 4,
                                        background: 'rgba(250,204,21,0.15)', color: '#eab308', fontWeight: 700, flexShrink: 0,
                                    }}>🏆</span>
                                )}
                            </div>
                        ))}
                    </div>

                    {/* Best model highlight */}
                    <div style={{
                        marginTop: 14, padding: '10px 14px',
                        background: 'rgba(245,158,11,0.06)', border: '1px solid rgba(245,158,11,0.2)',
                        borderRadius: 'var(--radius-sm)', display: 'flex', justifyContent: 'space-between',
                        alignItems: 'center', fontSize: '0.74rem',
                    }}>
                        <span style={{ color: 'var(--text-muted)' }}>
                            🏆 Best: <strong style={{ color: 'var(--text-primary)' }}>{bestModel}</strong>
                        </span>
                        <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                            Edge Latency: <strong style={{ color: 'var(--accent-green)' }}>{kpis.inference_latency_ms || 23.4}ms</strong>
                        </span>
                    </div>
                </>
            ) : (
                /* Fallback: original metrics display */
                <div className="metrics-grid">
                    {[
                        { label: 'Precision', value: `${(kpis.precision * 100).toFixed(1)}%`, fill: kpis.precision * 100, color: 'var(--accent-green)' },
                        { label: 'Recall', value: `${(kpis.recall * 100).toFixed(1)}%`, fill: kpis.recall * 100, color: 'var(--accent-blue)' },
                        { label: 'F1-Score', value: `${(kpis.f1_score * 100).toFixed(1)}%`, fill: kpis.f1_score * 100, color: 'var(--accent-cyan)' },
                        { label: 'mAP@0.5', value: `${(kpis.mAP * 100).toFixed(1)}%`, fill: kpis.mAP * 100, color: 'var(--accent-purple)' },
                        { label: 'Edge Latency', value: `${kpis.inference_latency_ms}ms`, fill: Math.max(0, 100 - kpis.inference_latency_ms * 2), color: 'var(--accent-yellow)' },
                        { label: 'Edge Uptime', value: `${kpis.edge_uptime_pct}%`, fill: kpis.edge_uptime_pct, color: 'var(--accent-green)' },
                    ].map((m, i) => (
                        <div key={i} className="metric-item">
                            <div className="metric-value" style={{ color: m.color }}>{m.value}</div>
                            <div className="metric-label">{m.label}</div>
                            <div className="metric-bar">
                                <div className="metric-fill" style={{ width: `${m.fill}%`, background: m.color }} />
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
