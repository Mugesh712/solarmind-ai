import { useMemo } from 'react'

export default function PanelDetail({ detail, onClose }) {
    if (!detail) return null
    const { panel, history = [], forecast = {}, recommendation = {} } = detail
    const trajectory = forecast.trajectory || []

    const statusColor = panel.status === 'critical' ? 'var(--accent-red)' : panel.status === 'warning' ? 'var(--accent-yellow)' : 'var(--accent-green)'
    const defectLabel = panel.defect.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())

    // Simulated attention regions for heatmap
    const attentionRegions = useMemo(() => {
        if (panel.defect === 'normal') return []
        const count = Math.floor(Math.random() * 3) + 1
        return Array.from({ length: count }, () => ({
            x: Math.random() * 70 + 10,
            y: Math.random() * 70 + 10,
            size: Math.random() * 40 + 30,
            intensity: Math.random() * 0.4 + 0.5,
        }))
    }, [panel.id])

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                    <div>
                        <h3 style={{ fontSize: '1.2rem', fontWeight: 700 }}>
                            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-accent)' }}>{panel.id}</span>
                            <span style={{ fontSize: '0.8rem', fontWeight: 400, color: 'var(--text-muted)', marginLeft: 12 }}>{panel.zone}</span>
                        </h3>
                        <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: 2 }}>
                            {panel.panel_type} • Installed: {panel.install_date}
                        </div>
                    </div>
                    <button className="modal-close" onClick={onClose}>✕</button>
                </div>

                {/* Status Banner */}
                <div style={{
                    display: 'flex', gap: 16, marginBottom: 20, padding: '14px', borderRadius: 'var(--radius-sm)',
                    background: panel.defect === 'normal' ? 'rgba(52,211,153,0.08)' : 'rgba(248,113,113,0.08)',
                    border: `1px solid ${panel.defect === 'normal' ? 'rgba(52,211,153,0.2)' : 'rgba(248,113,113,0.2)'}`,
                }}>
                    <div style={{ textAlign: 'center', flex: 1 }}>
                        <div style={{ fontSize: '1.4rem', fontWeight: 800, color: statusColor }}>{defectLabel}</div>
                        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Defect Type</div>
                    </div>
                    <div style={{ textAlign: 'center', flex: 1 }}>
                        <div style={{ fontSize: '1.4rem', fontWeight: 800, color: statusColor }}>{(panel.severity * 100).toFixed(0)}%</div>
                        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Severity</div>
                    </div>
                    <div style={{ textAlign: 'center', flex: 1 }}>
                        <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--accent-blue)' }}>{(panel.confidence * 100).toFixed(0)}%</div>
                        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Confidence</div>
                    </div>
                    <div style={{ textAlign: 'center', flex: 1 }}>
                        <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--accent-cyan)' }}>{panel.rul_days}</div>
                        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>RUL (days)</div>
                    </div>
                </div>

                {/* Telemetry Summary */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginBottom: 20 }}>
                    <div className="impact-item">
                        <div className="impact-value" style={{ color: 'var(--accent-red)', fontSize: '1.1rem' }}>{panel.energy_loss_kwh_day}</div>
                        <div className="impact-label">kWh/day Loss</div>
                    </div>
                    <div className="impact-item">
                        <div className="impact-value" style={{ color: 'var(--accent-yellow)', fontSize: '1.1rem' }}>${panel.cost_loss_usd_day}</div>
                        <div className="impact-label">Cost/day</div>
                    </div>
                    <div className="impact-item">
                        <div className="impact-value" style={{ color: 'var(--accent-orange)', fontSize: '1.1rem' }}>{panel.co2_impact_kg_day}</div>
                        <div className="impact-label">kg CO₂/day</div>
                    </div>
                </div>

                {/* Attention Heatmap */}
                {panel.defect !== 'normal' && (
                    <div style={{ marginBottom: 20 }}>
                        <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                            🔬 ViT Attention Map
                        </div>
                        <div className="attention-canvas" style={{ height: 160, background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)' }}>
                            {/* Grid overlay */}
                            <div style={{ position: 'absolute', inset: 0, display: 'grid', gridTemplateColumns: 'repeat(14, 1fr)', gridTemplateRows: 'repeat(14, 1fr)', opacity: 0.15 }}>
                                {Array.from({ length: 196 }).map((_, i) => (
                                    <div key={i} style={{ border: '0.5px solid rgba(255,255,255,0.2)' }} />
                                ))}
                            </div>
                            {/* Attention hotspots */}
                            {attentionRegions.map((region, i) => (
                                <div key={i} className="attention-hotspot" style={{
                                    left: `${region.x}%`, top: `${region.y}%`,
                                    width: `${region.size}px`, height: `${region.size}px`,
                                    opacity: region.intensity,
                                    transform: 'translate(-50%, -50%)',
                                }} />
                            ))}
                            <div style={{ position: 'absolute', bottom: 8, right: 10, fontSize: '0.6rem', color: 'rgba(255,255,255,0.5)' }}>
                                Attention regions highlighted
                            </div>
                        </div>
                    </div>
                )}

                {/* Progression Forecast Chart */}
                {trajectory.length > 0 && panel.defect !== 'normal' && (
                    <div style={{ marginBottom: 20 }}>
                        <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                            📈 Defect Progression Forecast (90 days)
                        </div>
                        <div className="chart-container">
                            <div className="chart-line">
                                {trajectory.map((point, i) => (
                                    <div
                                        key={i}
                                        className="chart-bar"
                                        data-value={`Day ${point.day}: ${(point.severity * 100).toFixed(0)}%`}
                                        style={{
                                            height: `${point.severity * 100}%`,
                                            background: point.severity > 0.8 ? 'var(--accent-red)' : point.severity > 0.5 ? 'var(--accent-yellow)' : 'var(--accent-blue)',
                                            opacity: 0.8 + point.severity * 0.2,
                                        }}
                                    />
                                ))}
                            </div>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: 4 }}>
                            <span>Today</span>
                            <span>30 days</span>
                            <span>60 days</span>
                            <span>90 days</span>
                        </div>
                    </div>
                )}

                {/* History */}
                {history.length > 0 && (
                    <div style={{ marginBottom: 20 }}>
                        <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                            📋 Inspection History
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                            {history.map((h, i) => (
                                <div key={i} style={{
                                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                                    padding: '8px 12px', background: 'var(--bg-primary)', borderRadius: 'var(--radius-sm)',
                                    fontSize: '0.75rem', border: '1px solid var(--border-color)',
                                }}>
                                    <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', minWidth: 85 }}>{h.date}</span>
                                    <span style={{ color: h.severity > 0.5 ? 'var(--accent-red)' : 'var(--text-secondary)' }}>
                                        Sev: {(h.severity * 100).toFixed(0)}%
                                    </span>
                                    <span style={{ color: 'var(--text-muted)' }}>{h.action_taken}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Recommendation */}
                {recommendation && recommendation.action && (
                    <div style={{
                        padding: '14px', borderRadius: 'var(--radius-sm)',
                        background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.2)',
                    }}>
                        <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--accent-blue)', marginBottom: 6 }}>
                            🛠️ AI Recommendation
                        </div>
                        <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>
                            {recommendation.action}
                        </div>
                        {recommendation.rationale && (
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                                {recommendation.rationale}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    )
}
