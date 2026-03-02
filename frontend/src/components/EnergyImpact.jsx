export default function EnergyImpact({ panels, kpis }) {
    const totalLoss = panels.reduce((sum, p) => sum + (p.energy_loss_kwh_day || 0), 0)
    const totalCostLoss = panels.reduce((sum, p) => sum + (p.cost_loss_usd_day || 0), 0)
    const totalCO2 = panels.reduce((sum, p) => sum + (p.co2_impact_kg_day || 0), 0)
    const recoverable = totalLoss * 0.82

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">⚡ Energy & Sustainability Impact</span>
                <span className="card-badge yellow">Live</span>
            </div>

            <div className="impact-grid">
                <div className="impact-item">
                    <div className="impact-value" style={{ color: 'var(--accent-red)' }}>{totalLoss.toFixed(0)}</div>
                    <div className="impact-label">kWh/day Lost</div>
                </div>
                <div className="impact-item">
                    <div className="impact-value" style={{ color: 'var(--accent-yellow)' }}>${totalCostLoss.toFixed(0)}</div>
                    <div className="impact-label">Revenue Loss/day</div>
                </div>
                <div className="impact-item">
                    <div className="impact-value" style={{ color: 'var(--accent-orange)' }}>{totalCO2.toFixed(0)}</div>
                    <div className="impact-label">kg CO₂ Impact/day</div>
                </div>
            </div>

            <div style={{ marginTop: 16, padding: '14px', background: 'rgba(52,211,153,0.08)', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(52,211,153,0.2)' }}>
                <div style={{ fontSize: '0.78rem', color: 'var(--accent-green)', fontWeight: 600, marginBottom: 4 }}>
                    💡 If All Faults Fixed Now
                </div>
                <div style={{ display: 'flex', gap: 20, fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                    <span>Recovery: <strong style={{ color: 'var(--accent-green)' }}>{recoverable.toFixed(0)} kWh/day</strong></span>
                    <span>Savings: <strong style={{ color: 'var(--accent-green)' }}>${(recoverable * 0.08).toFixed(0)}/day</strong></span>
                    <span>CO₂ Saved: <strong style={{ color: 'var(--accent-green)' }}>{(recoverable * 0.42).toFixed(0)} kg/day</strong></span>
                </div>
            </div>

            {kpis && (
                <div style={{ marginTop: 14, display: 'flex', gap: 16, fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    <span>Downtime ↓ <strong style={{ color: 'var(--accent-green)' }}>{kpis.downtime_reduction_pct}%</strong></span>
                    <span>Yield ↑ <strong style={{ color: 'var(--accent-green)' }}>{kpis.energy_yield_recovery_pct}%</strong></span>
                    <span>CO₂ Avoided: <strong style={{ color: 'var(--accent-green)' }}>{kpis.co2_savings_tonnes_year} t/yr</strong></span>
                </div>
            )}
        </div>
    )
}
