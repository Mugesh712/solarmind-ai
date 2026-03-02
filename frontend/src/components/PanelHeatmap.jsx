import { useState } from 'react'

export default function PanelHeatmap({ panels, onPanelClick }) {
    const [hoveredPanel, setHoveredPanel] = useState(null)

    const getStatusClass = (panel) => {
        if (panel.defect === 'normal' || panel.defect === 'Clean') return 'healthy'
        if (panel.severity > 0.7) return 'critical'
        return 'warning'
    }

    const getTooltip = (panel) => {
        return `${panel.id} | ${panel.defect.replace('_', ' ')} | Sev: ${(panel.severity * 100).toFixed(0)}%`
    }

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">📡 Panel Health Map</span>
                <span className="card-badge green">{panels.length} Panels</span>
            </div>

            <div className="heatmap-grid">
                {panels.map((panel) => (
                    <div
                        key={panel.id}
                        className={`heatmap-cell ${getStatusClass(panel)}`}
                        onClick={() => onPanelClick(panel)}
                        onMouseEnter={() => setHoveredPanel(panel)}
                        onMouseLeave={() => setHoveredPanel(null)}
                        title={getTooltip(panel)}
                    />
                ))}
            </div>

            <div className="heatmap-legend">
                <div className="legend-item">
                    <div className="legend-dot healthy"></div>
                    <span>Healthy</span>
                </div>
                <div className="legend-item">
                    <div className="legend-dot warning"></div>
                    <span>Warning</span>
                </div>
                <div className="legend-item">
                    <div className="legend-dot critical"></div>
                    <span>Critical</span>
                </div>
            </div>

            {hoveredPanel && hoveredPanel.defect !== 'normal' && hoveredPanel.defect !== 'Clean' && (
                <div style={{
                    marginTop: 12,
                    padding: '10px 14px',
                    background: 'var(--bg-secondary)',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--border-color)',
                    fontSize: '0.78rem',
                }}>
                    <strong style={{ color: 'var(--text-accent)', fontFamily: 'var(--font-mono)' }}>{hoveredPanel.id}</strong>
                    <span style={{ color: 'var(--text-muted)', margin: '0 8px' }}>|</span>
                    <span style={{ color: hoveredPanel.severity > 0.7 ? 'var(--accent-red)' : 'var(--accent-yellow)' }}>
                        {hoveredPanel.defect.replace('_', ' ')}
                    </span>
                    <span style={{ color: 'var(--text-muted)', margin: '0 8px' }}>|</span>
                    <span>Severity: {(hoveredPanel.severity * 100).toFixed(0)}%</span>
                    <span style={{ color: 'var(--text-muted)', margin: '0 8px' }}>|</span>
                    <span>Loss: {hoveredPanel.energy_loss_kwh_day} kWh/day</span>
                </div>
            )}
        </div>
    )
}
