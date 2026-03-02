import { useState } from 'react'

export default function PanelHeatmap({ panels, onPanelClick }) {
    const getStatusClass = (panel) => {
        if (panel.defect === 'normal' || panel.defect === 'Clean') return 'healthy'
        if (panel.severity > 0.7) return 'critical'
        return 'warning'
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
                        title={panel.id}
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
        </div>
    )
}
