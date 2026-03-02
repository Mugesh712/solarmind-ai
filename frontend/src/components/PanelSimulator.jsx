import { useState, useEffect, useRef } from 'react'

const API_BASE = 'http://localhost:8000'
const DEFECT_CLASSES = ['Clean', 'Bird-drop', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']
const DEFECT_COLORS = {
    'Clean': '#10b981',
    'normal': '#10b981',
    'Bird-drop': '#f59e0b',
    'Dusty': '#fb923c',
    'Electrical-damage': '#ef4444',
    'Physical-Damage': '#dc2626',
    'Snow-Covered': '#60a5fa',
    'micro_crack': '#fbbf24',
    'hotspot': '#ef4444',
    'dust_soiling': '#fb923c',
}
const DEFECT_ICONS = {
    'Clean': '✅',
    'normal': '✅',
    'Bird-drop': '🐦',
    'Dusty': '☁️',
    'Electrical-damage': '⚡',
    'Physical-Damage': '💥',
    'Snow-Covered': '❄️',
}

export default function PanelSimulator({ panels = [], onPanelsChanged }) {
    const [selectedPanel, setSelectedPanel] = useState(null)
    const [selectedDefect, setSelectedDefect] = useState('Clean')
    const [severity, setSeverity] = useState(0.5)
    const [loading, setLoading] = useState(false)
    const [activityLog, setActivityLog] = useState([])
    const [lastResult, setLastResult] = useState(null)
    const [eventLoading, setEventLoading] = useState('')

    const applyDefect = async () => {
        if (!selectedPanel) return
        setLoading(true)
        setLastResult(null)
        try {
            const res = await fetch(
                `${API_BASE}/api/simulate/panel/${selectedPanel.id}?defect=${encodeURIComponent(selectedDefect)}&severity=${severity}`,
                { method: 'POST' }
            )
            const data = await res.json()
            setLastResult(data.classification)
            setActivityLog(prev => [{
                time: new Date().toLocaleTimeString(),
                panel: selectedPanel.id,
                defect: selectedDefect,
                confidence: data.classification?.confidence,
                mode: data.classification?.mode || 'unknown',
            }, ...prev].slice(0, 15))
        } catch (err) {
            console.error('Failed to apply defect:', err)
        } finally {
            setLoading(false)
        }
    }

    const triggerEvent = async (eventType) => {
        setEventLoading(eventType)
        try {
            const res = await fetch(
                `${API_BASE}/api/simulate/event?event_type=${eventType}`,
                { method: 'POST' }
            )
            const data = await res.json()
            setActivityLog(prev => [{
                time: new Date().toLocaleTimeString(),
                panel: `${data.affected_count} panels`,
                defect: eventType,
                confidence: null,
                mode: 'event',
            }, ...prev].slice(0, 15))
        } catch (err) {
            console.error('Event failed:', err)
        } finally {
            setEventLoading('')
        }
    }

    const getDefectColor = (panel) => {
        return DEFECT_COLORS[panel.defect] || '#10b981'
    }

    return (
        <div className="simulator-layout">
            {/* Main Grid */}
            <div className="card simulator-grid-card">
                <div className="card-header">
                    <span className="card-title">🎛️ Panel Simulator — 200 Panel Grid</span>
                    <span className="card-badge blue">{panels.length} Panels</span>
                </div>
                <div className="simulator-grid">
                    {panels.map((panel) => (
                        <div
                            key={panel.id}
                            className={`sim-cell ${selectedPanel?.id === panel.id ? 'selected' : ''}`}
                            style={{ backgroundColor: getDefectColor(panel) }}
                            onClick={() => {
                                setSelectedPanel(panel)
                                setSelectedDefect(panel.defect === 'normal' ? 'Clean' : (DEFECT_CLASSES.includes(panel.defect) ? panel.defect : 'Clean'))
                                setSeverity(panel.severity || 0.5)
                            }}
                            title={`${panel.id} — ${panel.defect} (${panel.severity})`}
                        />
                    ))}
                </div>
                <div className="sim-legend">
                    {DEFECT_CLASSES.map(d => (
                        <div key={d} className="legend-item">
                            <span className="legend-dot" style={{ backgroundColor: DEFECT_COLORS[d] }}></span>
                            <span>{DEFECT_ICONS[d]} {d}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Controls Panel */}
            <div className="simulator-controls">
                {/* Selected Panel Info */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Selected Panel</span>
                    </div>
                    {selectedPanel ? (
                        <div className="sim-panel-info">
                            <div className="sim-panel-id">{selectedPanel.id}</div>
                            <div className="sim-panel-meta">
                                Row {selectedPanel.row}, Col {selectedPanel.col} • {selectedPanel.zone}
                            </div>
                            <div className="sim-panel-status">
                                <span style={{ color: getDefectColor(selectedPanel) }}>
                                    {DEFECT_ICONS[selectedPanel.defect] || '❓'} {selectedPanel.defect}
                                </span>
                                <span style={{ color: 'var(--text-muted)' }}>
                                    Severity: {selectedPanel.severity?.toFixed(2) || '0.00'}
                                </span>
                            </div>
                        </div>
                    ) : (
                        <div className="sim-panel-empty">
                            Click a panel in the grid to select it
                        </div>
                    )}
                </div>

                {/* Defect Type Selector */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Set Defect Type</span>
                    </div>
                    <div className="defect-buttons">
                        {DEFECT_CLASSES.map(d => (
                            <button
                                key={d}
                                className={`defect-btn ${selectedDefect === d ? 'active' : ''}`}
                                style={{
                                    borderColor: selectedDefect === d ? DEFECT_COLORS[d] : 'var(--border-color)',
                                    color: selectedDefect === d ? DEFECT_COLORS[d] : 'var(--text-secondary)',
                                    backgroundColor: selectedDefect === d ? DEFECT_COLORS[d] + '18' : 'transparent',
                                }}
                                onClick={() => {
                                    setSelectedDefect(d)
                                    if (d === 'Clean') setSeverity(0)
                                    else if (severity === 0) setSeverity(0.5)
                                }}
                            >
                                {DEFECT_ICONS[d]} {d}
                            </button>
                        ))}
                    </div>

                    {/* Severity Slider */}
                    <div className="severity-control">
                        <label>Severity: <strong>{severity.toFixed(2)}</strong></label>
                        <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={severity}
                            onChange={(e) => setSeverity(parseFloat(e.target.value))}
                            disabled={selectedDefect === 'Clean'}
                            className="severity-slider"
                        />
                        <div className="severity-labels">
                            <span>Low</span>
                            <span>Medium</span>
                            <span>High</span>
                            <span>Critical</span>
                        </div>
                    </div>

                    <button
                        className="apply-btn"
                        onClick={applyDefect}
                        disabled={!selectedPanel || loading}
                    >
                        {loading ? '🔄 Analyzing...' : '🔬 Apply & Analyze'}
                    </button>

                    {/* Last Classification Result */}
                    {lastResult && (
                        <div className="last-result">
                            <div className="result-header">
                                {lastResult.mode === 'real' ? '🤖 Real ViT Model' : lastResult.mode === 'analysis' ? '🔍 Pixel Analysis' : '📋 Manual'}
                            </div>
                            <div className="result-class" style={{ color: DEFECT_COLORS[lastResult.predicted_class] || 'var(--text-primary)' }}>
                                {lastResult.predicted_class}
                            </div>
                            <div className="result-conf">
                                Confidence: {((lastResult.confidence || 0) * 100).toFixed(1)}%
                            </div>
                        </div>
                    )}
                </div>

                {/* Quick Actions */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">⚡ Quick Actions</span>
                    </div>
                    <div className="quick-actions">
                        <button
                            className="event-btn dust"
                            onClick={() => triggerEvent('dust_storm')}
                            disabled={!!eventLoading}
                        >
                            {eventLoading === 'dust_storm' ? '⏳' : '☁️'} Dust Storm
                            <span className="event-desc">~10 panels get dusty</span>
                        </button>
                        <button
                            className="event-btn bird"
                            onClick={() => triggerEvent('bird_event')}
                            disabled={!!eventLoading}
                        >
                            {eventLoading === 'bird_event' ? '⏳' : '🐦'} Bird Event
                            <span className="event-desc">~3 panels get bird-drops</span>
                        </button>
                        <button
                            className="event-btn clean"
                            onClick={() => triggerEvent('maintenance')}
                            disabled={!!eventLoading}
                        >
                            {eventLoading === 'maintenance' ? '⏳' : '🧹'} Maintenance
                            <span className="event-desc">Clean all dusty panels</span>
                        </button>
                        <button
                            className="event-btn reset"
                            onClick={() => triggerEvent('reset')}
                            disabled={!!eventLoading}
                        >
                            {eventLoading === 'reset' ? '⏳' : '🔄'} Reset All
                            <span className="event-desc">All panels → Clean</span>
                        </button>
                    </div>
                </div>

                {/* Activity Log */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">📋 Activity Log</span>
                        <span className="card-badge blue">{activityLog.length}</span>
                    </div>
                    <div className="activity-log">
                        {activityLog.length === 0 ? (
                            <div className="sim-panel-empty">No activity yet</div>
                        ) : (
                            activityLog.map((entry, i) => (
                                <div key={i} className="log-entry">
                                    <span className="log-time">{entry.time}</span>
                                    <span className="log-panel">{entry.panel}</span>
                                    <span className="log-defect" style={{ color: DEFECT_COLORS[entry.defect] || 'var(--text-muted)' }}>
                                        {DEFECT_ICONS[entry.defect] || '📌'} {entry.defect}
                                    </span>
                                    {entry.confidence != null && (
                                        <span className="log-conf">{(entry.confidence * 100).toFixed(1)}%</span>
                                    )}
                                    {entry.mode === 'real' && <span className="log-badge real">ViT</span>}
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}
