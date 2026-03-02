import { useState, useEffect, useRef, useCallback } from 'react'
import Sidebar from './components/Sidebar'
import SiteOverview from './components/SiteOverview'
import PanelHeatmap from './components/PanelHeatmap'
import RecommendationQueue from './components/RecommendationQueue'
import EnergyImpact from './components/EnergyImpact'
import KPIMetrics from './components/KPIMetrics'
import ZoneHealth from './components/ZoneHealth'
import DefectDistribution from './components/DefectDistribution'
import WeatherWidget from './components/WeatherWidget'
import ProgressionChart from './components/ProgressionChart'
import AttentionMap from './components/AttentionMap'
import PanelDetail from './components/PanelDetail'
import ImageUpload from './components/ImageUpload'
import PanelSimulator from './components/PanelSimulator'

const API_BASE = 'http://localhost:8000'
const WS_URL = 'ws://localhost:8000/ws/live'

function App() {
    const [siteData, setSiteData] = useState(null)
    const [panels, setPanels] = useState([])
    const [recommendations, setRecommendations] = useState([])
    const [weather, setWeather] = useState([])
    const [selectedPanel, setSelectedPanel] = useState(null)
    const [activePage, setActivePage] = useState('dashboard')
    const [loading, setLoading] = useState(true)
    const [apiConnected, setApiConnected] = useState(false)
    const [toasts, setToasts] = useState([])
    const [wsConnected, setWsConnected] = useState(false)
    const [defectAnalysis, setDefectAnalysis] = useState(null)
    const [analyzingPanel, setAnalyzingPanel] = useState(null)
    const [selectedHeatmapPanel, setSelectedHeatmapPanel] = useState(null)
    const wsRef = useRef(null)
    const reconnectTimer = useRef(null)

    useEffect(() => {
        fetchData()
    }, [])

    // WebSocket connection
    useEffect(() => {
        if (!apiConnected) return

        const connectWs = () => {
            const ws = new WebSocket(WS_URL)

            ws.onopen = () => {
                setWsConnected(true)
                console.log('[WS] Connected')
            }

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data)
                    handleWsMessage(data)
                } catch (e) {
                    console.error('[WS] Parse error:', e)
                }
            }

            ws.onclose = () => {
                setWsConnected(false)
                console.log('[WS] Disconnected, reconnecting in 3s...')
                reconnectTimer.current = setTimeout(connectWs, 3000)
            }

            ws.onerror = () => {
                ws.close()
            }

            wsRef.current = ws
        }

        connectWs()

        // Ping every 30s to keep alive
        const pingInterval = setInterval(() => {
            if (wsRef.current?.readyState === WebSocket.OPEN) {
                wsRef.current.send('ping')
            }
        }, 30000)

        return () => {
            clearInterval(pingInterval)
            clearTimeout(reconnectTimer.current)
            wsRef.current?.close()
        }
    }, [apiConnected])

    const handleWsMessage = useCallback((data) => {
        if (data.type === 'panel_update') {
            // Single panel changed
            setPanels(prev => prev.map(p =>
                p.id === data.panel.id ? data.panel : p
            ))
            if (data.kpis) {
                setSiteData(prev => prev ? { ...prev, kpis: data.kpis, zone_health: data.zone_health } : prev)
            }
            // Show toast for defect detections
            if (data.panel.defect !== 'Clean' && data.panel.defect !== 'normal') {
                addToast(`🚨 ${data.panel.id}: ${data.panel.defect} detected!`, 'warning')
            }
        } else if (data.type === 'bulk_update') {
            // Bulk event (dust storm, maintenance, etc.)
            if (data.panels) {
                setPanels([...data.panels])
            }
            if (data.kpis) {
                setSiteData(prev => prev ? { ...prev, kpis: data.kpis, zone_health: data.zone_health } : prev)
            }
            const eventNames = {
                dust_storm: '☁️ Dust Storm',
                bird_event: '🐦 Bird Event',
                maintenance: '🧹 Maintenance Complete',
                reset: '🔄 Farm Reset',
            }
            addToast(`${eventNames[data.event] || data.event}: ${data.affected_count} panels affected`, 'info')
        }
    }, [])

    const addToast = (message, type = 'info') => {
        const id = Date.now()
        setToasts(prev => [...prev, { id, message, type }])
        setTimeout(() => {
            setToasts(prev => prev.filter(t => t.id !== id))
        }, 5000)
    }

    const fetchData = async () => {
        try {
            const [siteRes, panelsRes, recsRes, weatherRes] = await Promise.all([
                fetch(`${API_BASE}/api/site`),
                fetch(`${API_BASE}/api/panels`),
                fetch(`${API_BASE}/api/recommendations?limit=30`),
                fetch(`${API_BASE}/api/weather`),
            ])

            const site = await siteRes.json()
            const panelData = await panelsRes.json()
            const recData = await recsRes.json()
            const weatherData = await weatherRes.json()

            setSiteData(site)
            setPanels(panelData.panels)
            setRecommendations(recData.recommendations)
            setWeather(weatherData.forecast)
            setApiConnected(true)
        } catch (err) {
            console.log('API not available, using fallback data')
            loadFallbackData()
        } finally {
            setLoading(false)
        }
    }

    const loadFallbackData = () => {
        const zones = ['Zone A', 'Zone B', 'Zone C', 'Zone D', 'Zone E']
        const defectTypes = ['normal', 'micro_crack', 'hotspot', 'dust_soiling']
        const fallbackPanels = []

        for (let i = 0; i < 200; i++) {
            const row = Math.floor(i / 20)
            const col = i % 20
            const r = Math.random()
            const defect = r < 0.83 ? 'normal' : r < 0.89 ? 'micro_crack' : r < 0.94 ? 'hotspot' : 'dust_soiling'
            const severity = defect === 'normal' ? 0 : +(Math.random() * 0.7 + 0.2).toFixed(2)
            const maxOut = +(Math.random() * 0.7 + 4.8).toFixed(2)
            const eff = defect === 'normal' ? +(Math.random() * 0.08 + 0.92).toFixed(3) : +(Math.random() * 0.25 + 0.6).toFixed(3)
            const currOut = +(maxOut * eff).toFixed(2)
            const loss = +((maxOut - currOut) * 24).toFixed(2)

            fallbackPanels.push({
                id: `P-${(1000 + i).toString().padStart(4, '0')}`,
                row, col,
                zone: zones[Math.floor(row / 2)],
                defect, severity,
                confidence: defect === 'normal' ? +(Math.random() * 0.04 + 0.95).toFixed(2) : +(Math.random() * 0.17 + 0.82).toFixed(2),
                max_output_kw: maxOut,
                current_output_kw: currOut,
                efficiency: eff,
                energy_loss_kwh_day: loss,
                cost_loss_usd_day: +(loss * 0.08).toFixed(2),
                co2_impact_kg_day: +(loss * 0.42).toFixed(2),
                temperature_c: +(Math.random() * 30 + 35).toFixed(1),
                rul_days: defect === 'normal' ? 365 : Math.floor(Math.random() * 150 + 10),
                status: defect === 'normal' ? 'healthy' : severity > 0.7 ? 'critical' : 'warning',
                last_inspection: '2026-02-20',
            })
        }

        const healthy = fallbackPanels.filter(p => p.defect === 'normal').length
        const faulty = fallbackPanels.length - healthy
        const critical = fallbackPanels.filter(p => p.severity > 0.7).length

        setPanels(fallbackPanels)
        setSiteData({
            site_name: 'SolarMind Demo Farm',
            location: 'Hyderabad, Telangana',
            capacity_mw: 10,
            kpis: {
                precision: 0.962, recall: 0.938, f1_score: 0.950, mAP: 0.891,
                false_alarm_rate: 0.032,
                downtime_reduction_pct: 32.1, energy_yield_recovery_pct: 18.7,
                maintenance_cost_reduction_pct: 25.3,
                inference_latency_ms: 23.4, edge_uptime_pct: 99.7,
                recommendation_acceptance_pct: 84.3, co2_savings_tonnes_year: 1247,
                total_panels: 200, healthy_panels: healthy, faulty_panels: faulty, critical_alerts: critical,
            },
            zone_health: Object.fromEntries(zones.map(z => {
                const zp = fallbackPanels.filter(p => p.zone === z)
                const zh = zp.filter(p => p.defect === 'normal').length
                return [z, { total: zp.length, healthy: zh, health_pct: +(zh / zp.length * 100).toFixed(1) }]
            })),
        })

        const recs = fallbackPanels
            .filter(p => p.defect !== 'normal')
            .map(p => {
                const cps = +(0.25 * p.confidence * 0.85 + 0.25 * Math.min(1, p.energy_loss_kwh_day / 130) + 0.15 * 0.7 + 0.10 * 0.75 + 0.25 * (1 - p.rul_days / 365)).toFixed(3)
                let action, priority
                if (cps >= 0.8) { action = 'Replace panel within 72hrs'; priority = 'P1' }
                else if (cps >= 0.6) { action = p.defect === 'dust_soiling' ? 'Clean now' : 'Schedule repair'; priority = 'P2' }
                else if (cps >= 0.4) { action = 'Inspect within 48hrs'; priority = 'P3' }
                else { action = 'Recheck in 24hrs'; priority = 'P4' }
                return { panel_id: p.id, defect: p.defect, severity: p.severity, zone: p.zone, cps, action, priority, priority_label: priority, energy_loss_kwh_day: p.energy_loss_kwh_day, rul_days: p.rul_days, expected_yield_recovery_kwh: +(p.energy_loss_kwh_day * 0.82).toFixed(2) }
            })
            .sort((a, b) => b.cps - a.cps)
        setRecommendations(recs)

        const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
        const wf = Array.from({ length: 7 }, (_, i) => {
            const d = new Date(); d.setDate(d.getDate() + i)
            return {
                date: d.toISOString().slice(0, 10),
                day_name: dayNames[d.getDay()],
                temp_high_c: +(Math.random() * 12 + 30).toFixed(1),
                rain_probability: +(Math.random() * 0.4).toFixed(2),
                irradiance_forecast_w_m2: Math.round(Math.random() * 350 + 600),
            }
        })
        setWeather(wf)
        setApiConnected(false)
    }

    const handlePanelClick = async (panel) => {
        let detail = { panel, history: [], forecast: { trajectory: [] }, recommendation: {} }
        if (apiConnected) {
            try {
                const res = await fetch(`${API_BASE}/api/panels/${panel.id}`)
                detail = await res.json()
            } catch { /* use basic detail */ }
        } else {
            const hist = Array.from({ length: 6 }, (_, i) => {
                const d = new Date(); d.setDate(d.getDate() - (6 - i) * 30)
                const sev = panel.defect === 'normal' ? 0 : +(panel.severity * (0.5 + 0.5 * i / 5) + (Math.random() * 0.06 - 0.03)).toFixed(3)
                return { date: d.toISOString().slice(0, 10), severity: Math.max(0, Math.min(1, sev)), risk_score: +(Math.min(1, sev * 1.1)).toFixed(3), action_taken: ['Inspected', 'Cleaned', 'Monitored'][i % 3] }
            })
            const traj = Array.from({ length: 30 }, (_, i) => ({
                day: i * 3, severity: Math.min(1, +(panel.severity + 0.004 * i * 3).toFixed(3)), risk: Math.min(1, +(panel.severity * 1.08 + 0.004 * i * 3).toFixed(3))
            }))
            detail = { panel, history: hist, forecast: { trajectory: traj, rul_days: panel.rul_days, risk_level: panel.severity > 0.7 ? 'high' : 'medium' }, recommendation: { action: 'Inspect', priority: 'P3', cps: 0.5 } }
        }
        setSelectedPanel(detail)
    }

    if (loading) {
        return (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', background: '#0a0e17' }}>
                <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '3rem', marginBottom: 16 }}>☀️</div>
                    <div style={{ color: '#60a5fa', fontWeight: 600, fontSize: '1.1rem' }}>Loading SolarMind AI...</div>
                </div>
            </div>
        )
    }

    const analyzePanel = async (panel) => {
        setAnalyzingPanel(panel.id)
        setDefectAnalysis(null)
        try {
            const res = await fetch(`${API_BASE}/api/panels/${panel.id}/analyze`)
            const data = await res.json()
            setDefectAnalysis(data)
        } catch (err) {
            console.error('Panel analysis failed:', err)
            addToast('Analysis failed — is the backend running?', 'warning')
        } finally {
            setAnalyzingPanel(null)
        }
    }

    const pageTitles = {
        dashboard: '☀️ Solar Twin Dashboard',
        panels: '🔲 Panel Map',
        defects: '🔍 Defect Detection & Analysis',
        recommendations: '🛠️ Maintenance Recommendations',
        forecasting: '📈 Defect Forecasting',
        simulator: '🎛️ Panel Simulator',
        federation: '🌐 Federated Learning',
        model: '🤖 AI Model Info',
        settings: '⚙️ Settings',
    }

    const renderPage = () => {
        switch (activePage) {
            case 'panels':
                return (
                    <>
                        <div className="grid-full">
                            <PanelHeatmap panels={panels} onPanelClick={handlePanelClick} />
                        </div>
                        <div className="grid-bottom">
                            <ZoneHealth zones={siteData?.zone_health} />
                            <DefectDistribution panels={panels} />
                        </div>
                    </>
                )
            case 'defects':
                return (
                    <>
                        {/* Panel Health Map — click to select a panel */}
                        <div className="grid-full">
                            <PanelHeatmap panels={panels} onPanelClick={(panel) => {
                                setSelectedHeatmapPanel(panel)
                                setDefectAnalysis(null)
                            }} />
                        </div>

                        {/* Selected Panel Photo + Analyze Button */}
                        {selectedHeatmapPanel && (
                            <div className="grid-full">
                                <div className="card defect-analyze-card">
                                    <div className="defect-analyze-layout">
                                        {/* Panel Photo */}
                                        <div className="defect-panel-photo-section">
                                            {selectedHeatmapPanel.image_url ? (
                                                <img
                                                    src={`${API_BASE}${selectedHeatmapPanel.image_url}`}
                                                    alt={`${selectedHeatmapPanel.id}`}
                                                    className="defect-panel-photo"
                                                />
                                            ) : (
                                                <div className="defect-panel-photo-placeholder">📷 No image assigned</div>
                                            )}
                                            <div className="defect-panel-photo-info">
                                                <strong>{selectedHeatmapPanel.id}</strong> • Row {selectedHeatmapPanel.row}, Col {selectedHeatmapPanel.col} • {selectedHeatmapPanel.zone}
                                            </div>
                                        </div>
                                        {/* Info + Analyze */}
                                        <div className="defect-panel-info-section">
                                            <div className="defect-panel-stats">
                                                <div className="defect-stat">
                                                    <span className="defect-stat-label">Current Defect</span>
                                                    <span className="defect-stat-value" style={{ color: (selectedHeatmapPanel.defect === 'normal' || selectedHeatmapPanel.defect === 'Clean') ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                                                        {selectedHeatmapPanel.defect}
                                                    </span>
                                                </div>
                                                <div className="defect-stat">
                                                    <span className="defect-stat-label">Severity</span>
                                                    <span className="defect-stat-value">{(selectedHeatmapPanel.severity * 100).toFixed(0)}%</span>
                                                </div>
                                                <div className="defect-stat">
                                                    <span className="defect-stat-label">Confidence</span>
                                                    <span className="defect-stat-value">{(selectedHeatmapPanel.confidence * 100).toFixed(0)}%</span>
                                                </div>
                                            </div>
                                            <button
                                                className="analyze-defect-btn"
                                                onClick={() => analyzePanel(selectedHeatmapPanel)}
                                                disabled={!!analyzingPanel}
                                            >
                                                {analyzingPanel === selectedHeatmapPanel.id ? '🔄 Analyzing...' : '🔬 Analyze Defect'}
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Analysis Result (same as image upload result) */}
                        {defectAnalysis && (
                            <div className="grid-full">
                                <div className="card" style={{ padding: 20 }}>
                                    <div className="card-header" style={{ marginBottom: 16 }}>
                                        <span className="card-title">🔬 Analysis Result — {defectAnalysis.panel_id}</span>
                                        <span className="card-badge green">{defectAnalysis.filename}</span>
                                    </div>
                                    <div style={{ display: 'grid', gridTemplateColumns: '200px 1fr', gap: 20 }}>
                                        {defectAnalysis.image_url && (
                                            <img
                                                src={`${API_BASE}${defectAnalysis.image_url}`}
                                                alt="Analyzed panel"
                                                style={{ width: '100%', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}
                                            />
                                        )}
                                        <div>
                                            <div style={{ fontSize: '1.3rem', fontWeight: 800, color: defectAnalysis.classification?.predicted_class === 'Clean' ? 'var(--accent-green)' : 'var(--accent-red)', marginBottom: 8 }}>
                                                {defectAnalysis.classification?.predicted_class}
                                            </div>
                                            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: 12 }}>
                                                Confidence: <strong>{((defectAnalysis.classification?.confidence || 0) * 100).toFixed(1)}%</strong>
                                                {' • Mode: '}<strong>{defectAnalysis.classification?.mode}</strong>
                                            </div>
                                            {/* Probabilities */}
                                            {defectAnalysis.classification?.probabilities && (
                                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 12 }}>
                                                    {Object.entries(defectAnalysis.classification.probabilities)
                                                        .sort(([, a], [, b]) => b - a)
                                                        .map(([cls, prob]) => (
                                                            <span key={cls} style={{
                                                                padding: '4px 10px', borderRadius: 'var(--radius-pill)',
                                                                background: prob > 0.5 ? 'rgba(239,68,68,0.12)' : 'var(--bg-secondary)',
                                                                fontSize: '0.72rem', fontWeight: 600,
                                                                color: prob > 0.5 ? 'var(--accent-red)' : 'var(--text-muted)',
                                                                border: '1px solid var(--border-color)',
                                                            }}>
                                                                {cls}: {(prob * 100).toFixed(1)}%
                                                            </span>
                                                        ))}
                                                </div>
                                            )}
                                            {/* Sarvam AI Analysis */}
                                            {defectAnalysis.analysis?.report && (
                                                <div style={{ padding: 12, background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                                                    <div style={{ fontSize: '0.72rem', color: 'var(--accent-purple)', fontWeight: 700, marginBottom: 6 }}>🤖 Sarvam AI Analysis</div>
                                                    <div className="analysis-text" dangerouslySetInnerHTML={{ __html: defectAnalysis.analysis.report.replace(/\n/g, '<br/>') }} />
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* ViT Attention Map + Distribution */}
                        <div className="grid-bottom">
                            <AttentionMap panel={selectedHeatmapPanel || panels.find(p => p.defect !== 'normal' && p.defect !== 'Clean') || panels[0]} />
                            <DefectDistribution panels={panels} />
                        </div>
                    </>
                )
            case 'recommendations':
                return (
                    <>
                        <div className="grid-full">
                            <RecommendationQueue recommendations={recommendations} onItemClick={(rec) => {
                                const panel = panels.find(p => p.id === rec.panel_id)
                                if (panel) handlePanelClick(panel)
                            }} />
                        </div>
                        <div className="grid-bottom">
                            <EnergyImpact panels={panels} kpis={siteData?.kpis} />
                            <KPIMetrics kpis={siteData?.kpis} />
                        </div>
                    </>
                )
            case 'forecasting':
                return (
                    <>
                        <div className="grid-full">
                            <ProgressionChart panels={panels} />
                        </div>
                        <div className="grid-bottom">
                            <WeatherWidget forecast={weather} />
                            <EnergyImpact panels={panels} kpis={siteData?.kpis} />
                        </div>
                    </>
                )
            case 'simulator':
                return (
                    <div className="grid-full">
                        <PanelSimulator panels={panels} onPanelsChanged={fetchData} />
                    </div>
                )
            case 'federation':
                return (
                    <div className="grid-full">
                        <div className="card" style={{ textAlign: 'center', padding: '60px 40px' }}>
                            <div style={{ fontSize: '3rem', marginBottom: 16 }}>🌐</div>
                            <h3 style={{ marginBottom: 8 }}>Federated Learning Hub</h3>
                            <p style={{ color: 'var(--text-secondary)', maxWidth: 500, margin: '0 auto' }}>
                                Distributed model training across multiple solar farm edge nodes. Models are trained locally and aggregated centrally — no raw data leaves the site.
                            </p>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginTop: 24, maxWidth: 400, margin: '24px auto 0' }}>
                                <div className="card" style={{ padding: 16, textAlign: 'center' }}>
                                    <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--accent-green)' }}>5</div>
                                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Active Nodes</div>
                                </div>
                                <div className="card" style={{ padding: 16, textAlign: 'center' }}>
                                    <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--accent-blue)' }}>98.2%</div>
                                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Sync Rate</div>
                                </div>
                                <div className="card" style={{ padding: 16, textAlign: 'center' }}>
                                    <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--accent-cyan)' }}>v2.1</div>
                                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Model Ver.</div>
                                </div>
                            </div>
                        </div>
                    </div>
                )
            case 'model':
                return (
                    <div className="grid-full">
                        <div className="card" style={{ padding: 28 }}>
                            <h3 style={{ marginBottom: 16 }}>🤖 AI Model Architecture</h3>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                                <div className="card" style={{ padding: 20 }}>
                                    <h4 style={{ color: 'var(--accent-blue)', marginBottom: 12 }}>ViT-Base/16 Classifier</h4>
                                    <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.8 }}>
                                        <div>• Architecture: Vision Transformer (ViT-Base/16)</div>
                                        <div>• Parameters: 86M</div>
                                        <div>• Input: 224×224 RGB images</div>
                                        <div>• Classes: Bird-drop, Clean, Dusty, Electrical-damage, Physical-Damage, Snow-Covered</div>
                                        <div>• Dataset: <a href="https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset" target="_blank" rel="noreferrer" style={{ color: 'var(--accent-blue)' }}>PV Panel Defect Dataset</a></div>
                                        <div>• Inference: ~23ms (edge), ~8ms (cloud)</div>
                                    </div>
                                </div>
                                <div className="card" style={{ padding: 20 }}>
                                    <h4 style={{ color: 'var(--accent-purple)', marginBottom: 12 }}>Sarvam AI Analysis</h4>
                                    <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.8 }}>
                                        <div>• Model: sarvam-m (24B parameters)</div>
                                        <div>• Task: Maintenance report generation</div>
                                        <div>• Input: Classification results + panel metadata</div>
                                        <div>• Output: Diagnosis, severity, action plan</div>
                                        <div>• Languages: English + Indian languages</div>
                                        <div>• API: <a href="https://sarvam.ai" target="_blank" rel="noreferrer" style={{ color: 'var(--accent-purple)' }}>sarvam.ai</a></div>
                                    </div>
                                </div>
                            </div>
                            <div className="card" style={{ padding: 20, marginTop: 16 }}>
                                <h4 style={{ color: 'var(--accent-green)', marginBottom: 12 }}>Performance Metrics</h4>
                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
                                    {[
                                        { label: 'Precision', value: siteData?.kpis?.precision || 0.962, color: 'var(--accent-green)' },
                                        { label: 'Recall', value: siteData?.kpis?.recall || 0.938, color: 'var(--accent-blue)' },
                                        { label: 'F1 Score', value: siteData?.kpis?.f1_score || 0.950, color: 'var(--accent-cyan)' },
                                        { label: 'mAP', value: siteData?.kpis?.mAP || 0.891, color: 'var(--accent-yellow)' },
                                    ].map(m => (
                                        <div key={m.label} style={{ textAlign: 'center', padding: 12, background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)' }}>
                                            <div style={{ fontSize: '1.5rem', fontWeight: 800, color: m.color }}>{(m.value * 100).toFixed(1)}%</div>
                                            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 4 }}>{m.label}</div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                )
            case 'settings':
                return (
                    <div className="grid-full">
                        <div className="card" style={{ textAlign: 'center', padding: '60px 40px' }}>
                            <div style={{ fontSize: '3rem', marginBottom: 16 }}>⚙️</div>
                            <h3 style={{ marginBottom: 8 }}>System Settings</h3>
                            <p style={{ color: 'var(--text-secondary)', maxWidth: 500, margin: '0 auto' }}>
                                Configure alert thresholds, API keys, edge node settings, and notification preferences.
                            </p>
                            <div style={{ marginTop: 24, display: 'flex', flexDirection: 'column', gap: 12, maxWidth: 400, margin: '24px auto 0' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '12px 16px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)' }}>
                                    <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Backend API</span>
                                    <span style={{ color: apiConnected ? 'var(--accent-green)' : 'var(--accent-red)', fontSize: '0.85rem', fontWeight: 600 }}>{apiConnected ? 'Connected' : 'Disconnected'}</span>
                                </div>
                                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '12px 16px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)' }}>
                                    <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>WebSocket</span>
                                    <span style={{ color: wsConnected ? 'var(--accent-green)' : 'var(--accent-red)', fontSize: '0.85rem', fontWeight: 600 }}>{wsConnected ? 'Live' : 'Disconnected'}</span>
                                </div>
                                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '12px 16px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)' }}>
                                    <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Total Panels</span>
                                    <span style={{ color: 'var(--text-primary)', fontSize: '0.85rem', fontWeight: 600 }}>{panels.length}</span>
                                </div>
                                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '12px 16px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)' }}>
                                    <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Sarvam AI</span>
                                    <span style={{ color: 'var(--accent-yellow)', fontSize: '0.85rem', fontWeight: 600 }}>Set SARVAM_API_KEY</span>
                                </div>
                            </div>
                        </div>
                    </div>
                )
            default: // dashboard
                return (
                    <>
                        <div className="grid-top">
                            <SiteOverview kpis={siteData?.kpis} />
                        </div>
                        <div className="grid-full">
                            <ImageUpload />
                        </div>
                        <div className="grid-main">
                            <RecommendationQueue recommendations={recommendations} onItemClick={(rec) => {
                                const panel = panels.find(p => p.id === rec.panel_id)
                                if (panel) handlePanelClick(panel)
                            }} />
                            <DefectDistribution panels={panels} />
                        </div>
                        <div className="grid-bottom">
                            <EnergyImpact panels={panels} kpis={siteData?.kpis} />
                            <ZoneHealth zones={siteData?.zone_health} />
                        </div>
                        <div className="grid-bottom">
                            <KPIMetrics kpis={siteData?.kpis} />
                            <ProgressionChart panels={panels} />
                        </div>
                        <div className="grid-full">
                            <WeatherWidget forecast={weather} />
                        </div>
                    </>
                )
        }
    }

    return (
        <div className="app-layout">
            <Sidebar activePage={activePage} onNavigate={setActivePage} apiConnected={apiConnected} />
            <main className="main-content">
                <div className="page-header">
                    <div>
                        <h2>{pageTitles[activePage] || '☀️ Solar Twin Dashboard'}</h2>
                        <div className="breadcrumb">{siteData?.site_name} • {siteData?.location} • {siteData?.capacity_mw} MW</div>
                    </div>
                    <div className="header-actions">
                        <span className={`header-badge ${apiConnected ? 'live' : 'demo'}`}>
                            {apiConnected ? '⚡ Live' : '📡 Demo Mode'}
                        </span>
                        {wsConnected && <span className="header-badge live">🔌 WebSocket</span>}
                        {apiConnected && <span className="header-badge live">Live Monitoring</span>}
                    </div>
                </div>

                {renderPage()}

                {/* Panel Detail Modal */}
                {selectedPanel && (
                    <PanelDetail
                        detail={selectedPanel}
                        onClose={() => setSelectedPanel(null)}
                    />
                )}

                {/* Toast Notifications */}
                <div className="toast-container">
                    {toasts.map(toast => (
                        <div key={toast.id} className={`toast toast-${toast.type}`}>
                            <span>{toast.message}</span>
                            <button className="toast-close" onClick={() => setToasts(prev => prev.filter(t => t.id !== toast.id))}>×</button>
                        </div>
                    ))}
                </div>
            </main>
        </div>
    )
}

export default App
