export default function RecommendationQueue({ recommendations, onItemClick }) {
    if (!recommendations || recommendations.length === 0) return null

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">🛠️ Maintenance Queue</span>
                <span className="card-badge blue">{recommendations.length} Actions</span>
            </div>
            <div className="rec-list">
                {recommendations.slice(0, 15).map((rec, i) => (
                    <div key={i} className="rec-item" onClick={() => onItemClick(rec)}>
                        <span className={`rec-priority ${rec.priority}`}>{rec.priority}</span>
                        <div className="rec-info">
                            <div className="rec-panel-id">
                                {rec.panel_id}
                                <span style={{ color: 'var(--text-muted)', fontWeight: 400, fontSize: '0.72rem', marginLeft: 8 }}>
                                    {rec.zone}
                                </span>
                            </div>
                            <div className="rec-action">{rec.action}</div>
                        </div>
                        <div className="rec-cps">{rec.cps.toFixed(2)}</div>
                    </div>
                ))}
            </div>
        </div>
    )
}
