export default function ZoneHealth({ zones }) {
    if (!zones) return null

    const getBarClass = (pct) => {
        if (pct >= 95) return 'excellent'
        if (pct >= 85) return 'good'
        if (pct >= 70) return 'warning'
        return 'critical'
    }

    const getPctColor = (pct) => {
        if (pct >= 95) return 'var(--accent-green)'
        if (pct >= 85) return 'var(--accent-blue)'
        if (pct >= 70) return 'var(--accent-yellow)'
        return 'var(--accent-red)'
    }

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">🏗️ Zone Health</span>
            </div>
            <div className="zone-list">
                {Object.entries(zones).map(([name, data]) => (
                    <div key={name} className="zone-item">
                        <span className="zone-name">{name}</span>
                        <div className="zone-bar-container">
                            <div
                                className={`zone-bar ${getBarClass(data.health_pct)}`}
                                style={{ width: `${data.health_pct}%` }}
                            />
                        </div>
                        <span className="zone-pct" style={{ color: getPctColor(data.health_pct) }}>
                            {data.health_pct}%
                        </span>
                    </div>
                ))}
            </div>
        </div>
    )
}
