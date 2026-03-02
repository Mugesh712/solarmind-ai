export default function SiteOverview({ kpis }) {
    if (!kpis) return null

    const cards = [
        {
            icon: '🔲',
            iconColor: 'blue',
            value: kpis.total_panels,
            label: 'Total Panels',
            valueColor: 'blue',
            trend: null,
        },
        {
            icon: '✅',
            iconColor: 'green',
            value: `${kpis.healthy_panels}`,
            label: 'Healthy Panels',
            valueColor: 'green',
            trend: { direction: 'up', text: `${((kpis.healthy_panels / kpis.total_panels) * 100).toFixed(1)}%` },
        },
        {
            icon: '⚠️',
            iconColor: 'yellow',
            value: kpis.faulty_panels,
            label: 'Faulty Panels',
            valueColor: 'yellow',
            trend: null,
        },
        {
            icon: '🔴',
            iconColor: 'red',
            value: kpis.critical_alerts,
            label: 'Critical Alerts',
            valueColor: 'red',
            trend: { direction: 'down', text: 'Action needed' },
        },
    ]

    return (
        <>
            {cards.map((card, i) => (
                <div key={i} className="card kpi-card">
                    <div className={`kpi-icon ${card.iconColor}`}>{card.icon}</div>
                    <div className={`kpi-value ${card.valueColor} animate-number`}>{card.value}</div>
                    <div className="kpi-label">{card.label}</div>
                    {card.trend && (
                        <span className={`kpi-trend ${card.trend.direction}`}>
                            {card.trend.direction === 'up' ? '↑' : '↓'} {card.trend.text}
                        </span>
                    )}
                </div>
            ))}
        </>
    )
}
