export default function WeatherWidget({ forecast }) {
    if (!forecast || forecast.length === 0) return null

    const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

    const getIcon = (day) => {
        if (day.rain_probability > 0.5) return '🌧️'
        if (day.rain_probability > 0.25) return '⛅'
        return '☀️'
    }

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">🌤️ 7-Day Weather Forecast</span>
                <span className="card-badge blue">Cleaning Planner</span>
            </div>
            <div className="weather-grid">
                {forecast.map((day, i) => {
                    const d = new Date(day.date)
                    const dayName = day.day_name || dayNames[d.getDay()]
                    return (
                        <div key={i} className="weather-day">
                            <div className="day-name">{dayName}</div>
                            <div className="weather-icon">{getIcon(day)}</div>
                            <div className="temp">{day.temp_high_c}°C</div>
                            <div style={{ fontSize: '0.65rem', color: day.rain_probability > 0.3 ? 'var(--accent-yellow)' : 'var(--text-muted)', marginTop: 4 }}>
                                🌧 {(day.rain_probability * 100).toFixed(0)}%
                            </div>
                            <div style={{ fontSize: '0.62rem', color: 'var(--text-muted)', marginTop: 2 }}>
                                ☀️ {day.irradiance_forecast_w_m2} W
                            </div>
                        </div>
                    )
                })}
            </div>
        </div>
    )
}
