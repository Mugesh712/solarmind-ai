import SiteOverview from '../components/SiteOverview'
import PanelHeatmap from '../components/PanelHeatmap'
import RecommendationQueue from '../components/RecommendationQueue'
import EnergyImpact from '../components/EnergyImpact'
import KPIMetrics from '../components/KPIMetrics'
import ZoneHealth from '../components/ZoneHealth'
import DefectDistribution from '../components/DefectDistribution'
import WeatherWidget from '../components/WeatherWidget'
import ProgressionChart from '../components/ProgressionChart'
import AttentionMap from '../components/AttentionMap'
import ImageUpload from '../components/ImageUpload'

export default function Dashboard({
    siteData, panels, recommendations, weather,
    onPanelClick, onRecommendationClick,
}) {
    // Select a representative faulty panel for the attention map
    const faultyPanel = panels.find(p => p.defect !== 'normal') || panels[0]

    return (
        <>
            {/* KPI Cards Row */}
            <div className="grid-top">
                <SiteOverview kpis={siteData?.kpis} />
            </div>

            {/* Image Upload Analyzer */}
            <div className="grid-full">
                <ImageUpload />
            </div>

            {/* Main Grid: Heatmap + Recommendations */}
            <div className="grid-main">
                <PanelHeatmap panels={panels} onPanelClick={onPanelClick} />
                <RecommendationQueue
                    recommendations={recommendations}
                    onItemClick={onRecommendationClick}
                />
            </div>

            {/* Impact + Zone Health */}
            <div className="grid-bottom">
                <EnergyImpact panels={panels} kpis={siteData?.kpis} />
                <ZoneHealth zones={siteData?.zone_health} />
            </div>

            {/* KPI + Defect Distribution */}
            <div className="grid-bottom">
                <KPIMetrics kpis={siteData?.kpis} />
                <DefectDistribution panels={panels} />
            </div>

            {/* Progression + Attention Map */}
            <div className="grid-bottom">
                <ProgressionChart panels={panels} />
                <AttentionMap panel={faultyPanel} />
            </div>

            {/* Weather */}
            <div className="grid-full">
                <WeatherWidget forecast={weather} />
            </div>
        </>
    )
}
