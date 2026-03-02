import { useState, useRef } from 'react';

const API_URL = 'http://localhost:8000';

export default function ImageUpload() {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [dragOver, setDragOver] = useState(false);
    const fileRef = useRef(null);

    const handleFile = (f) => {
        if (!f) return;
        const validTypes = ['image/jpeg', 'image/png', 'image/bmp'];
        if (!validTypes.includes(f.type)) {
            setError('Please upload a JPG, PNG, or BMP image.');
            return;
        }
        if (f.size > 10 * 1024 * 1024) {
            setError('File too large. Max 10MB.');
            return;
        }
        setFile(f);
        setError(null);
        setResult(null);
        const reader = new FileReader();
        reader.onload = (e) => setPreview(e.target.result);
        reader.readAsDataURL(f);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setDragOver(false);
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    };

    const handleAnalyze = async () => {
        if (!file) return;
        setLoading(true);
        setError(null);
        try {
            const formData = new FormData();
            formData.append('file', file);
            const res = await fetch(`${API_URL}/api/analyze`, {
                method: 'POST',
                body: formData,
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Analysis failed');
            }
            const data = await res.json();
            setResult(data);
        } catch (err) {
            setError(err.message || 'Failed to analyze image');
        } finally {
            setLoading(false);
        }
    };

    const handleReset = () => {
        setFile(null);
        setPreview(null);
        setResult(null);
        setError(null);
    };

    const getSeverityColor = (text) => {
        if (!text) return '#6b7280';
        const lower = text.toLowerCase();
        if (lower.includes('critical')) return '#ef4444';
        if (lower.includes('high')) return '#f97316';
        if (lower.includes('medium')) return '#eab308';
        if (lower.includes('low')) return '#22c55e';
        return '#6b7280';
    };

    const getClassIcon = (cls) => {
        const icons = {
            'Bird-drop': '🐦',
            'Clean': '✅',
            'Dusty': '🌫️',
            'Electrical-damage': '⚡',
            'Physical-Damage': '💥',
            'Snow-Covered': '❄️',
        };
        return icons[cls] || '🔍';
    };

    return (
        <div className="image-upload-card">
            <div className="card-header">
                <h3>🔬 Panel Defect Analyzer</h3>
                <p className="subtitle">Upload a solar panel image for AI-powered defect analysis</p>
                <p className="dataset-badge">
                    📊 Powered by <a href="https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset" target="_blank" rel="noreferrer">PV Panel Defect Dataset</a> + <strong>Sarvam AI</strong>
                </p>
            </div>

            {/* Upload Zone */}
            {!result && (
                <div
                    className={`upload-zone ${dragOver ? 'drag-over' : ''} ${preview ? 'has-preview' : ''}`}
                    onDrop={handleDrop}
                    onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                    onDragLeave={() => setDragOver(false)}
                    onClick={() => !preview && fileRef.current?.click()}
                >
                    <input
                        ref={fileRef}
                        type="file"
                        accept=".jpg,.jpeg,.png,.bmp"
                        style={{ display: 'none' }}
                        onChange={(e) => handleFile(e.target.files?.[0])}
                    />
                    {preview ? (
                        <div className="preview-container">
                            <img src={preview} alt="Preview" className="preview-image" />
                            <div className="preview-info">
                                <span className="filename">{file?.name}</span>
                                <span className="filesize">{(file?.size / 1024).toFixed(1)} KB</span>
                            </div>
                        </div>
                    ) : (
                        <div className="upload-placeholder">
                            <span className="upload-icon">📤</span>
                            <p>Drag & drop a solar panel image here</p>
                            <p className="upload-hint">or click to browse (JPG, PNG, BMP — Max 10MB)</p>
                        </div>
                    )}
                </div>
            )}

            {/* Action Buttons */}
            {!result && (
                <div className="upload-actions">
                    {preview && (
                        <>
                            <button
                                className="btn-analyze"
                                onClick={handleAnalyze}
                                disabled={loading}
                            >
                                {loading ? (
                                    <><span className="spinner"></span> Analyzing...</>
                                ) : (
                                    '🔍 Analyze Defect'
                                )}
                            </button>
                            <button className="btn-reset" onClick={handleReset}>Clear</button>
                        </>
                    )}
                </div>
            )}

            {/* Error */}
            {error && (
                <div className="upload-error">⚠️ {error}</div>
            )}

            {/* Results */}
            {result && (
                <div className="analysis-results">
                    <div className="result-header">
                        <div className="predicted-class">
                            <span className="class-icon">{getClassIcon(result.classification?.predicted_class)}</span>
                            <div>
                                <h4>{result.classification?.predicted_class}</h4>
                                <span className="confidence-badge">
                                    {(result.classification?.confidence * 100).toFixed(1)}% confidence
                                </span>
                            </div>
                        </div>
                        <span className={`mode-badge ${result.classification?.mode}`}>
                            {result.classification?.mode === 'real' ? '🤖 Real Model' : '🔮 Simulated'}
                        </span>
                    </div>

                    {/* Probability Bars */}
                    <div className="probability-bars">
                        <h5>Class Probabilities</h5>
                        {result.classification?.probabilities &&
                            Object.entries(result.classification.probabilities)
                                .sort(([, a], [, b]) => b - a)
                                .map(([cls, prob]) => (
                                    <div key={cls} className="prob-bar-row">
                                        <span className="prob-label">
                                            {getClassIcon(cls)} {cls}
                                        </span>
                                        <div className="prob-bar-track">
                                            <div
                                                className="prob-bar-fill"
                                                style={{
                                                    width: `${Math.max(1, prob * 100)}%`,
                                                    backgroundColor:
                                                        cls === result.classification.predicted_class
                                                            ? '#3b82f6'
                                                            : '#475569',
                                                }}
                                            ></div>
                                        </div>
                                        <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
                                    </div>
                                ))}
                    </div>

                    {/* AI Analysis */}
                    <div className="ai-analysis">
                        <h5>
                            {result.analysis?.source === 'sarvam-ai' ? '🧠 Sarvam AI Analysis' : '📋 Analysis Report'}
                        </h5>
                        <div className="analysis-text">
                            {result.analysis?.analysis?.split('\n').map((line, i) => (
                                <p key={i}>{line.replace(/\*\*/g, '')}</p>
                            ))}
                        </div>
                        <span className="analysis-source">
                            Source: {result.analysis?.source === 'sarvam-ai' ? 'Sarvam AI (sarvam-m)' : 'Built-in Template'}
                        </span>
                    </div>

                    <button className="btn-reset" onClick={handleReset} style={{ marginTop: '16px', width: '100%' }}>
                        🔄 Analyze Another Image
                    </button>
                </div>
            )}
        </div>
    );
}
