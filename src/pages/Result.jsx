import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { CheckCircle2, AlertTriangle, MapPin, Phone, Home } from 'lucide-react';
import Button from '../components/Button';
import { useTheme } from '../context/ThemeContext';

const Result = () => {
    const { state } = useLocation();
    const navigate = useNavigate();
    const { isDarkMode } = useTheme();
    const risk = state?.risk || 'Low';
    const confidence = state?.confidence || 0.15; // Default value

    const isHighRisk = risk === 'High';

    React.useEffect(() => {
        // Save result to localStorage for Dashboard
        try {
            const saved = localStorage.getItem('respi_user');
            const data = saved ? JSON.parse(saved) : {};

            // Convert confidence to percentage (0-100)
            const score = Math.round(confidence * 100);

            // Save as 'healthScore' to match Dashboard expectation
            const updated = { ...data, healthScore: score };
            localStorage.setItem('respi_user', JSON.stringify(updated));
        } catch (e) {
            console.error("Failed to save result:", e);
        }
    }, [confidence]);

    let theme;
    if (risk === 'High') {
        theme = {
            color: isDarkMode ? 'text-red-400' : 'text-red-600',
            bg: isDarkMode ? 'bg-red-900/20 shadow-red-900/10' : 'bg-red-50 shadow-xl',
            ring: isDarkMode ? 'ring-red-900/30' : 'ring-white',
            border: isDarkMode ? 'border-red-800' : 'border-red-100',
            lucide: 'text-red-500',
            message: 'Risiko Tinggi Terdeteksi',
            description: 'Indikasi pola batuk tidak wajar. Disarankan pemeriksaan medis lanjut.',
            icon: <AlertTriangle size={64} className="text-red-500" />
        };
    } else if (risk === 'Medium') {
        theme = {
            color: isDarkMode ? 'text-amber-400' : 'text-amber-600',
            bg: isDarkMode ? 'bg-amber-900/20 shadow-amber-900/10' : 'bg-amber-50 shadow-xl',
            ring: isDarkMode ? 'ring-amber-900/30' : 'ring-white',
            border: isDarkMode ? 'border-amber-800' : 'border-amber-100',
            lucide: 'text-amber-500',
            message: 'Resiko Sedang',
            description: 'Pola pernapasan menunjukan kejanggalan ringan. Pantau terus kesehatan Anda.',
            icon: <AlertTriangle size={64} className="text-amber-500" />
        };
    } else {
        theme = {
            color: isDarkMode ? 'text-green-400' : 'text-green-600',
            bg: isDarkMode ? 'bg-green-900/20 shadow-green-900/10' : 'bg-green-50 shadow-xl',
            ring: isDarkMode ? 'ring-green-900/30' : 'ring-white',
            border: isDarkMode ? 'border-green-800' : 'border-green-100',
            lucide: 'text-green-500',
            message: 'Resiko Rendah',
            description: 'Pola pernapasan normal. Tetap jaga kesehatan dan lakukan skrining rutin.',
            icon: <CheckCircle2 size={64} className="text-green-500" />
        };
    }

    const handleNearestHospital = () => {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition((position) => {
                const { latitude, longitude } = position.coords;
                // Query for "puskesmas" around the user's location
                const url = `https://www.google.com/maps/search/puskesmas/@${latitude},${longitude},15z`;
                window.open(url, '_blank');
            }, (error) => {
                console.error("Geolocation error:", error);
                // Fallback if permission denied
                const url = `https://www.google.com/maps/search/puskesmas+terdekat`;
                window.open(url, '_blank');
            });
        } else {
            const url = `https://www.google.com/maps/search/puskesmas+terdekat`;
            window.open(url, '_blank');
        }
    };

    return (
        <div className={`min-h-screen flex flex-col p-6 transition-colors duration-300 ${isDarkMode ? 'bg-slate-900' : 'bg-hospital-blue-50'}`}>
            <div className={`flex-1 flex flex-col items-center justify-center text-center`}>
                {/* Circular Progress Gauge */}
                <div className="relative w-48 h-48 mb-6 flex items-center justify-center">
                    {/* Background Circle */}
                    <svg className="w-full h-full transform -rotate-90">
                        <circle
                            cx="96"
                            cy="96"
                            r="88"
                            stroke="currentColor"
                            strokeWidth="12"
                            fill="transparent"
                            className={`${isDarkMode ? 'text-slate-800' : 'text-slate-200'}`}
                        />
                        {/* Progress Circle */}
                        <circle
                            cx="96"
                            cy="96"
                            r="88"
                            stroke="currentColor"
                            strokeWidth="12"
                            fill="transparent"
                            strokeDasharray={2 * Math.PI * 88}
                            strokeDashoffset={2 * Math.PI * 88 * (1 - confidence)}
                            strokeLinecap="round"
                            className={`${risk === 'High' ? 'text-red-500' : (risk === 'Medium' ? 'text-amber-500' : 'text-green-500')} transition-all duration-1000 ease-out`}
                        />
                    </svg>

                    {/* Center Text */}
                    <div className="absolute inset-0 flex flex-col items-center justify-center">
                        <span className={`text-5xl font-bold ${theme.color}`}>
                            {Math.round(confidence * 100)}%
                        </span>
                        <span className={`text-xs font-semibold mt-1 tracking-widest ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                            RASIO RESIKO
                        </span>
                    </div>
                </div>

                <h2 className={`text-2xl font-bold ${theme.color} mb-3`}>{theme.message}</h2>

                <p className={`leading-relaxed max-w-xs mb-8 text-sm ${isDarkMode ? 'text-slate-400' : 'text-hospital-blue-600'}`}>
                    {theme.description}
                </p>

                {/* Action Card */}
                {isHighRisk && (
                    <div className={`w-full rounded-2xl p-4 shadow-lg border mb-6 transition-colors ${isDarkMode ? 'bg-slate-800 border-red-900/50' : 'bg-white border-red-100'}`}>
                        <h3 className={`font-bold text-left mb-3 text-sm ${isDarkMode ? 'text-white' : 'text-hospital-blue-900'}`}>Rekomendasi Tindakan</h3>

                        <div className="space-y-3">
                            <button
                                onClick={handleNearestHospital}
                                className={`w-full flex items-center justify-between p-3 rounded-xl transition-colors group ${isDarkMode ? 'bg-slate-700/50 hover:bg-slate-700' : 'bg-hospital-blue-50 hover:bg-hospital-blue-100'}`}
                            >
                                <div className="flex items-center gap-3">
                                    <div className={`p-2 rounded-lg ${isDarkMode ? 'bg-blue-900/50 text-blue-400' : 'bg-blue-100 text-hospital-blue-600'}`}>
                                        <MapPin size={20} />
                                    </div>
                                    <div className="text-left">
                                        <p className={`text-sm font-semibold ${isDarkMode ? 'text-white' : 'text-hospital-blue-900'}`}>Puskesmas Terdekat</p>
                                        <p className={`text-xs ${isDarkMode ? 'text-slate-400' : 'text-hospital-blue-500'}`}>Ketuk untuk buka Peta</p>
                                    </div>
                                </div>
                                <div className={`text-xs font-bold px-2 py-1 rounded border ${isDarkMode ? 'bg-slate-600 text-blue-300 border-slate-500' : 'bg-white text-hospital-blue-600 border-hospital-blue-200'}`}>NAVIGASI</div>
                            </button>

                            <button className={`w-full flex items-center justify-between p-3 rounded-xl transition-colors ${isDarkMode ? 'bg-slate-700/50 hover:bg-slate-700' : 'bg-hospital-blue-50 hover:bg-hospital-blue-100'}`}>
                                <div className="flex items-center gap-3">
                                    <div className={`p-2 rounded-lg ${isDarkMode ? 'bg-green-900/50 text-green-400' : 'bg-green-100 text-green-600'}`}>
                                        <Phone size={20} />
                                    </div>
                                    <div className="text-left">
                                        <p className={`text-sm font-semibold ${isDarkMode ? 'text-white' : 'text-hospital-blue-900'}`}>Konsultasi Dokter</p>
                                        <p className={`text-xs ${isDarkMode ? 'text-slate-400' : 'text-hospital-blue-500'}`}>HaloDoc / Alodokter</p>
                                    </div>
                                </div>
                            </button>
                        </div>
                    </div>
                )}
            </div>

            <Button onClick={() => navigate('/dashboard')} variant="secondary" className={`w-full mb-2 ${isDarkMode ? 'bg-slate-800 text-white border-slate-700 hover:bg-slate-700' : ''}`}>
                <Home className="mr-2" size={20} />
                Kembali ke Beranda
            </Button>

            <p className={`text-[10px] text-center max-w-sm mx-auto opacity-60 leading-tight mt-4 px-4 ${isDarkMode ? 'text-slate-500' : 'text-slate-400'}`}>
                <strong>Harap Diperhatikan:</strong> Ini hanyalah hasil skrining awal dan bukan diagnosis medis. Akurasi tidak 100%. Konsultasikan dengan dokter untuk pemeriksaan lebih lanjut.
            </p>
        </div>
    );
};

export default Result;
