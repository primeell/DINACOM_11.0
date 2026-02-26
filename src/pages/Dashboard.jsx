import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Wind, Activity, Bell, ChevronRight, Play, MapPin } from 'lucide-react';
import BottomNav from '../components/BottomNav';
import HealthGauge from '../components/HealthGauge';
import SparkLine from '../components/SparkLine';
import { getAirQualityByLocation, getCoordinatesByCity } from '../services/airQualityService';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar, CartesianGrid } from 'recharts';
import { useTheme } from '../context/ThemeContext';
import { useFitness } from '../context/FitnessContext';
import LungMascot from '../components/LungMascot';

const Dashboard = () => {
    const navigate = useNavigate();
    const { isDarkMode } = useTheme();
    const { runGoalMet, runTargetKm, runCurrentKm } = useFitness();
    const [showNotifications, setShowNotifications] = useState(false);

    const [user] = useState(() => {
        try {
            const saved = localStorage.getItem('respi_user');
            const initial = saved ? JSON.parse(saved) : { name: 'Guest' };
            // Defaults first, then spread saved data to allow overrides
            return { name: 'Guest', healthScore: 20, ...initial };
        } catch (e) {
            console.error("Failed to parse user data:", e);
            return { name: 'Guest', healthScore: 85 };
        }
    });

    const [airQuality, setAirQuality] = useState({
        aqi: 0,
        city: 'Detecting...',
        country: '',
        status: 'Loading',
        color: 'bg-gray-100 text-gray-600'
    });

    React.useEffect(() => {
        const fetchAQI = async (lat, lon) => {
            const data = await getAirQualityByLocation(lat, lon);
            let status = 'Baik';
            let color = 'bg-green-100 text-green-700';


            if (data.aqi > 50) { status = 'Sedang'; color = 'bg-yellow-100 text-yellow-700'; }
            if (data.aqi > 100) { status = 'Tidak Sehat Bagi Kelompok Sensitif'; color = 'bg-orange-100 text-orange-700'; }
            if (data.aqi > 150) { status = 'Tidak Sehat'; color = 'bg-red-100 text-red-700'; }
            if (data.aqi > 200) { status = 'Sangat Tidak Sehat'; color = 'bg-purple-100 text-purple-700'; }

            setAirQuality({
                aqi: data.aqi,
                city: data.city,
                country: data.country,
                status,
                color
            });
        };

        const initAirQuality = async () => {
            if (user.location) {
                const coords = await getCoordinatesByCity(user.location);
                if (coords) {
                    fetchAQI(coords.lat, coords.lon);
                } else {
                    // Fallback if city not found
                    fetchAQI(0, 0);
                }
            } else if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(
                    (position) => fetchAQI(position.coords.latitude, position.coords.longitude),
                    () => fetchAQI(0, 0)
                );
            } else {
                fetchAQI(0, 0);
            }
        };

        initAirQuality();
    }, [user.location]);

    // Mock Data
    // Dynamic Trends Data based on current Health Score (Risk Score)
    const riskScore = user.healthScore || 20;
    const isHighRisk = riskScore > 70;

    // Generate mock history that trends towards the current score
    const trendsData = React.useMemo(() => {
        const data = [];
        const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

        let current = riskScore;
        // Generate reverse from today
        for (let i = 6; i >= 0; i--) {
            // Add some random variance but keep trend
            const variance = Math.random() * 20 - 10; // +/- 10
            let score = current + variance;
            // Clamp between 0 and 100
            score = Math.max(0, Math.min(100, score));

            data.unshift({ name: days[i], score: Math.round(score) });

            // Move "current" slightly towards a neutral baseline for past days
            // so it looks like it evolved to the current state
            current = current + (Math.random() * 10 - 5);
        }
        // Force last point to match current score exactly
        data[6].score = riskScore;
        return data;
    }, [riskScore]);

    const chartColor = isHighRisk ? "#ef4444" : "#10b981"; // Red vs Emerald



    const respRateData = [16, 17, 16, 18, 17, 19, 18];
    const aqiData = [40, 45, 42, 50, 48, 55, 60];

    // Framer Motion Variants
    const containerVariants = {
        hidden: { opacity: 0 },
        visible: {
            opacity: 1,
            transition: { staggerChildren: 0.1 }
        }
    };

    const itemVariants = {
        hidden: { y: 20, opacity: 0 },
        visible: { y: 0, opacity: 1 }
    };

    return (
        <div className={`min-h-screen relative pb-32 overflow-hidden font-sans transition-colors duration-300 ${isDarkMode ? 'bg-hospital-blue-950' : 'bg-hospital-blue-900'}`}>
            <div
                className="fixed inset-0 w-full h-full -z-20 bg-cover bg-center bg-no-repeat opacity-50"
                style={{ backgroundImage: "url('/hospital-bg.png')" }}
            />

            {/* Dark Overlay to ensure readability */}
            <div className={`fixed inset-0 w-full h-full -z-10 transition-colors duration-300 ${isDarkMode ? 'bg-hospital-blue-950/90' : 'bg-hospital-blue-900/70'}`} />

            {/* Ambient Decorations */}
            <div className={`absolute top-[-10%] right-[-10%] w-96 h-96 rounded-full blur-3xl opacity-20 animate-pulse pointer-events-none transition-colors duration-700 ${isDarkMode ? 'bg-blue-600' : 'bg-cyan-400'}`}></div>
            <div className={`absolute bottom-[10%] left-[-10%] w-80 h-80 rounded-full blur-3xl opacity-20 animate-pulse pointer-events-none transition-colors duration-700 delay-1000 ${isDarkMode ? 'bg-purple-600' : 'bg-fuchsia-400'}`}></div>
            <div className={`absolute top-[40%] left-[50%] transform -translate-x-1/2 w-[500px] h-[500px] rounded-full blur-3xl opacity-10 animate-pulse pointer-events-none transition-colors duration-700 delay-500 ${isDarkMode ? 'bg-indigo-600' : 'bg-blue-300'}`}></div>

            {/* Header */}
            <header className="px-6 pt-12 pb-6 flex justify-between items-center text-white">
                <div className="flex items-center gap-4">
                    <div>
                        <p className="text-hospital-blue-200 text-xs font-bold uppercase tracking-wider">Selamat Datang Kembali</p>

                        <h1 className="text-2xl font-bold">{user.name}</h1>
                        <div className="flex items-center gap-1 text-hospital-blue-200 text-xs mt-1 font-medium">
                            <MapPin size={12} />
                            <span>{user.location || airQuality.city}, {airQuality.country}</span>
                        </div>
                        {user.age && (
                            <p className="text-hospital-blue-300 text-sm font-medium mt-0.5 h-0 overflow-hidden">{user.age} Tahun</p>
                        )}
                    </div>
                </div>
                <div className="relative">
                    <button
                        onClick={() => setShowNotifications(!showNotifications)}
                        className="p-3 rounded-full bg-hospital-blue-900 text-white backdrop-blur-md border border-white/20 hover:bg-white/20 transition-colors"
                    >
                        <Bell size={20} />
                    </button>
                    {!runGoalMet && (
                        <span className="absolute top-0 right-0 w-3 h-3 bg-rose-500 rounded-full border-2 border-hospital-blue-900"></span>
                    )}

                    {/* Notification Popup */}
                    {showNotifications && (
                        <div className={`absolute right-0 top-14 w-64 rounded-xl shadow-xl p-4 z-50 border backdrop-blur-md ${isDarkMode ? 'bg-slate-800/90 border-slate-700 text-white' : 'bg-white/90 border-slate-200 text-slate-800'}`}>
                            <h3 className="font-bold mb-2 text-sm">Notifikasi</h3>
                            <div className={`p-3 rounded-lg text-sm ${runGoalMet ? 'bg-green-100 text-green-800' : 'bg-orange-100 text-orange-800'}`}>
                                <p className="font-semibold">{runGoalMet ? 'Target Tercapai!' : 'Latihan Belum Selesai'}</p>
                                <p className="text-xs mt-1">
                                    {runGoalMet
                                        ? "Kerja bagus! Anda telah mencapai target lari harian Anda."
                                        : `Anda memiliki ${Math.max(0, (runTargetKm - runCurrentKm).toFixed(1))} km lagi untuk lari hari ini.`}
                                </p>
                            </div>
                        </div>
                    )}
                </div>
            </header>

            <motion.div
                className="px-6 space-y-6"
                variants={containerVariants}
                initial="hidden"
                animate="visible"
            >
                {/* Hero: Health Score */}
                <motion.div variants={itemVariants} className={`backdrop-blur-xl border rounded-[32px] p-6 shadow-glass relative overflow-hidden transition-colors duration-300 ${isDarkMode ? 'bg-slate-800/50 border-slate-700 text-white' : 'bg-hospital-blue-100 border-hospital-blue-200 text-hospital-blue-900'}`}>
                    <div className="flex flex-col items-center">
                        <HealthGauge score={user.healthScore} darkMode={isDarkMode} />

                        <motion.button
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            onClick={() => navigate('/record')}
                            className={`mt-6 w-full py-4 rounded-2xl font-bold text-lg shadow-lg flex items-center justify-center gap-2 group animate-pulse-slow transition-colors ${isDarkMode ? 'bg-blue-600 text-white hover:bg-blue-500' : 'bg-white text-hospital-blue-900'}`}
                        >
                            <div className={`w-8 h-8 rounded-full flex items-center justify-center transition-colors ${isDarkMode ? 'bg-white/20' : 'bg-hospital-blue-100 group-hover:bg-hospital-blue-200'}`}>
                                <Play size={16} className={`ml-0.5 ${isDarkMode ? 'fill-white text-white' : 'fill-current text-hospital-blue-600'}`} />
                            </div>
                            Mulai Pemindaian Baru
                        </motion.button>
                    </div>
                </motion.div>

                {/* Vitals Grid */}
                <div className="grid grid-cols-2 gap-4">
                    <motion.div
                        variants={itemVariants}
                        whileHover={{ scale: 1.02 }}
                        className={`rounded-3xl p-5 shadow-sm border relative overflow-hidden transition-colors duration-300 ${isDarkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-100'}`}
                    >
                        <div className="flex justify-between items-start mb-2">
                            <div className={`w-10 h-10 rounded-full flex items-center justify-center ${isDarkMode ? 'bg-blue-900/50 text-blue-400' : 'bg-blue-50 text-blue-600'}`}>
                                <Wind size={20} />
                            </div>
                            <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full font-bold">Normal</span>
                        </div>
                        <h3 className={`text-xs uppercase font-bold tracking-wider mb-1 transition-colors ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>Laju Pernapasan</h3>
                        <div className="flex justify-between items-end mb-2">
                            <p className={`text-2xl font-bold transition-colors ${isDarkMode ? 'text-white' : 'text-slate-800'}`}>
                                {user.lastRespRate || 18} <span className={`text-sm font-normal ${isDarkMode ? 'text-slate-500' : 'text-slate-400'}`}>bpm</span>
                            </p>
                            <button
                                onClick={(e) => { e.stopPropagation(); navigate('/measure-breath'); }}
                                className="text-xs bg-blue-500 text-white px-3 py-1.5 rounded-lg font-bold hover:bg-blue-600 transition-colors z-10"
                            >
                                Measure
                            </button>
                        </div>
                        <SparkLine data={respRateData} color="#3b82f6" />
                    </motion.div>

                    <motion.div
                        variants={itemVariants}
                        whileHover={{ scale: 1.02 }}
                        className={`rounded-3xl p-5 shadow-sm border relative overflow-hidden transition-colors duration-300 ${isDarkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-100'}`}
                    >
                        <div className="flex justify-between items-start mb-2">
                            <div className={`w-10 h-10 rounded-full flex items-center justify-center ${isDarkMode ? 'bg-purple-900/50 text-purple-400' : 'bg-purple-50 text-purple-600'}`}>
                                <Activity size={20} />
                            </div>
                            <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold ${airQuality.color}`}>{airQuality.status}</span>
                        </div>
                        <h3 className={`text-xs uppercase font-bold tracking-wider mb-1 transition-colors ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>Kualitas Udara</h3>
                        <p className={`text-2xl font-bold mb-2 transition-colors ${isDarkMode ? 'text-white' : 'text-slate-800'}`}>{airQuality.aqi} <span className={`text-sm font-normal ${isDarkMode ? 'text-slate-500' : 'text-slate-400'}`}>AQI</span></p>
                        <SparkLine data={aqiData} color="#a855f7" />
                    </motion.div>
                </div>


                {/* Lung Mascot with Health Tips */}
                <motion.div variants={itemVariants}>
                    <LungMascot darkMode={isDarkMode} />
                </motion.div>

                {/* Trends Chart */}
                <motion.div
                    variants={itemVariants}
                    className={`rounded-3xl p-6 shadow-sm border transition-colors duration-300 ${isDarkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-100'}`}
                >
                    <div className="flex justify-between items-center mb-6">
                        <h3 className={`font-bold transition-colors ${isDarkMode ? 'text-white' : 'text-hospital-blue-900'}`}>Riwayat Resiko TBC</h3>
                        <button className="text-xs text-hospital-blue-500 font-bold flex items-center">
                            7 Hari Terakhir <ChevronRight size={14} />
                        </button>
                    </div>

                    <div className="h-48 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={trendsData}>
                                <defs>
                                    <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor={chartColor} stopOpacity={0.3} />
                                        <stop offset="95%" stopColor={chartColor} stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: isDarkMode ? '#64748b' : '#94a3b8' }} />
                                <YAxis hide domain={[0, 100]} />
                                <Tooltip
                                    contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)', backgroundColor: isDarkMode ? '#1e293b' : '#fff', color: isDarkMode ? '#fff' : '#000' }}
                                    cursor={{ stroke: isDarkMode ? '#334155' : '#cbd5e1', strokeWidth: 1, strokeDasharray: '4 4' }}
                                />
                                <Area
                                    type="monotone"
                                    dataKey="score"
                                    stroke={chartColor}
                                    strokeWidth={3}
                                    fillOpacity={1}
                                    fill="url(#colorScore)"
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </motion.div>

            </motion.div >

            <BottomNav />
        </div >
    );
};

export default Dashboard;
