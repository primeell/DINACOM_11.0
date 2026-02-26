import React from 'react';
import { motion } from 'framer-motion';

const HealthGauge = ({ score = 20, darkMode = false }) => {
    // 8a. Calculate circumference for stroke-dasharray (r=45, C = 2*pi*45 ≈ 283)
    const radius = 70;
    const circumference = 2 * Math.PI * radius;
    // For risk: High score = High bar.
    const offset = circumference - (score / 100) * circumference;

    // Color logic (Risk: Low is Good/Green, High is Bad/Red)
    const getColor = (s) => {
        if (s < 40) return "#10B981"; // Green (Low Risk)
        if (s < 70) return "#F59E0B"; // Amber (Med Risk)
        return "#EF4444"; // Red (High Risk)
    };

    const getLabel = (s) => {
        if (s < 40) return "Resiko Rendah";
        if (s < 70) return "Resiko Sedang";
        return "Resiko Tinggi";
    };

    return (
        <div className="relative w-48 h-48 flex items-center justify-center">
            {/* Background Circle */}
            <svg className="w-full h-full transform -rotate-90">
                <circle
                    cx="96"
                    cy="96"
                    r={radius}
                    stroke={darkMode ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}
                    strokeWidth="18"
                    fill="transparent"
                />

                {/* Progress Circle */}
                <motion.circle
                    cx="96"
                    cy="96"
                    r={radius}
                    stroke={getColor(score)}
                    strokeWidth="18"
                    fill="transparent"
                    strokeDasharray={circumference}
                    strokeDashoffset={offset}
                    strokeLinecap="round"
                    initial={{ strokeDashoffset: circumference }}
                    animate={{ strokeDashoffset: offset }}
                    transition={{ duration: 1.5, ease: "easeOut" }}
                />
            </svg>

            {/* Content */}
            <div className={`absolute flex flex-col items-center justify-center ${darkMode ? 'text-white' : 'text-hospital-blue-900'}`}>
                <span className={`text-sm font-medium ${darkMode ? 'text-slate-400' : 'text-slate-500'}`}>Resiko TBC</span>
                <span className="text-5xl font-bold tracking-tighter">{Math.round(score)}%</span>
                <span className={`text-xs font-bold mt-1 ${score < 40 ? 'text-green-500' : score > 70 ? 'text-red-500' : 'text-orange-500'}`}>
                    {getLabel(score)}
                </span>
            </div>

            {/* Glow Effect */}
            <div className="absolute inset-0 rounded-full bg-medical-teal-500 blur-3xl opacity-10 pointer-events-none"></div>
        </div>
    );
};

export default HealthGauge;
