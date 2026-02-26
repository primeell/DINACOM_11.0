import React from 'react';
import { Home, Activity, Dumbbell, User, Users } from 'lucide-react';
import { useNavigate, useLocation } from 'react-router-dom';

import { useTheme } from '../context/ThemeContext';

const BottomNav = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const { isDarkMode } = useTheme();

    const navItems = [
        { icon: Home, label: 'Beranda', path: '/dashboard' },
        { icon: Users, label: 'Komunitas', path: '/community' }, // Replaced Activity/Pindai
        { icon: Dumbbell, label: 'Latihan', path: '/exercise' },
        { icon: User, label: 'Profil', path: '/profile' },
    ];

    return (
        <div className={`fixed bottom-0 left-0 right-0 backdrop-blur-lg border-t px-6 py-4 pb-6 z-50 rounded-t-3xl shadow-[0_-5px_20px_rgba(0,0,0,0.05)] transition-colors duration-300 ${isDarkMode ? 'bg-slate-900/90 border-slate-700' : 'bg-white/80 border-slate-200'
            }`}>
            <div className="flex justify-between items-center max-w-md mx-auto">
                {navItems.map((item) => {
                    const isActive = location.pathname === item.path;
                    return (
                        <button
                            key={item.label}
                            onClick={() => navigate(item.path)}
                            className={`flex flex-col items-center gap-1 transition-all duration-300 ${isActive
                                ? 'text-hospital-blue-600 -translate-y-2'
                                : isDarkMode ? 'text-slate-500 hover:text-slate-300' : 'text-slate-400 hover:text-hospital-blue-400'
                                }`}
                        >
                            <div className={`p-2 rounded-xl transition-all ${isActive ? 'bg-hospital-blue-100 shadow-lg shadow-hospital-blue-100' : ''}`}>
                                <item.icon size={isActive ? 24 : 22} strokeWidth={isActive ? 2.5 : 2} />
                            </div>
                            <span className={`text-[10px] font-medium ${isActive ? 'opacity-100' : 'opacity-0 h-0 overflow-hidden'}`}>
                                {item.label}
                            </span>
                        </button>
                    );
                })}
            </div>
        </div>
    );
};

export default BottomNav;
