import React from 'react';
import { motion } from 'framer-motion';
import { Dumbbell, Bell, MapPin } from 'lucide-react';
import BottomNav from '../components/BottomNav';
import BreathingExercise from '../components/BreathingExercise';
import DailyRun from '../components/DailyRun';
import YogaExercise from '../components/YogaExercise';
import AlarmManager from '../components/AlarmManager';
import { useTheme } from '../context/ThemeContext';

const Exercise = () => {
    const { isDarkMode } = useTheme();
    // Helper to get user data - duplicating from Dashboard for now as it's local state there
    // In a real app this should be in context
    const user = { name: 'User' };

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
        <div className={`min-h-screen relative pb-32 overflow-hidden font-sans transition-colors duration-300 ${isDarkMode ? 'bg-slate-900' : 'bg-hospital-blue-900'}`}>
            <div
                className="fixed inset-0 w-full h-full -z-20 bg-cover bg-center bg-no-repeat opacity-50"
                style={{ backgroundImage: "url('/hospital-bg.png')" }}
            />

            {/* Dark Overlay */}
            <div className={`fixed inset-0 w-full h-full -z-10 transition-colors duration-300 ${isDarkMode ? 'bg-slate-900/90' : 'bg-hospital-blue-900/70'}`} />

            {/* Ambient Decorations */}
            <div className={`absolute top-[-10%] right-[-10%] w-96 h-96 rounded-full blur-3xl opacity-20 animate-pulse pointer-events-none transition-colors duration-700 ${isDarkMode ? 'bg-blue-600' : 'bg-cyan-400'}`}></div>
            <div className={`absolute bottom-[10%] left-[-10%] w-80 h-80 rounded-full blur-3xl opacity-20 animate-pulse pointer-events-none transition-colors duration-700 delay-1000 ${isDarkMode ? 'bg-purple-600' : 'bg-fuchsia-400'}`}></div>
            <div className={`absolute top-[40%] left-[50%] transform -translate-x-1/2 w-[500px] h-[500px] rounded-full blur-3xl opacity-10 animate-pulse pointer-events-none transition-colors duration-700 delay-500 ${isDarkMode ? 'bg-indigo-600' : 'bg-blue-300'}`}></div>

            {/* Header */}
            <header className="px-6 pt-12 pb-6 flex justify-between items-center text-white">
                <div>
                    <h1 className="text-2xl font-bold">Exercises</h1>
                    <p className="text-hospital-blue-200 text-xs font-medium mt-1">
                        Stay fit and healthy
                    </p>
                </div>
                <div className="relative">
                    <button className="p-3 rounded-full bg-hospital-blue-900 text-white backdrop-blur-md border border-white/20 hover:bg-white/20 transition-colors">
                        <Bell size={20} />
                    </button>
                    <span className="absolute top-0 right-0 w-3 h-3 bg-rose-500 rounded-full border-2 border-hospital-blue-900"></span>
                </div>
            </header>

            <motion.div
                className="px-6 space-y-6"
                variants={containerVariants}
                initial="hidden"
                animate="visible"
            >
                {/* Breathing Exercise */}
                <motion.div variants={itemVariants}>
                    <BreathingExercise darkMode={isDarkMode} />
                </motion.div>

                {/* Daily Run Tracker */}
                <motion.div variants={itemVariants}>
                    <DailyRun darkMode={isDarkMode} />
                </motion.div>

                {/* Yoga Exercise */}
                <motion.div variants={itemVariants}>
                    <YogaExercise darkMode={isDarkMode} />
                </motion.div>
            </motion.div>

            <AlarmManager />
            <BottomNav />
        </div>
    );
};

export default Exercise;
