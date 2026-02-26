import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Pause, Timer, RefreshCw } from 'lucide-react';

const poses = [
    {
        id: 1,
        name: "Mountain Pose (Tadasana)",
        image: "/yoga/pose_1.png",
        instruction: "Stand tall with feet together, shoulders relaxed, weight evenly distributed."
    },
    {
        id: 2,
        name: "Downward-Facing Dog",
        image: "/yoga/pose_2.png",
        instruction: "Form an inverted V-shape, hands shoulder-width apart, feet hip-width apart."
    },
    {
        id: 3,
        name: "Warrior II",
        image: "/yoga/pose_3.png",
        instruction: "Legs wide, right knee bent at 90°, arms parallel to floor, gaze over right hand."
    },
    {
        id: 4,
        name: "Tree Pose",
        image: "/yoga/pose_4.png",
        instruction: "Stand on one leg, place other foot on inner thigh/calf, hands in prayer position."
    },
    {
        id: 5,
        name: "Child's Pose",
        image: "/yoga/pose_5.png",
        instruction: "Kneel, sit back on heels, fold forward resting forehead on mat, arms extended."
    }
];

const YogaExercise = ({ darkMode }) => {
    const [currentIndex, setCurrentIndex] = useState(0);
    const [timeLeft, setTimeLeft] = useState(10);
    const [isActive, setIsActive] = useState(false);

    useEffect(() => {
        let interval = null;
        if (isActive) {
            interval = setInterval(() => {
                setTimeLeft((prevTime) => {
                    if (prevTime <= 1) {
                        // Switch to next pose
                        setCurrentIndex((prevIndex) => (prevIndex + 1) % poses.length);
                        return 10; // Reset timer
                    }
                    return prevTime - 1;
                });
            }, 1000);
        } else if (!isActive && timeLeft !== 0) {
            clearInterval(interval);
        }
        return () => clearInterval(interval);
    }, [isActive, timeLeft]);

    const toggleTimer = () => {
        setIsActive(!isActive);
    };

    const resetTimer = () => {
        setIsActive(false);
        setTimeLeft(10);
        setCurrentIndex(0);
    };

    const currentPose = poses[currentIndex];

    // Progress ring calculation (simple width for now)
    const progressPercentage = ((10 - timeLeft) / 10) * 100;

    return (
        <div className={`rounded-3xl p-6 border shadow-sm relative overflow-hidden transition-colors duration-300 ${darkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-100'}`}>
            <div className="flex justify-between items-center mb-4">
                <div className="flex items-center gap-2">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center ${darkMode ? 'bg-teal-900/50 text-teal-400' : 'bg-teal-50 text-teal-600'}`}>
                        <RefreshCw size={16} />
                    </div>
                    <span className={`font-bold ${darkMode ? 'text-white' : 'text-slate-800'}`}>Yoga Flow</span>
                </div>
                <div className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium ${darkMode ? 'bg-slate-700 text-slate-300' : 'bg-slate-100 text-slate-600'}`}>
                    <Timer size={14} />
                    <span>{timeLeft}s</span>
                </div>
            </div>

            <div className="relative aspect-video rounded-2xl overflow-hidden bg-gradient-to-br from-teal-50 to-blue-50 mb-4 flex items-center justify-center p-4">
                <AnimatePresence mode='wait'>
                    <motion.div
                        key={currentPose.id}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                        transition={{ duration: 0.5 }}
                        className="flex flex-col items-center text-center w-full h-full"
                    >
                        <img
                            src={currentPose.image}
                            alt={currentPose.name}
                            className="h-32 object-contain mb-2 drop-shadow-md"
                            onError={(e) => {
                                e.target.onerror = null;
                                e.target.src = 'https://placehold.co/200x200?text=Pose+Image';
                            }} // Fallback
                        />
                        <h3 className={`font-bold text-lg ${darkMode ? 'text-slate-800' : 'text-slate-800'}`}>{currentPose.name}</h3>
                    </motion.div>
                </AnimatePresence>
            </div>

            <div className="space-y-3">
                <p className={`text-xs text-center min-h-[40px] ${darkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                    {currentPose.instruction}
                </p>

                {/* Progress Bar */}
                <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                    <motion.div
                        className="h-full bg-teal-500 rounded-full"
                        animate={{ width: `${progressPercentage}%` }}
                        transition={{ duration: 1, ease: "linear" }}
                    />
                </div>

                <div className="flex justify-center gap-3 pt-2">
                    <button
                        onClick={toggleTimer}
                        className={`p-3 rounded-full transition-colors ${isActive
                            ? (darkMode ? 'bg-amber-500/10 text-amber-500 hover:bg-amber-500/20' : 'bg-amber-50 text-amber-600 hover:bg-amber-100')
                            : (darkMode ? 'bg-teal-500 text-white hover:bg-teal-400' : 'bg-teal-600 text-white hover:bg-teal-700')
                            }`}
                    >
                        {isActive ? <Pause size={20} fill="currentColor" /> : <Play size={20} fill="currentColor" className="ml-0.5" />}
                    </button>
                    <button
                        onClick={resetTimer}
                        className={`p-3 rounded-full transition-colors ${darkMode
                            ? 'bg-slate-700 text-slate-400 hover:bg-slate-600 hover:text-white'
                            : 'bg-slate-100 text-slate-500 hover:bg-slate-200 hover:text-slate-700'
                            }`}
                    >
                        <RefreshCw size={20} />
                    </button>
                </div>
            </div>
        </div>
    );
};

export default YogaExercise;
