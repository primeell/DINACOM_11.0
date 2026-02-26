import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ChevronLeft, ChevronRight, User, Bell, Moon, Save, Camera, Clock, Music, Image as ImageIcon } from 'lucide-react';
import BottomNav from '../components/BottomNav';
import { useTheme } from '../context/ThemeContext';

const Profile = () => {
    const navigate = useNavigate();
    const { isDarkMode, setDarkMode } = useTheme();
    const [loading, setLoading] = useState(false);
    const [isReminderOpen, setIsReminderOpen] = useState(false);

    const fileInputRef = React.useRef(null);

    // Initial State from LocalStorage
    const [formData, setFormData] = useState(() => {
        const saved = localStorage.getItem('respi_user');
        return saved ? JSON.parse(saved) : {
            name: '',
            age: '',
            notifications: true,
            // darkMode: isDarkMode, // Don't rely on local storage here for initial render if context handles it, but context also reads LS.
            // Let's rely on context for the source of truth for 'darkMode' value.
            reminderEnabled: false,
            reminderTime: '08:00',
            reminderSound: 'Default',
            avatar: null
        };
    });

    const [currentTime, setCurrentTime] = useState(new Date());

    useEffect(() => {
        const timer = setInterval(() => setCurrentTime(new Date()), 1000);
        return () => clearInterval(timer);
    }, []);

    const handleChange = (e) => {
        const { name, value, type, checked } = e.target;
        if (name === 'darkMode') {
            setDarkMode(checked);
        } else {
            setFormData(prev => ({
                ...prev,
                [name]: type === 'checkbox' ? checked : value
            }));
        }
    };

    const handleImageUpload = (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onloadend = () => {
                setFormData(prev => ({
                    ...prev,
                    avatar: reader.result
                }));
            };
            reader.readAsDataURL(file);
        }
    };

    const [isCameraOpen, setIsCameraOpen] = useState(false);
    const videoRef = React.useRef(null);
    const streamRef = React.useRef(null);

    const startCamera = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            setIsCameraOpen(true);
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
            }
            streamRef.current = stream;
        } catch (err) {
            console.error("Error accessing camera:", err);
            // Fallback to file picker if camera access fails (e.g., blocked or not available)
            alert("Unable to access camera. Opening file selection instead.");
            fileInputRef.current.click();
            setIsCameraOpen(false);
        }
    };

    const stopCamera = () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
        setIsCameraOpen(false);
    };

    const capturePhoto = () => {
        if (videoRef.current) {
            const canvas = document.createElement('canvas');
            canvas.width = videoRef.current.videoWidth;
            canvas.height = videoRef.current.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(videoRef.current, 0, 0);
            const dataUrl = canvas.toDataURL('image/jpeg');
            setFormData(prev => ({
                ...prev,
                avatar: dataUrl
            }));
            stopCamera();
        }
    };

    const [showPhotoMenu, setShowPhotoMenu] = useState(false);

    const triggerFileInput = () => {
        setShowPhotoMenu(true);
    };

    const handleMenuSelection = (option) => {
        setShowPhotoMenu(false);
        if (option === 'camera') {
            startCamera();
        } else {
            fileInputRef.current.click();
        }
    };

    const handleSave = () => {
        setLoading(true);
        // Simulate API call/processing
        setTimeout(() => {
            localStorage.setItem('respi_user', JSON.stringify(formData));
            setLoading(false);
            navigate('/dashboard');
        }, 800);
    };

    const playNotificationSound = (type) => {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);

        const now = audioContext.currentTime;

        if (type === 'Gentle') {
            oscillator.type = 'sine';
            oscillator.frequency.setValueAtTime(440, now);
            oscillator.frequency.exponentialRampToValueAtTime(880, now + 1);
            gainNode.gain.setValueAtTime(0.1, now);
            gainNode.gain.exponentialRampToValueAtTime(0.01, now + 1);
            oscillator.start(now);
            oscillator.stop(now + 1);
        } else if (type === 'Energetic') {
            oscillator.type = 'square';
            oscillator.frequency.setValueAtTime(523.25, now);
            oscillator.frequency.linearRampToValueAtTime(659.25, now + 0.1);
            oscillator.frequency.linearRampToValueAtTime(783.99, now + 0.2);
            gainNode.gain.setValueAtTime(0.1, now);
            gainNode.gain.exponentialRampToValueAtTime(0.01, now + 0.3);
            oscillator.start(now);
            oscillator.stop(now + 0.3);
        } else if (type === 'Nature') {
            oscillator.type = 'triangle';
            oscillator.frequency.setValueAtTime(329.63, now);
            oscillator.frequency.linearRampToValueAtTime(392.00, now + 0.5);
            gainNode.gain.setValueAtTime(0.1, now);
            gainNode.gain.linearRampToValueAtTime(0, now + 2);
            oscillator.start(now);
            oscillator.stop(now + 2);
        } else {
            // Default Beep
            oscillator.type = 'sine';
            oscillator.frequency.setValueAtTime(880, now);
            gainNode.gain.setValueAtTime(0.1, now);
            gainNode.gain.exponentialRampToValueAtTime(0.01, now + 0.5);
            oscillator.start(now);
            oscillator.stop(now + 0.5);
        }
    };

    const containerVariants = {
        hidden: { opacity: 0 },
        visible: { opacity: 1, transition: { duration: 0.5 } }
    };

    return (
        <div className="min-h-screen bg-hospital-blue-900 relative pb-32 overflow-hidden font-sans">
            {/* Background */}
            <div
                className="fixed inset-0 w-full h-full -z-20 bg-cover bg-center bg-no-repeat"
                style={{ backgroundImage: "url('/hospital-bg.png')" }}
            />
            <div className="fixed inset-0 w-full h-full bg-hospital-blue-900/80 -z-10" />

            {/* Ambient Decorations */}
            <div className={`absolute top-[-10%] right-[-10%] w-96 h-96 rounded-full blur-3xl opacity-20 animate-pulse pointer-events-none transition-colors duration-700 ${isDarkMode ? 'bg-blue-600' : 'bg-cyan-400'}`}></div>
            <div className={`absolute bottom-[10%] left-[-10%] w-80 h-80 rounded-full blur-3xl opacity-20 animate-pulse pointer-events-none transition-colors duration-700 delay-1000 ${isDarkMode ? 'bg-purple-600' : 'bg-fuchsia-400'}`}></div>
            <div className={`absolute top-[40%] left-[50%] transform -translate-x-1/2 w-[500px] h-[500px] rounded-full blur-3xl opacity-10 animate-pulse pointer-events-none transition-colors duration-700 delay-500 ${isDarkMode ? 'bg-indigo-600' : 'bg-blue-300'}`}></div>

            {/* Header */}
            <header className="px-6 pt-12 pb-6 flex items-center justify-between text-white relative z-10">
                <button
                    onClick={() => navigate('/dashboard')}
                    className="p-2 rounded-full bg-white/10 backdrop-blur-md hover:bg-white/20 transition-colors"
                >
                    <ChevronLeft size={24} />
                </button>
                <h1 className="text-xl font-bold">My Profile</h1>
                <div className="w-10" /> {/* Spacer for centering */}
            </header>

            <motion.div
                className="px-6 space-y-6"
                variants={containerVariants}
                initial="hidden"
                animate="visible"
            >
                {/* Avatar Section */}
                <div className="flex flex-row items-center gap-6 py-6 pb-8">
                    <div className="relative shrink-0">
                        <div className="w-24 h-24 rounded-full bg-gradient-to-tr from-hospital-blue-400 to-medical-teal-400 p-1">
                            <div className="w-full h-full rounded-full bg-hospital-blue-100 flex items-center justify-center overflow-hidden border-4 border-white/20">
                                {formData.avatar ? (
                                    <img src={formData.avatar} alt="Profile" className="w-full h-full object-cover" />
                                ) : (
                                    <User size={40} className="text-hospital-blue-400" />
                                )}
                            </div>
                        </div>
                        <button
                            onClick={triggerFileInput}
                            className="absolute bottom-0 right-0 p-2 rounded-full bg-medical-teal-500 text-white shadow-lg border-2 border-hospital-blue-900 hover:scale-110 transition-transform"
                        >
                            <Camera size={14} />
                        </button>
                        <input
                            type="file"
                            ref={fileInputRef}
                            onChange={handleImageUpload}
                            accept="image/*"
                            className="hidden"
                        />
                    </div>

                    <div className="flex flex-col items-start min-w-0 flex-1">
                        <h2 className="text-2xl font-bold text-white w-full truncate text-left">
                            {formData.name || 'Guest User'}
                        </h2>


                        {/* Digital Clock */}
                        <div className="flex items-baseline gap-2">
                            <div className="text-2xl font-mono font-bold text-white/90 tracking-widest drop-shadow-sm">
                                {currentTime.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })}
                            </div>
                            <div className="text-hospital-blue-200 text-xs font-medium uppercase tracking-wide">
                                {currentTime.toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' })}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Form Section */}
                <div className={`backdrop-blur-xl rounded-[32px] p-6 shadow-xl space-y-6 transition-colors duration-300 ${isDarkMode ? 'bg-slate-900/95 text-white border border-slate-700' : 'bg-white/95 text-slate-800'}`}>


                    <div>
                        <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">App Settings</label>
                        <div className="space-y-3">
                            <div className={`flex items-center justify-between p-3 rounded-xl transition-colors ${isDarkMode ? 'hover:bg-slate-800' : 'hover:bg-slate-50'}`}>
                                <div className="flex items-center gap-3">
                                    <div className="p-2 rounded-lg bg-orange-100 text-orange-600">
                                        <Bell size={20} />
                                    </div>
                                    <span className={`font-medium ${isDarkMode ? 'text-slate-200' : 'text-slate-700'}`}>Notifications</span>
                                </div>
                                <label className="relative inline-flex items-center cursor-pointer">
                                    <input
                                        type="checkbox"
                                        name="notifications"
                                        checked={formData.notifications}
                                        onChange={handleChange}
                                        className="sr-only peer"
                                    />
                                    <div className={`w-11 h-6 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-green-500 ${isDarkMode ? 'bg-slate-700' : 'bg-slate-200'}`}></div>
                                </label>
                            </div>

                            <div className={`flex items-center justify-between p-3 rounded-xl transition-colors ${isDarkMode ? 'hover:bg-slate-800' : 'hover:bg-slate-50'}`}>
                                <div className="flex items-center gap-3">
                                    <div className={`p-2 rounded-lg ${isDarkMode ? 'bg-indigo-900/50 text-indigo-400' : 'bg-indigo-100 text-indigo-600'}`}>
                                        <Moon size={20} />
                                    </div>
                                    <span className={`font-medium ${isDarkMode ? 'text-slate-200' : 'text-slate-700'}`}>Dark Mode</span>
                                </div>
                                <label className="relative inline-flex items-center cursor-pointer">
                                    <input
                                        type="checkbox"
                                        name="darkMode"
                                        checked={isDarkMode}
                                        onChange={handleChange}
                                        className="sr-only peer"
                                    />
                                    <div className={`w-11 h-6 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-indigo-500 ${isDarkMode ? 'bg-slate-700' : 'bg-slate-200'}`}></div>
                                </label>
                            </div>

                            <button
                                onClick={() => setIsReminderOpen(true)}
                                className={`w-full flex items-center justify-between p-3 rounded-xl transition-colors ${isDarkMode ? 'hover:bg-slate-800' : 'hover:bg-slate-50'}`}
                            >
                                <div className="flex items-center gap-3">
                                    <div className="p-2 rounded-lg bg-blue-100 text-blue-600">
                                        <Clock size={20} />
                                    </div>
                                    <div className="text-left">
                                        <div className={`font-medium ${isDarkMode ? 'text-slate-200' : 'text-slate-700'}`}>Daily Check-up</div>
                                        <div className={`text-xs ${isDarkMode ? 'text-slate-500' : 'text-slate-400'}`}>
                                            {formData.reminderEnabled ? formData.reminderTime : 'Off'}
                                        </div>
                                    </div>
                                </div>
                                <ChevronRight size={20} className={isDarkMode ? 'text-slate-600' : 'text-slate-300'} />
                            </button>
                        </div>
                    </div>

                    <div className="flex flex-col gap-3">
                        <button
                            onClick={handleSave}
                            disabled={loading}
                            className="w-full py-4 bg-hospital-blue-600 text-white rounded-2xl font-bold text-lg shadow-lg shadow-hospital-blue-600/30 active:scale-[0.98] transition-all flex items-center justify-center gap-2 disabled:opacity-70 disabled:cursor-not-allowed"
                        >
                            {loading ? (
                                <span className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                            ) : (
                                <>
                                    <Save size={20} />
                                    Save Changes
                                </>
                            )}
                        </button>

                        <button
                            onClick={() => {
                                if (window.confirm('Are you sure you want to log out?')) {
                                    localStorage.removeItem('respi_user');
                                    navigate('/');
                                }
                            }}
                            className={`w-full py-4 rounded-2xl font-bold text-lg active:scale-[0.98] transition-all border-2 ${isDarkMode
                                ? 'border-red-500/50 text-red-500 hover:bg-red-500/10'
                                : 'border-red-100 text-red-500 hover:bg-red-50'
                                }`}
                        >
                            Log Out
                        </button>
                    </div>
                </div>
            </motion.div>

            {/* Reminder Settings Sub-view */}
            <motion.div
                initial={{ x: '100%' }}
                animate={{ x: isReminderOpen ? 0 : '100%' }}
                transition={{ type: 'spring', damping: 25, stiffness: 200 }}
                className={`fixed inset-0 z-50 overflow-hidden ${isDarkMode ? 'bg-slate-900' : 'bg-hospital-blue-50'}`}
            >
                <div className="flex flex-col h-full">
                    {/* Header */}
                    <div className={`px-6 pt-12 pb-6 flex items-center gap-4 ${isDarkMode ? 'bg-slate-900 text-white' : 'bg-white text-hospital-blue-900 shadow-sm'}`}>
                        <button
                            onClick={() => setIsReminderOpen(false)}
                            className={`p-2 rounded-full ${isDarkMode ? 'hover:bg-slate-800' : 'hover:bg-slate-100'} transition-colors`}
                        >
                            <ChevronLeft size={24} />
                        </button>
                        <h2 className="text-xl font-bold">Daily Check-up</h2>
                    </div>

                    <div className="flex-1 p-6">
                        <div className={`rounded-3xl p-6 shadow-sm space-y-6 ${isDarkMode ? 'bg-slate-800 text-white' : 'bg-white text-slate-800'}`}>
                            <div className="flex items-center justify-between">
                                <span className="font-medium text-lg">Enable Reminder</span>
                                <label className="relative inline-flex items-center cursor-pointer">
                                    <input
                                        type="checkbox"
                                        name="reminderEnabled"
                                        checked={formData.reminderEnabled}
                                        onChange={handleChange}
                                        className="sr-only peer"
                                    />
                                    <div className={`w-14 h-8 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[4px] after:left-[4px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-6 after:w-6 after:transition-all peer-checked:bg-blue-500 ${isDarkMode ? 'bg-slate-700' : 'bg-slate-200'}`}></div>
                                </label>
                            </div>

                            {formData.reminderEnabled && (
                                <motion.div
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="space-y-2"
                                >
                                    <label className={`block text-sm font-medium ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                                        Reminder Time
                                    </label>
                                    <input
                                        type="time"
                                        name="reminderTime"
                                        value={formData.reminderTime}
                                        onChange={handleChange}
                                        className={`w-full px-4 py-4 rounded-xl text-3xl font-bold text-center focus:outline-none focus:ring-2 focus:ring-blue-500/50 ${isDarkMode
                                            ? 'bg-slate-700 text-white border-transparent'
                                            : 'bg-slate-50 text-hospital-blue-900 border-slate-200'
                                            }`}
                                    />
                                    <p className={`text-center text-sm mt-4 ${isDarkMode ? 'text-slate-500' : 'text-slate-400'}`}>
                                        We'll remind you to check your respiratory health daily at this time.
                                    </p>
                                </motion.div>
                            )}

                            {formData.reminderEnabled && (
                                <motion.div
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.1 }}
                                    className="space-y-3 pt-4 border-t border-slate-200/20"
                                >
                                    <label className={`block text-sm font-medium ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                                        Ringtone
                                    </label>
                                    <div className="space-y-2">
                                        {['Default', 'Gentle', 'Energetic', 'Nature'].map((sound) => (
                                            <button
                                                key={sound}
                                                onClick={() => {
                                                    setFormData(prev => ({ ...prev, reminderSound: sound }));
                                                    playNotificationSound(sound);
                                                }}
                                                className={`w-full flex items-center justify-between p-4 rounded-xl transition-all ${formData.reminderSound === sound
                                                    ? (isDarkMode ? 'bg-blue-600 text-white' : 'bg-hospital-blue-100 text-hospital-blue-700 ring-2 ring-hospital-blue-500')
                                                    : (isDarkMode ? 'bg-slate-700 text-slate-300 hover:bg-slate-600' : 'bg-slate-50 text-slate-700 hover:bg-slate-100')
                                                    }`}
                                            >
                                                <div className="flex items-center gap-3">
                                                    <Music size={18} className={formData.reminderSound === sound ? 'opacity-100' : 'opacity-50'} />
                                                    <span className="font-medium">{sound}</span>
                                                </div>
                                                {formData.reminderSound === sound && (
                                                    <div className="w-2 h-2 rounded-full bg-current"></div>
                                                )}
                                            </button>
                                        ))}
                                    </div>
                                </motion.div>
                            )}
                        </div>
                    </div>

                    <div className="p-6 pb-10">
                        <button
                            onClick={() => setIsReminderOpen(false)}
                            className="w-full py-4 bg-hospital-blue-600 text-white rounded-2xl font-bold text-lg shadow-lg active:scale-[0.98] transition-all"
                        >
                            Done
                        </button>
                    </div>
                </div>
            </motion.div>

            {/* Photo Selection Menu */}
            {showPhotoMenu && (
                <div className="fixed inset-0 z-[60] flex items-end sm:items-center justify-center bg-black/60 backdrop-blur-sm" onClick={() => setShowPhotoMenu(false)}>
                    <motion.div
                        initial={{ y: "100%" }}
                        animate={{ y: 0 }}
                        exit={{ y: "100%" }}
                        transition={{ type: "spring", damping: 25, stiffness: 300 }}
                        className={`w-full max-w-sm mx-4 mb-4 sm:mb-0 rounded-3xl p-6 space-y-6 shadow-2xl ${isDarkMode ? 'bg-slate-900 border border-slate-700' : 'bg-white'}`}
                        onClick={e => e.stopPropagation()}
                    >
                        <div className="flex items-center justify-between">
                            <h3 className={`font-bold text-lg ${isDarkMode ? 'text-white' : 'text-slate-800'}`}>
                                Change Photo
                            </h3>
                            <button
                                onClick={() => setShowPhotoMenu(false)}
                                className={`text-sm font-medium ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}
                            >
                                Cancel
                            </button>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            <button
                                onClick={() => handleMenuSelection('camera')}
                                className={`flex flex-col items-center justify-center gap-3 p-6 rounded-2xl transition-all active:scale-95 ${isDarkMode ? 'bg-slate-800 hover:bg-slate-700 text-blue-400' : 'bg-blue-50 hover:bg-blue-100 text-blue-600'}`}
                            >
                                <div className={`p-4 rounded-full ${isDarkMode ? 'bg-slate-700' : 'bg-white shadow-sm'}`}>
                                    <Camera size={32} />
                                </div>
                                <span className={`font-medium ${isDarkMode ? 'text-slate-200' : 'text-slate-700'}`}>Camera</span>
                            </button>

                            <button
                                onClick={() => handleMenuSelection('upload')}
                                className={`flex flex-col items-center justify-center gap-3 p-6 rounded-2xl transition-all active:scale-95 ${isDarkMode ? 'bg-slate-800 hover:bg-slate-700 text-emerald-400' : 'bg-emerald-50 hover:bg-emerald-100 text-emerald-600'}`}
                            >
                                <div className={`p-4 rounded-full ${isDarkMode ? 'bg-slate-700' : 'bg-white shadow-sm'}`}>
                                    <ImageIcon size={32} />
                                </div>
                                <span className={`font-medium ${isDarkMode ? 'text-slate-200' : 'text-slate-700'}`}>Gallery</span>
                            </button>
                        </div>
                    </motion.div>
                </div >
            )}
            {/* Webcam Modal */}
            {
                isCameraOpen && (
                    <div className="fixed inset-0 z-[100] bg-black flex flex-col">
                        <div className="relative flex-1 bg-black flex items-center justify-center overflow-hidden">
                            <video
                                ref={videoRef}
                                autoPlay
                                playsInline
                                className="w-full h-full object-cover"
                            />
                            <div className="absolute top-0 left-0 right-0 p-6 flex justify-between items-center bg-gradient-to-b from-black/50 to-transparent">
                                <button onClick={stopCamera} className="p-2 rounded-full bg-black/20 text-white backdrop-blur-md">
                                    <ChevronLeft size={24} />
                                </button>
                                <span className="text-white font-medium">Take Photo</span>
                                <div className="w-10"></div>
                            </div>
                        </div>
                        <div className="bg-black p-8 pb-12 flex items-center justify-center gap-8">
                            <button
                                onClick={capturePhoto}
                                className="w-20 h-20 rounded-full border-4 border-white flex items-center justify-center relative active:scale-95 transition-transform"
                            >
                                <div className="w-16 h-16 bg-white rounded-full"></div>
                            </button>
                        </div>
                    </div>
                )
            }

            <BottomNav />
        </div >
    );
};

export default Profile;
