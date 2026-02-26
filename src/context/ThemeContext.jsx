import React, { createContext, useContext, useState, useEffect } from 'react';

const ThemeContext = createContext();

export const ThemeProvider = ({ children }) => {
    const [isDarkMode, setIsDarkMode] = useState(() => {
        try {
            const saved = localStorage.getItem('respi_user');
            return saved ? JSON.parse(saved).darkMode : false;
        } catch (e) {
            console.error("Failed to parse user settings:", e);
            return false;
        }
    });

    useEffect(() => {
        // Sync with localStorage whenever it changes (though usually driven by Profile)
        // Actually, we should probably start listening to it or just manage it here.
        // For now, let's keep it simple: We update state here and also save to localStorage.
        const saved = localStorage.getItem('respi_user');
        if (saved) {
            const parsed = JSON.parse(saved);
            if (parsed.darkMode !== isDarkMode) {
                // If we want to persist ONLY dark mode updates from here back to the big user object:
                const updated = { ...parsed, darkMode: isDarkMode };
                localStorage.setItem('respi_user', JSON.stringify(updated));
            }
        }
    }, [isDarkMode]);

    // Force body class for global base styles
    useEffect(() => {
        if (isDarkMode) {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
    }, [isDarkMode]);

    const toggleDarkMode = () => setIsDarkMode(prev => !prev);

    // Explicit setter for Profile page compatibility
    const setDarkMode = (value) => setIsDarkMode(value);

    return (
        <ThemeContext.Provider value={{ isDarkMode, toggleDarkMode, setDarkMode }}>
            {children}
        </ThemeContext.Provider>
    );
};

export const useTheme = () => useContext(ThemeContext);
