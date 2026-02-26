import React, { createContext, useContext, useState } from 'react';

const FitnessContext = createContext();

export const FitnessProvider = ({ children }) => {
    // Mock data initialized here
    const [runTargetKm, setRunTargetKm] = useState(3);
    const [runCurrentKm, setRunCurrentKm] = useState(0);

    const runGoalMet = runCurrentKm >= runTargetKm;

    const updateRunProgress = (newDistance) => {
        setRunCurrentKm(newDistance);
    };

    return (
        <FitnessContext.Provider value={{
            runTargetKm,
            runCurrentKm,
            runGoalMet,
            updateRunProgress
        }}>
            {children}
        </FitnessContext.Provider>
    );
};

export const useFitness = () => {
    const context = useContext(FitnessContext);
    if (!context) {
        throw new Error('useFitness must be used within a FitnessProvider');
    }
    return context;
};
