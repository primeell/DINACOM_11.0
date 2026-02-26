import React, { useEffect, useRef } from 'react';

const SpectrogramView = ({ progress }) => {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // Draw background
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, width, height);

        // Draw "spectrogram" data
        // We simulate this by drawing random colored blocks that scroll
        const cols = 50;
        const rows = 30;
        const cellWidth = width / cols;
        const cellHeight = height / rows;

        for (let i = 0; i < cols; i++) {
            // Only draw up to current progress
            if ((i / cols) * 100 > progress) continue;

            for (let j = 0; j < rows; j++) {
                // Perlin-ish noise simulation
                const intensity = Math.random();
                const r = intensity * 255;
                const g = intensity * 50;
                const b = 255 - (intensity * 100);
                const a = 0.5 + Math.random() * 0.5;

                ctx.fillStyle = `rgba(${r},${g},${b},${a})`;
                ctx.fillRect(i * cellWidth, height - (j + 1) * cellHeight, cellWidth, cellHeight);
            }
        }

        // Scanline
        if (progress < 100) {
            const scanX = (progress / 100) * width;
            ctx.strokeStyle = '#14B8A6'; // Medical Teal
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(scanX, 0);
            ctx.lineTo(scanX, height);
            ctx.stroke();

            ctx.fillStyle = 'rgba(20, 184, 166, 0.2)';
            ctx.fillRect(scanX - 10, 0, 10, height);
        }

    }, [progress]);

    return (
        <div className="relative w-full rounded-xl overflow-hidden border border-hospital-blue-800 shadow-inner">
            <canvas ref={canvasRef} width={320} height={180} className="w-full h-auto block" />
            <div className="absolute top-2 left-2 text-[10px] text-medical-teal-500 font-mono bg-black/50 px-1 rounded">
                MEL-SPECTROGRAM GENERATOR
            </div>
        </div>
    );
};

export default SpectrogramView;
