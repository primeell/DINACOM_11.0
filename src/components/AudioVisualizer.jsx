import React, { useEffect, useRef } from 'react';

const AudioVisualizer = ({ isRecording, stream }) => {
    const canvasRef = useRef(null);
    const animationRef = useRef(null);
    const audioContextRef = useRef(null);
    const analyserRef = useRef(null);
    const sourceRef = useRef(null);

    useEffect(() => {
        if (!canvasRef.current) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // Initialize Audio Context if stream is available
        if (isRecording && stream) {
            if (!audioContextRef.current) {
                audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
                analyserRef.current = audioContextRef.current.createAnalyser();
                analyserRef.current.fftSize = 256;
                sourceRef.current = audioContextRef.current.createMediaStreamSource(stream);
                sourceRef.current.connect(analyserRef.current);
            }
        }

        const draw = () => {
            if (!isRecording) {
                // Idle animation (flat line or gentle wave)
                ctx.clearRect(0, 0, width, height);
                ctx.beginPath();
                ctx.moveTo(0, height / 2);
                for (let i = 0; i < width; i++) {
                    ctx.lineTo(i, height / 2 + Math.sin(i * 0.05 + Date.now() * 0.005) * 5);
                }
                ctx.strokeStyle = '#38bdf8'; // hospital-blue-400
                ctx.lineWidth = 2;
                ctx.stroke();
            } else if (analyserRef.current) {
                // Real visualizations
                const bufferLength = analyserRef.current.frequencyBinCount;
                const dataArray = new Uint8Array(bufferLength);
                analyserRef.current.getByteFrequencyData(dataArray);

                ctx.fillStyle = 'rgba(240, 249, 255, 0.2)'; // Clear with fade for trail effect
                ctx.fillRect(0, 0, width, height);

                const barWidth = (width / bufferLength) * 2.5;
                let barHeight;
                let x = 0;

                for (let i = 0; i < bufferLength; i++) {
                    barHeight = dataArray[i] / 2;

                    // Gradient color based on height
                    const r = 50 + (barHeight + 25) * 2;
                    const g = 200 - barHeight;
                    const b = 250;

                    ctx.fillStyle = `rgb(${r},${g},${b})`;

                    // Mirror visualization
                    ctx.fillRect(x, height / 2 - barHeight / 2, barWidth, barHeight);

                    x += barWidth + 1;
                }
            }

            animationRef.current = requestAnimationFrame(draw);
        };

        draw();

        return () => {
            cancelAnimationFrame(animationRef.current);
            if (audioContextRef.current && !isRecording) {
                // Optimization: Suspend or close if not needed, but for now we keep it simple
            }
        };
    }, [isRecording, stream]);

    return (
        <canvas
            ref={canvasRef}
            width={300}
            height={150}
            className="w-full h-40 rounded-xl bg-hospital-blue-950/5 border border-hospital-blue-100"
        />
    );
};

export default AudioVisualizer;
