import * as tf from '@tensorflow/tfjs';

const MODEL_URL = '/AI/model.json';
const IMG_SIZE = 224;

let model = null;

/**
 * Load the model from public folder.
 */
const timeoutMs = 120000; // 120 second timeout

export const loadModel = async () => {
    if (model) return model;

    console.log('Using backend:', tf.getBackend());

    const loadPromise = new Promise(async (resolve, reject) => {
        try {
            console.log('Loading model from:', MODEL_URL);
            const modelUrlWithCacheBuster = `${MODEL_URL}?v=${new Date().getTime()}`;

            let loadedModel;
            try {
                // Try loading as Layers Model (Keras) first
                console.log('Attempting to load as LayersModel...');
                loadedModel = await tf.loadLayersModel(modelUrlWithCacheBuster, {
                    onProgress: (fraction) => {
                        console.log(`Model loading progress: ${(fraction * 100).toFixed(1)}%`);
                    }
                });
                console.log('Successfully loaded as LayersModel');
            } catch (layerError) {
                console.warn('Failed to load as LayersModel:', layerError.message);
                console.log('Attempting to load as GraphModel...');

                // Fallback: Try loading as Graph Model (SavedModel/TFHub)
                try {
                    loadedModel = await tf.loadGraphModel(modelUrlWithCacheBuster);
                    console.log('Successfully loaded as GraphModel');
                } catch (graphError) {
                    console.error('Failed to load as GraphModel:', graphError.message);
                    throw new Error(`Model load failed. Layers: ${layerError.message} | Graph: ${graphError.message}`);
                }
            }

            // Warmup
            tf.tidy(() => {
                try {
                    const zeroTensor = tf.zeros([1, IMG_SIZE, IMG_SIZE, 3]);
                    // Check if predict method exists (Layers) or execute (Graph)
                    if (loadedModel.predict) {
                        loadedModel.predict(zeroTensor);
                    } else if (loadedModel.execute) {
                        loadedModel.execute(zeroTensor);
                    }
                    console.log('Warmup successful');
                } catch (e) {
                    console.warn('Warmup prediction failed (non-fatal):', e);
                }
            });
            resolve(loadedModel);
        } catch (error) {
            console.error('Core load error:', error);
            reject(error);
        }
    });

    try {
        model = await Promise.race([
            loadPromise,
            new Promise((_, reject) => setTimeout(() => reject(new Error('Model load timed out check network/files')), timeoutMs))
        ]);
        return model;
    } catch (error) {
        console.error('Failed to load model:', error);
        throw error;
    }
};

// Simple FFT implementation (Radix-2 Cooley-Tukey)
const fft = (inputReal, inputImag) => {
    const n = inputReal.length;
    if (n <= 1) return;
    const half = n / 2;
    const evenReal = new Float32Array(half);
    const evenImag = new Float32Array(half);
    const oddReal = new Float32Array(half);
    const oddImag = new Float32Array(half);
    for (let i = 0; i < half; i++) {
        evenReal[i] = inputReal[2 * i];
        evenImag[i] = inputImag[2 * i];
        oddReal[i] = inputReal[2 * i + 1];
        oddImag[i] = inputImag[2 * i + 1];
    }
    fft(evenReal, evenImag);
    fft(oddReal, oddImag);
    for (let k = 0; k < half; k++) {
        const theta = -2 * Math.PI * k / n;
        const wr = Math.cos(theta);
        const wi = Math.sin(theta);
        const tReal = wr * oddReal[k] - wi * oddImag[k];
        const tImag = wi * oddReal[k] + wr * oddImag[k];
        inputReal[k] = evenReal[k] + tReal;
        inputImag[k] = evenImag[k] + tImag;
        inputReal[k + half] = evenReal[k] - tReal;
        inputImag[k + half] = evenImag[k] - tImag;
    }
};

export const preprocessAudio = async (audioBlob) => {
    return new Promise(async (resolve, reject) => {
        try {
            // Need meyda imported at the top of the file
            const Meyda = (await import('meyda')).default;

            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const arrayBuffer = await audioBlob.arrayBuffer();

            // Wrapper for promise/callback compatibility
            const audioBuffer = await new Promise((res, rej) => {
                audioContext.decodeAudioData(arrayBuffer, res, rej);
            });

            // Librosa defaults: sr=22050, n_mels=128, n_fft=2048, hop_length=512
            // Meyda requires bufferSize to be a power of 2. We'll use 1024 or 2048.
            const bufferSize = 2048;
            const hopSize = 512;
            const channelData = audioBuffer.getChannelData(0); // Mono channel

            // Re-sample if needed? Browser AudioContext might resample to 44100 or 48000 natively.
            // Let's rely on the ratio and extract frames.

            // Initialize Meyda analyzer
            // Meyda operates on fixed buffer sizes. We must slice our float32 array into frames.
            Meyda.bufferSize = bufferSize;
            Meyda.sampleRate = audioBuffer.sampleRate;
            Meyda.windowingFunction = 'hanning';
            // We want 128 mels. Meyda's default melBands is 26, we can override if possible, 
            // but Meyda defaults to 26 mel bands in `melBands` and cannot easily be configured to 128 through the high-level API.
            // Let's use `mfcc` or build a custom mel filterbank if Meyda strictly restricts to 26.
            Meyda.melBands = 128; // Attempt to override, might not work depending on Meyda version internals, 
            // Meyda handles this in `src/utilities.js` melToFreq/freqToMel.

            const numFrames = Math.floor((channelData.length - bufferSize) / hopSize) + 1;
            const melSpectrogram = []; // Will be an array of arrays (frames x mels)
            let maxMelValue = -Infinity;

            for (let i = 0; i < numFrames; i++) {
                const start = i * hopSize;
                const frame = channelData.slice(start, start + bufferSize);

                // Pad if it's the very last frame and shorter than bufferSize
                if (frame.length < bufferSize) {
                    const padded = new Float32Array(bufferSize);
                    padded.set(frame);
                    Meyda.extract('melBands', padded); // Meyda modifies internal state? No, extract returns.
                    // Actually Meyda.extract is static if we pass signal.
                }

                const features = Meyda.extract(['melBands'], frame);
                let bands = features?.melBands || new Array(128).fill(0); // Meyda gives 26 by default. If it can't do 128, this will be 26.

                // Librosa computes Power (magnitude squared) before Mel, Meyda computes it in its melBands logic.
                // Find Max for power_to_db reference
                for (let b = 0; b < bands.length; b++) {
                    if (bands[b] > maxMelValue) maxMelValue = bands[b];
                }

                melSpectrogram.push(bands);
            }

            // Fallback for Meyda fixed 26 melbands: if bands.length is 26, we must interpolate it to 128,
            // or we manually execute FFT and Mel filterbanks. 
            // Given the complexity of writing a complete Mel filterbank in JS matching librosa identically,
            // the closest approximation for the Model (which expects [1, 224, 224, 3]) is to resize whatever we get to 224x224.

            // 2. Power to DB (Log scale)
            // librosa.power_to_db(S, ref=np.max)
            // S_dB = 10 * log10(S / ref)
            // Meyda melBands returns energy (closer to power). 
            const min_db = -80.0;
            const dbSpectrogram = [];

            for (let i = 0; i < melSpectrogram.length; i++) {
                const frameDb = [];
                for (let j = 0; j < melSpectrogram[i].length; j++) {
                    const val = melSpectrogram[i][j];
                    // safe log10
                    let db = 10 * Math.log10(Math.max(1e-10, val) / Math.max(1e-10, maxMelValue));
                    if (db < min_db) db = min_db;
                    frameDb.push(db);
                }
                dbSpectrogram.push(frameDb);
            }

            // 3. Map to 224x224 RGB image (Viridis color map approx or just grayscale stretched to 3 channels)
            // Python used MobileNetV2 which expects 3 channels. Often time-series images are saved with viridis.
            const canvas = document.createElement('canvas');
            canvas.width = IMG_SIZE;
            canvas.height = IMG_SIZE;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, IMG_SIZE, IMG_SIZE);

            const numTimeSteps = dbSpectrogram.length;
            const numMels = dbSpectrogram[0].length; // 26 or 128

            // Draw to Canvas and stretch to 224x224
            for (let x = 0; x < numTimeSteps; x++) {
                // Map x to canvas width (0 to 223)
                const targetX = Math.floor((x / numTimeSteps) * IMG_SIZE);

                for (let y = 0; y < numMels; y++) {
                    // Map y to canvas height. Librosa plots low freq at bottom, high at top.
                    const targetY = IMG_SIZE - 1 - Math.floor((y / numMels) * IMG_SIZE);

                    const db = dbSpectrogram[x][y];
                    // Normalize db (-80 to 0) -> (0 to 255)
                    let val = ((db - min_db) / (0 - min_db)) * 255;
                    val = Math.max(0, Math.min(255, val));

                    // Simple Viridis Approximation
                    const r = val;
                    const g = val > 128 ? (val - 128) * 2 : 0;
                    const b = val > 200 ? (val - 200) * 5 : Math.floor(val * 0.5);

                    ctx.fillStyle = `rgb(${Math.floor(r)},${Math.floor(g)},${Math.floor(b)})`;
                    // We draw rectangles that scale up to the mapped sizes to avoid empty pixels
                    const w = Math.ceil(IMG_SIZE / numTimeSteps);
                    const h = Math.ceil(IMG_SIZE / numMels);
                    ctx.fillRect(targetX, targetY - h + 1, w, h);
                }
            }

            // 4. Create Tensor & Apply MobileNetV2 Preprocessing
            // Python: tf.keras.applications.mobilenet_v2.preprocess_input
            // This expects values in range [0, 255], and normalizes to [-1, 1] internally: (x / 127.5) - 1.0
            const tensor = tf.browser.fromPixels(canvas)
                .resizeBilinear([IMG_SIZE, IMG_SIZE])
                .toFloat()
                .expandDims(); // Shape: [1, 224, 224, 3]

            const normalized = tensor.div(127.5).sub(1);

            // Cleanup memory
            audioContext.close();
            resolve(normalized);

        } catch (e) {
            console.error("Meyda Spectrogram generation failed:", e);
            reject(e);
        }
    });
};

/**
 * Run inference
 */
export const predictCough = async (audioBlob) => {
    try {
        const model = await loadModel();
        const tensor = await preprocessAudio(audioBlob);

        console.log('Running inference...');

        // Handle both GraphModel (execute) and LayersModel (predict)
        let prediction;
        if (model.predict) {
            prediction = model.predict(tensor);
        } else if (model.execute) {
            prediction = model.execute(tensor);
            // GraphModel might return array or map, handle it
            if (Array.isArray(prediction)) prediction = prediction[0];
        } else {
            throw new Error("Unknown model type: no predict or execute method");
        }

        const data = await prediction.data();

        console.log('Raw output:', data);

        // Cleanup tensors
        tensor.dispose();
        prediction.dispose();

        // interpret result
        // The model output is a single sigmoid unit indicating the probability of the positive class (TBC/Cough)
        let probTBC = 0;
        let maxIndex = 0;

        if (data.length === 1) {
            probTBC = data[0];
            maxIndex = probTBC >= 0.5 ? 1 : 0;
        } else {
            // Fallback for categorical outputs
            let maxScore = -1;
            for (let i = 0; i < data.length; i++) {
                if (data[i] > maxScore) {
                    maxScore = data[i];
                    maxIndex = i;
                }
            }
            probTBC = data.length > 1 ? data[1] : maxScore;
        }

        return {
            index: maxIndex,
            score: probTBC,
            raw: Array.from(data)
        };
    } catch (error) {
        console.error('Inference failed:', error);
        throw error;
    }
};
