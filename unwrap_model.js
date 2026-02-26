import fs from 'fs';
const inputPath = 'public/AI/model.json';
const outputPath = 'public/AI/model.json';

try {
    const raw = fs.readFileSync(inputPath, 'utf8');
    let data = JSON.parse(raw);

    // Check if it is wrapped in Sequential
    if (data.modelTopology &&
        data.modelTopology.class_name === 'Sequential' &&
        data.modelTopology.config &&
        data.modelTopology.config.layers &&
        data.modelTopology.config.layers.length > 1 &&
        data.modelTopology.config.layers[1].class_name === 'Model') {

        console.log('Found Sequential wrapper. Unwrapping inner Model...');

        const innerModel = data.modelTopology.config.layers[1];

        // 1. Promote inner model to topology
        data.modelTopology.class_name = 'Model';
        data.modelTopology.config = innerModel.config;

        // 2. Fix weights paths in manifest
        // The prefix is likely "sequential/mobilenetv2_1.00_224/" 
        // We need to match what's actually in the manifest.
        // Let's find the prefix by looking at the first weight.

        if (data.weightsManifest && data.weightsManifest.length > 0) {
            const firstWeight = data.weightsManifest[0].weights[0];
            const name = firstWeight.name;
            console.log('First weight name:', name);

            // Heuristic to find prefix: everything before "Conv1" or "input"?
            // Or just hardcode based on inspection "sequential/mobilenetv2_1.00_224/"

            const prefix = "sequential/mobilenetv2_1.00_224/";
            if (name.startsWith(prefix)) {
                console.log(`Stripping prefix "${prefix}" from weights...`);
                data.weightsManifest.forEach(group => {
                    group.weights.forEach(w => {
                        if (w.name.startsWith(prefix)) {
                            w.name = w.name.substring(prefix.length);
                        }
                    });
                });
            } else {
                // Maybe prefix is different? e.g. "sequential/"
                const seqPrefix = "sequential/";
                if (name.startsWith(seqPrefix)) {
                    console.log(`Stripping prefix "${seqPrefix}"...`);
                    data.weightsManifest.forEach(group => {
                        group.weights.forEach(w => {
                            if (w.name.startsWith(seqPrefix)) {
                                w.name = w.name.substring(seqPrefix.length);
                            }
                        });
                    });
                }
            }
        }

        fs.writeFileSync(outputPath, JSON.stringify(data, null, 2));
        console.log('✅ Unwrapped model saved to ' + outputPath);

    } else {
        console.log('Sequential wrapper not found or unexpected structure. Skipping unwrap.');
        // If it's already unwrapped or different, we do nothing.
    }

} catch (err) {
    console.error('❌ Error:', err);
}
