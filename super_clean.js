import fs from 'fs';
const inputPath = 'model.json';
const outputPath = 'public/AI/model.json';

try {
    const raw = fs.readFileSync(inputPath, 'utf8');
    let data = JSON.parse(raw);

    console.log('Original format:', data.format);

    // 1. Unwrap model_config if present (Keras 3 style)
    if (data.modelTopology && data.modelTopology.model_config) {
        console.log('Unwrapping model_config...');
        const mc = data.modelTopology.model_config;
        data.modelTopology.class_name = mc.class_name;
        data.modelTopology.config = mc.config;
        data.modelTopology.keras_version = '2.15.0'; // Downgrade version string to avoid panic
        delete data.modelTopology.model_config;
    }

    // 2. Recursive cleaning function
    function clean(obj) {
        if (Array.isArray(obj)) return obj.map(clean);
        if (obj && typeof obj === 'object') {
            // Fix DTypePolicy -> simple string
            if (obj.class_name === 'DTypePolicy' && obj.config && obj.config.name) {
                return obj.config.name;
            }

            // Fix batch_shape -> batch_input_shape (TFJS prefers this)
            if (obj.class_name === 'InputLayer' && obj.config && obj.config.batch_shape) {
                obj.config.batch_input_shape = obj.config.batch_shape;
                delete obj.config.batch_shape;
            }

            // Fix Functional -> Model
            if (obj.class_name === 'Functional') {
                obj.class_name = 'Model';
            }

            // Remove Keras 3 metadata
            delete obj.module;
            delete obj.registered_name;
            delete obj.build_config;
            delete obj.compile_config;

            // Recurse
            for (const key in obj) {
                obj[key] = clean(obj[key]);
            }
        }
        return obj;
    }

    data = clean(data);

    // 3. Ensure format is layers-model
    data.format = 'layers-model';

    fs.writeFileSync(outputPath, JSON.stringify(data, null, 2));
    console.log('✅ Success! Cleaned model saved to ' + outputPath);

} catch (err) {
    console.error('❌ Error during cleaning:', err);
}
