import fs from 'fs';
const filePath = 'public/AI/model.json';
try {
    const raw = fs.readFileSync(filePath, 'utf8');
    let data = JSON.parse(raw);

    // 1. Structural Fix: Unwrap model_config
    if (data.modelTopology && data.modelTopology.model_config) {
        console.log('Found nested model_config. Unwrapping...');
        const nested = data.modelTopology.model_config;

        // Move class_name and config to modelTopology root
        data.modelTopology.class_name = nested.class_name;
        data.modelTopology.config = nested.config;

        // Remove the wrapper
        delete data.modelTopology.model_config;
    } else {
        console.log('No nested model_config found (already unwrapped?).');
    }

    // 2. Recursive Clean (Deep sanitize)
    function clean(item) {
        if (Array.isArray(item)) return item.map(clean);
        if (item && typeof item === 'object') {
            // Fix DTypePolicy -> string
            if (item.class_name === 'DTypePolicy' && item.config && item.config.name) return item.config.name;

            // Fix Functional -> Model
            if (item.class_name === 'Functional') item.class_name = 'Model';

            // Remove Keras 3 specific keys
            if (item.module && item.class_name) {
                delete item.module;
                delete item.registered_name;
            }

            delete item.build_config;
            delete item.compile_config;

            // Recurse
            for (const key in item) item[key] = clean(item[key]);
            return item;
        }
        return item;
    }
    data = clean(data);

    // 3. Force Metadata
    if (data.modelTopology) {
        data.modelTopology.keras_version = '2.15.0';
        data.modelTopology.backend = 'tensorflow';
    }

    fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
    console.log('Successfully restructured model.json for TFJS.');
} catch (e) {
    console.error('Fix script failed:', e);
    process.exit(1);
}
