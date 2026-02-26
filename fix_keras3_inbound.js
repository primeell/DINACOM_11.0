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
        data.modelTopology.keras_version = '2.15.0';
        delete data.modelTopology.model_config;
    }

    // Helper to convert a Keras 3 tensor ref to Keras 2 style
    function convertTensor(tensor) {
        if (tensor && tensor.class_name === '__keras_tensor__' && tensor.config && tensor.config.keras_history) {
            const history = tensor.config.keras_history;
            return [history[0], history[1], history[2], {}];
        }
        return null;
    }

    // 2. Recursive cleaning function
    function clean(obj) {
        if (Array.isArray(obj)) return obj.map(clean);
        if (obj && typeof obj === 'object') {
            // Fix DTypePolicy -> simple string
            if (obj.class_name === 'DTypePolicy' && obj.config && obj.config.name) {
                return obj.config.name;
            }

            // Fix batch_shape -> batch_input_shape
            if (obj.class_name === 'InputLayer' && obj.config && obj.config.batch_shape) {
                obj.config.batch_input_shape = obj.config.batch_shape;
                delete obj.config.batch_shape;
            }

            // Fix Functional -> Model
            if (obj.class_name === 'Functional') {
                obj.class_name = 'Model';
            }

            // --- THE CRITICAL FIX: inbound_nodes ---
            if (obj.inbound_nodes && Array.isArray(obj.inbound_nodes)) {
                const newNodes = [];
                for (const node of obj.inbound_nodes) {
                    if (node && typeof node === 'object' && !Array.isArray(node)) {
                        // Keras 3 style: { args: [...], kwargs: {...} }
                        const args = node.args || [];
                        const flatArgs = [];

                        for (const arg of args) {
                            if (Array.isArray(arg)) {
                                // Nested array (e.g. for Add/Concatenate)
                                const nested = arg.map(convertTensor).filter(x => x !== null);
                                if (nested.length > 0) flatArgs.push(nested);
                            } else {
                                const converted = convertTensor(arg);
                                if (converted) flatArgs.push(converted);
                            }
                        }

                        // Keras 2 expects an array of lists of [layer, node, tensor, config]
                        // Most layers are [[ [prev, 0, 0, {}] ]]
                        // If flatArgs is [ [prev, 0, 0, {}] ], we wrap it once more if needed.
                        if (flatArgs.length > 0) {
                            newNodes.push(flatArgs);
                        }
                    } else {
                        // Already Keras 2 style or something else? Keep it.
                        newNodes.push(node);
                    }
                }
                obj.inbound_nodes = newNodes;
            }

            // Remove Keras 3 metadata
            delete obj.module;
            delete obj.registered_name;
            delete obj.build_config;
            delete obj.compile_config;

            // Recurse
            for (const key in obj) {
                if (key !== 'inbound_nodes') { // already handled
                    obj[key] = clean(obj[key]);
                }
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
