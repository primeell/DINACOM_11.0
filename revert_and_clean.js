import fs from 'fs';
const filePath = 'public/AI/model.json';
const backupPath = 'public/AI/model.json.bak';

try {
    // 1. Restore from backup if exists
    if (fs.existsSync(backupPath)) {
        console.log('Restoring from ' + backupPath);
        fs.copyFileSync(backupPath, filePath);
    } else {
        console.log('No backup found, using current file.');
    }

    const raw = fs.readFileSync(filePath, 'utf8');
    let data = JSON.parse(raw);

    // 2. Clean WITHOUT renaming Functional
    function clean(item) {
        if (Array.isArray(item)) return item.map(clean);
        if (item && typeof item === 'object') {
            // Fix DTypePolicy -> string
            if (item.class_name === 'DTypePolicy' && item.config && item.config.name) return item.config.name;

            // Remove Keras 3 artifacts
            if (item.module && item.class_name) {
                delete item.module;
                delete item.registered_name;
            }
            delete item.build_config;
            delete item.compile_config;

            // Revert Model -> Functional if we accidentally changed it and now want to try Functional?
            // Or if the backup had "Model" (from patch_model.js), we might want to try "Functional"?
            // Let's assume TFJS 4.22 might prefer "Functional" if it matches Keras 3?
            // Actually, let's keep it as is from backup (which was "Model" if patch_model.js ran).
            // Wait, let's try setting it to "Functional" explicitly?
            if (item.class_name === 'Model') {
                // item.class_name = 'Functional'; // Uncomment to try Functional
            }

            for (const key in item) item[key] = clean(item[key]);
            return item;
        }
        return item;
    }

    data = clean(data);

    if (data.modelTopology) {
        data.modelTopology.keras_version = '2.15.0';
        data.modelTopology.backend = 'tensorflow';
    }

    fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
    console.log('Restored and cleaned (DTypePolicy removed).');

} catch (e) {
    console.error(e);
}
