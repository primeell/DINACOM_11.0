import fs from 'fs';
const filePath = 'public/AI/model.json';
try {
    if (fs.existsSync(filePath + '.bak')) fs.unlinkSync(filePath + '.bak');
    fs.copyFileSync(filePath, filePath + '.bak');
    const raw = fs.readFileSync(filePath, 'utf8');
    let data = JSON.parse(raw);
    function clean(item) {
        if (Array.isArray(item)) return item.map(clean);
        if (item && typeof item === 'object') {
            if (item.class_name === 'DTypePolicy' && item.config && item.config.name) return item.config.name;
            if (item.class_name === 'Functional') item.class_name = 'Model';
            if (item.module && item.class_name) { delete item.module; delete item.registered_name; }
            delete item.build_config;
            delete item.compile_config;
            for (const key in item) item[key] = clean(item[key]);
            return item;
        }
        return item;
    }
    data = clean(data);
    if (data.modelTopology) data.modelTopology.keras_version = '2.15.0';
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
    console.log('Cleaned Keras3 metadata.');
} catch (e) {
    console.error(e);
    process.exit(1);
}
