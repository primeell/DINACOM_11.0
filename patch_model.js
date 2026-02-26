import fs from 'fs';

const filePath = 'public/AI/model.json';

try {
    const raw = fs.readFileSync(filePath, 'utf8');
    // Replace "Functional" with "Model" to satisfy TFJS Layers loader
    const fixed = raw.replace(/"class_name": "Functional"/g, '"class_name": "Model"');
    fs.writeFileSync(filePath, fixed);
    console.log('Successfully patched model.json');
} catch (err) {
    console.error('Error patching file:', err);
}
