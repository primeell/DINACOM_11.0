import * as tf from '@tensorflow/tfjs-node';

async function testLoad() {
    try {
        console.log("Testing loadLayersModel...");
        const model = await tf.loadLayersModel('file://public/AI/model.json');
        console.log("Successfully loaded layers model!");
        model.summary();
    } catch (e) {
        console.error("Failed to load layers model:");
        console.error(e);

        console.log("\nTesting loadGraphModel instead...");
        try {
            const graphModel = await tf.loadGraphModel('file://public/AI/model.json');
            console.log("Successfully loaded graph model!");
        } catch (e2) {
            console.error("Failed to load graph model:");
            console.error(e2);
        }
    }
}

testLoad();
