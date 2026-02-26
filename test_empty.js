import * as tf from '@tensorflow/tfjs-node';

async function test() {
    const model = await tf.loadLayersModel('file://public/AI/model.json');

    // Test with completely empty audio (spectrogram is all -1)
    const emptyTensor = tf.fill([1, 224, 224, 3], -1);
    const pred1 = model.predict(emptyTensor);
    const data1 = await pred1.data();
    console.log("Empty audio:", data1);

    // Test with completely full audio (spectrogram is all +1)
    const fullTensor = tf.fill([1, 224, 224, 3], 1);
    const pred2 = model.predict(fullTensor);
    const data2 = await pred2.data();
    console.log("Full audio:", data2);

    // Test with zeros
    const zeroTensor = tf.zeros([1, 224, 224, 3]);
    const pred3 = model.predict(zeroTensor);
    const data3 = await pred3.data();
    console.log("Zero audio:", data3);
}

test();
