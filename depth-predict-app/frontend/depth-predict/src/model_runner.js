import * as ort from 'onnxruntime-web';

ort.env.wasm.wasmPaths = 'onnx/';

let session = null;

/**
 * Load the ONNX model (only once)
 */
export async function loadModel() {
  if (!session) {
    try {
        session = await ort.InferenceSession.create('/depth_model.onnx', {
            executionProviders: ['webgl', 'wasm'], 
    });
    }
    catch (e) {
        console.error('Failed to load ONNX model:', e);
        throw e;
    }
    
    console.log('ONNX model loaded (WebGL)');
  }
  return session;
}

/**
 * Run inference on a preprocessed Float32 array
 * @param {Float32Array} inputArray - shape [3*256*256], normalized RGB
 * @returns output tensor from model
 */
export async function runModel(inputArray) {
    const sess = await loadModel();

    // ONNX expects shape [1, 3, 256, 256]
    console.log({inputArray})
    const inputTensor = new ort.Tensor('float32', inputArray, [1, 3, 256, 256]);

    // Run inference
    const outputMap = await sess.run({ "input.1": inputTensor });

    // Usually outputMap has a key like 'output' or 'out'
    // Adjust based on your model's actual output key
    const outputKey = Object.keys(outputMap)[0];
    const depthArray = outputMap[outputKey].data; // Float32Array
    console.log({depthArray})
    return { depthArray}; // Float32Array of depth values
}

/**
 * Preprocess an uploaded image file into Float32Array
 * Resize to 256x256, normalize 0–1, channel-first [3, 256, 256]
 */
export async function preprocessImage(file) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            const originalWidth = img.width;
            const originalHeight = img.height;

            const canvas = document.createElement('canvas');
            canvas.width = 256;
            canvas.height = 256;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, 256, 256);

            const imageData = ctx.getImageData(0, 0, 256, 256);
            const data = imageData.data;
            const float32Data = new Float32Array(3 * 256 * 256);

            // Convert to channel-first RGB and normalize
            for (let i = 0; i < 256 * 256; i++) {
                float32Data[i] = data[i * 4] / 255;                 // R
                float32Data[i + 256 * 256] = data[i * 4 + 1] / 255; // G
                float32Data[i + 2 * 256 * 256] = data[i * 4 + 2] / 255; // B
            }
            console.log({float32Data})
            resolve({float32Data, originalWidth, originalHeight});
        };
        img.src = URL.createObjectURL(file);
    });
}

/** Convert depth array to colored base64 PNG (hot colormap) */
export async function depthToColoredBase64(depthArray, width, height, targetWidth, targetHeight) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(width, height);
    const data = imageData.data;

    // Normalize depth to 0-1
    const { min, max } = getMinMax(depthArray);
    const scale = 1 / (max - min + 1e-8);

    for (let i = 0; i < width * height; i++) {
        let value = (depthArray[i] - min) * scale; // normalized
        // Apply simple hot colormap: R=val, G=val*0.5, B=0
        data[i * 4] = Math.floor(value * 255);       // R
        data[i * 4 + 1] = Math.floor(value * 128);   // G
        data[i * 4 + 2] = 0;                          // B
        data[i * 4 + 3] = 255;                        // Alpha
    }

    ctx.putImageData(imageData, 0, 0);

    // Now resize canvas to original image size
    const finalCanvas = document.createElement('canvas');
    finalCanvas.width = targetWidth;
    finalCanvas.height = targetHeight;
    const fCtx = finalCanvas.getContext('2d');
    fCtx.drawImage(canvas, 0, 0, targetWidth, targetHeight);


    return finalCanvas.toDataURL('image/png'); // base64 PNG
}

function getMinMax(arr) {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (Number.isFinite(v)) { // ignore NaN / Infinity
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  return { min, max };
}