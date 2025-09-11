
import './App.css'
import placeHolderImage from './assets/Placeholder-_-Glossary.svg';
import React, { useState, useEffect } from 'react';
import { runModel, depthToColoredBase64, preprocessImage } from './model_runner';

function App() {
  
  const [selectedFile, setSelectedFile] = useState(null);
  const [inputImage, setInputImage] = useState(null);
  const [outputImage, setOutputImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [modelLoaded, setModelLoaded] = useState(false);
  const [depthData, setDepthData] = useState(null);

  const [hoverValue, setHoverValue] = useState(null);



    useEffect(() => {
    // preload the model with dummy data
    runModel(new Float32Array(3*256*256).fill(0)).then(() => setModelLoaded(true));
  }, []);

  

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setInputImage(URL.createObjectURL(event.target.files[0]));
    setLoading(null);
    setOutputImage(null);
    setError(null);
    setDepthData(null);
    setHoverValue(null);
  }

  const handleUpload = async () => {
    if (!selectedFile) return;

    // preprocess image here (resize, normalize)
    const {float32Data, originalWidth, originalHeight} = await preprocessImage(selectedFile);
    console.log({float32Data, originalWidth, originalHeight })
    // run inference
    const result = await runModel(float32Data);
    const depthArray = result.depthArray;
    console.log({depthArray})
    const coloredBase64 = await depthToColoredBase64(depthArray, 256, 256, originalWidth, originalHeight);

    // Convert depthArray (Float32Array) to 2D array for easier access
    const depth2D = [];
    for (let i = 0; i < 256; i++) {
      depth2D.push(depthArray.slice(i * 256, (i + 1) * 256));
    }
    setOutputImage(coloredBase64);
    setDepthData(depth2D);
    setLoading(false);
    setError(null);
    setHoverValue(null);
  };

  const handleMouseMove = (e) => {
    console.log({depthData})
    if (!depthData) return;
    const img = e.target;
    const rect = img.getBoundingClientRect();

    // Mouse position relative to image
    const x = Math.floor((e.clientX - rect.left) * (depthData[0].length / rect.width));
    const y = Math.floor((e.clientY - rect.top) * (depthData.length / rect.height));

    // Safety check
    if (x >= 0 && x < depthData[0].length && y >= 0 && y < depthData.length) {
      setHoverValue(depthData[y][x]);
    }
  };

  return (
    <>
      <div className="App">
        <div className="Title">
          <h1>Depth Prediction</h1>
        </div>
        <div className="Description">
          Predict depth from a single image using deep learning. 
          <br/>
          Take an indoor photo and see the estimated depth map! (Model trained on NYU Depth V2 dataset)
        </div>
        
        <div className="UploadDisplay">
          {!modelLoaded && <div className="Loading">⏳ Loading model...</div>}
          {modelLoaded && <div className="ModelReady">🤗 Model loaded! Upload an indoor image to get started.</div>}
          <div className="Uploader">
            <input type="file" accept="image/*" onChange={handleFileChange}/>
            <button onClick={handleUpload}>Upload & Predict</button>
          </div>
        </div>

        <div className="ImagesDisplay">

          <div className="InputImage">
            <h2>Input Image</h2>
            <img src={error && loading ? placeHolderImage : inputImage || placeHolderImage} alt="Input" onMouseMove={handleMouseMove} />
          </div>

          <div className="OutputImage">
            <h2>Predicted Depth Map</h2>
            <img src={error && loading ? placeHolderImage : outputImage || placeHolderImage} alt="Output" onMouseMove={handleMouseMove}/>
          </div>

        </div>
        <div className="StatusDisplay">
        {loading && <div className="Loading">Processing...</div>}
        {error && <div className="Error">{error}</div>}
        {hoverValue === null && !loading && !error && (
          <div className="Instructions">
            Hover over the images to see depth values at specific points.
          </div>
        )}
        {hoverValue !== null && (
          <div className="DepthValue">
            Depth at cursor: {hoverValue.toFixed(2)} meters | {Math.round(hoverValue * 3.28084)} feet
          </div>
        )}
        </div>
        
        
        
        <div className="Footer">
          Mitchel Tu → <a href="https://github.com/mitcheltu/DepthPredict"> Go to Git Repository</a>
        </div>
      </div>
    </>
  )
}

export default App
