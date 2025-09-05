
import './App.css'
import placeHolderImage from './assets/Placeholder-_-Glossary.svg';

import { useState } from 'react';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [inputImage, setInputImage] = useState(null);
  const [outputImage, setOutputImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [depthData, setDepthData] = useState(null);
  const [hoverValue, setHoverValue] = useState(null);

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
    if (!selectedFile) {
      setError("Please select a file first.");
      return;
    }

    const formData = new FormData();
    formData.append("image", selectedFile);
    
    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.statusText}`);
      }

      const data = await response.json();
      console.log(data);
      setOutputImage(`data:image/png;base64,${data.depth_png}`);
      setDepthData(data.depth_raw);
      setError(null);
    } catch (err) {
      setError(`Upload failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  const handleMouseMove = (e) => {
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
        </div>
        
        <div className="UploadDisplay">

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
            Click Upload and Predict! Hover over the images to see depth values at specific points.
          </div>
        )}
        {hoverValue !== null && (
          <div className="DepthValue">
            Depth at cursor: {hoverValue.toFixed(2)} meters | {Math.round(hoverValue * 3.28084)} feet
          </div>
        )}
        </div>
        
        
        
        <div className="Footer">
          Depth Prediction Web App - Mitchel Tu
        </div>
      </div>
    </>
  )
}

export default App
