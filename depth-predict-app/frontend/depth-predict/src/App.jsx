
import './App.css'
import placeHolderImage from './assets/Placeholder-_-Glossary.svg';

function App() {

  return (
    <>
      <div className="App">
        <div className="Title">
          <h1>Depth Prediction Web App</h1>
        </div>
        <div className="Description">
          Predict depth from a single image using deep learning.
        </div>
        
        <div className="UploadDisplay">

          <div className="Uploader">
            <input type="file" />
          </div>
        </div>

        <div className="ImagesDisplay">

          <div className="InputImage">
            <h2>Input Image</h2>
            <img src={placeHolderImage} alt="Input" />
          </div>

          <div className="OutputImage">
            <h2>Predicted Depth Map</h2>
            <img src={placeHolderImage} alt="Output" />
          </div>

        </div>
        
        
        <div className="Footer">
          Depth Prediction Web App - Mitchel Tu
        </div>
      </div>
    </>
  )
}

export default App
