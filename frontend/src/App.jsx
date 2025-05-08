// App.jsx - Main React application
import { useState } from 'react';
import './App.css';
import ResultDisplay from './components/ResultDisplay';
import FactCheckSuggestions from './components/FactCheckSuggestions';
import LoadingSpinner from './components/LoadingSpinner';

function App() {
  const [inputType, setInputType] = useState('text');
  const [inputContent, setInputContent] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    
    try {
      // Form validation
      if (!inputContent.trim()) {
        throw new Error('Please enter some text or a URL to analyze');
      }
      
      // URL validation if URL type is selected
      if (inputType === 'url' && !isValidUrl(inputContent)) {
        throw new Error('Please enter a valid URL');
      }

      const response = await fetch('http://localhost:5000/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          type: inputType,
          content: inputContent
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to analyze content');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setIsLoading(false);
    }
  };

  const isValidUrl = (string) => {
    try {
      new URL(string);
      return true;
    } catch (_) {
      return false;
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center pt-16 px-4">
      <div className="w-full max-w-2xl bg-white rounded-lg shadow-xl p-8">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-800">Fake News Detector</h1>
          <p className="text-gray-600 mt-2">
            Analyze news articles or social media posts for potential misinformation
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="flex items-center justify-center space-x-4">
            <label className="inline-flex items-center">
              <input
                type="radio"
                value="text"
                checked={inputType === 'text'}
                onChange={() => setInputType('text')}
                className="form-radio h-4 w-4 text-blue-600"
              />
              <span className="ml-2 text-gray-700">Analyze Text</span>
            </label>
            <label className="inline-flex items-center">
              <input
                type="radio"
                value="url"
                checked={inputType === 'url'}
                onChange={() => setInputType('url')}
                className="form-radio h-4 w-4 text-blue-600"
              />
              <span className="ml-2 text-gray-700">Analyze URL</span>
            </label>
          </div>

          <div>
            {inputType === 'text' ? (
              <textarea
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                rows="5"
                placeholder="Paste article text here..."
                value={inputContent}
                onChange={(e) => setInputContent(e.target.value)}
              ></textarea>
            ) : (
              <input
                type="url"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Enter article URL..."
                value={inputContent}
                onChange={(e) => setInputContent(e.target.value)}
              />
            )}
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full bg-blue-600 text-white font-medium py-3 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors disabled:bg-blue-400"
          >
            {isLoading ? 'Analyzing...' : 'Analyze Content'}
          </button>
        </form>

        {error && (
          <div className="mt-6 p-4 bg-red-100 border-l-4 border-red-500 text-red-700">
            <p>{error}</p>
          </div>
        )}

        {isLoading && (
          <div className="mt-6 flex justify-center">
            <LoadingSpinner />
          </div>
        )}

        {result && !isLoading && (
          <div className="mt-8 space-y-6">
            <ResultDisplay result={result} />
            <FactCheckSuggestions category={result.category} />
          </div>
        )}
      </div>
      
      <footer className="mt-8 text-center text-gray-500 text-sm pb-6">
        <p>This is a hackathon project. Results should not be considered definitive.</p>
      </footer>
    </div>
  );
}

export default App;
