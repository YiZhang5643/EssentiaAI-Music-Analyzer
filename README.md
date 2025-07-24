# EssentiaAI Music Analyzer

🎵 **A powerful AI-driven music analysis tool built with Essentia and TensorFlow**

This tool provides comprehensive music analysis using pre-trained deep learning models, extracting advanced features including genre classification, mood detection, instrument recognition, and detailed audio characteristics.

## ✨ Features

### 🎯 AI-Powered Analysis
- **Genre Classification**: 400+ genres using Discogs database + 87 genres using MTG-Jamendo dataset
- **Mood Detection**: Happy/sad mood classification with confidence scores
- **Instrument Recognition**: Multi-instrument detection and confidence estimation
- **Voice/Instrumental Classification**: Automatic vocal content detection

### 🎼 Comprehensive Audio Analysis
- **Basic Features**: BPM, duration, sample rate
- **Tonal Analysis**: Key detection, chord analysis, tonal strength
- **Spectral Features**: Spectral centroid, rolloff, zero-crossing rate
- **Rhythm Analysis**: Beat tracking, tempo estimation

### 🔧 Advanced Capabilities
- **578+ Audio Features**: Rhythm (31), Tonal (59), Low-level (470), Metadata (18)
- **Pre-trained Models**: Automatic download and management of Essentia AI models
- **Multiple Formats**: Support for MP3, WAV, FLAC, OGG, M4A
- **Batch Processing**: Easy file selection and analysis workflow

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Ubuntu/Linux (recommended)
- ~100MB free disk space for AI models

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/YiZhang5643/EssentiaAI-Music-Analyzer.git
cd EssentiaAI-Music-Analyzer
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure music directory:**
Edit the `music_directory` variable in the script:
```python
music_directory = "/path/to/your/music/files/"  # Update this path
```

4. **Run the analyzer:**
```bash
python music_analyzer.py
```

### First Run Setup

When you first run the tool, it will offer to download pre-trained AI models:

```
🎵 Essentia Music Analysis Tool 🎵
📁 Found 5 music files:
  [1] song1.mp3
  [2] song2.wav
  ...
  [0] Show detailed installation guide
  [-1] Check Essentia installation status  
  [-2] Download pre-trained models

Please enter your choice (-2, -1, 0, 1-5): -2
```

Select `[-2]` to automatically download all required AI models (~100MB).

## 📊 Usage

### Basic Analysis
1. Run the script: `python music_analyzer.py`
2. Select a music file from the list
3. Wait for analysis to complete
4. View comprehensive results

### Menu Options
- **[-2]**: Download/update AI models
- **[-1]**: Check installation and model status
- **[0]**: Show detailed setup guide
- **[1-N]**: Analyze specific music file

## 📈 Sample Output

The tool generates detailed analysis reports with multiple categories of features:

```
--------------------------------------------------
🎵 Music Analysis Report 🎵
--------------------------------------------------
📊 Available Feature List:
  RHYTHM: 31 features
  TONAL: 59 features  
  LOWLEVEL: 470 features
  METADATA: 18 features

==================================================
🎵 Detailed Analysis Results 🎵
==================================================
🎶 Basic Audio Features:
  Tempo (BPM): 155.59
  Duration: 38.90 seconds
  Sample Rate: 44100.0 Hz

🎼 Tonal Features:
  Key: B [tonal.key_temperley.key]
  Chord Key: B major

🎨 Advanced Feature Analysis:
  🎵 Discogs Genre Analysis (Top 10):
     1. Electronic---Vaporwave (confidence: 0.434)
     2. Electronic---Experimental (confidence: 0.394)
     3. Electronic---Darkwave (confidence: 0.052)
     ...

  🎵 Jamendo Genre Analysis (Top 10):
     1. experimental         (confidence: 0.550)
     2. alternative          (confidence: 0.547)
     3. electronic           (confidence: 0.406)
     ...

  🎸 Detected Instruments:
    1. synthesizer     (confidence: 0.462)
    2. guitar          (confidence: 0.147)
    3. drummachine     (confidence: 0.128)
    ...

  😊 Mood: happy (confidence: 0.942)
  🎤 Audio Type: vocal (confidence: 0.708)

🎼 Spectral Features:
  Spectral Centroid: 1207.64 Hz
  Spectral Rolloff: 1879.44 Hz
  Zero Crossing Rate: 0.0698
--------------------------------------------------
```

## 🧠 AI Models

The tool uses state-of-the-art pre-trained models from Essentia:

### Genre Classification
- **Discogs-400**: 400 detailed electronic music genres
- **MTG-Jamendo**: 87 general music genres

### Audio Content Analysis  
- **Mood Detection**: Happy/sad emotion classification
- **Instrument Recognition**: Multi-instrument detection
- **Voice Classification**: Vocal vs. instrumental content

### Technical Details
- **Feature Extraction**: Discogs-EfficientNet embeddings
- **Model Format**: TensorFlow .pb files
- **Input**: 30-second audio segments (automatically handled)
- **Output**: Confidence scores and rankings

## 🔧 Technical Requirements

### Core Dependencies
- `essentia-tensorflow>=2.1b5` - Main audio analysis library
- `tensorflow>=2.19.0` - Deep learning framework
- `numpy>=1.26.4` - Numerical computing
- `scipy>=1.13.1` - Scientific computing
- `librosa>=0.11.0` - Audio processing utilities

### System Requirements
- **OS**: Ubuntu 18.04+ (Linux recommended)
- **Python**: 3.9 or higher
- **Memory**: 4GB+ RAM recommended
- **Storage**: 100MB+ free space (including models)
- **CPU**: Multi-core processor recommended for faster analysis

## 🐛 Troubleshooting

### Common Issues

**Models not downloading:**
```bash
# Check network connection and try manual download
python music_analyzer.py
# Select [-1] to check status, then [-2] to retry download
```

**TensorFlow errors:**
```bash
# Ensure compatible TensorFlow version
pip install tensorflow==2.19.0
```

**Audio format not supported:**
```bash
# Install additional audio codecs
sudo apt-get install ffmpeg
pip install ffmpeg-python
```

**Memory issues with large files:**
- Use shorter audio files (<5 minutes recommended)
- Close other applications to free RAM
- Consider using a machine with more memory

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Ways to Contribute
- 🐛 **Bug Reports**: Report issues or unexpected behavior
- 💡 **Feature Requests**: Suggest new analysis capabilities
- 📝 **Documentation**: Improve README, add examples
- 🔧 **Code**: Submit pull requests with improvements
- 🎵 **Testing**: Test with different audio formats and genres

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Submit a pull request with clear description

### Code Style
- Follow PEP 8 conventions
- Add comments for complex functions
- Include error handling for new features
- Test with various audio file types

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Essentia Team**: For the incredible audio analysis library
- **MTG Barcelona**: For the research and pre-trained models  
- **Discogs Database**: For comprehensive music genre data
- **TensorFlow Team**: For the deep learning framework

## 📚 References

- [Essentia Official Documentation](https://essentia.upf.edu/)
- [MTG-Jamendo Dataset](https://github.com/MTG/mtg-jamendo-dataset)
- [Discogs Genre Classification](https://essentia.upf.edu/models/)
- [Music Information Retrieval Research](https://www.ismir.net/)

## 🔗 Related Projects

- [Essentia.js](https://essentia.upf.edu/docs/api/javascript_api.html) - Browser-based audio analysis
- [librosa](https://librosa.org/) - Python audio analysis library
- [AudioSet](https://research.google.com/audioset/) - Google's audio event dataset

---

⭐ **Star this repository if you find it useful!** ⭐

For questions, issues, or suggestions, please open an issue or contact the maintainers.
