# Instagram AI Demos

Educational AI/ML projects designed for Instagram Reels content - demonstrating machine learning concepts through visual, engaging demos.

## Projects

### 1. Fake News Detector (CNN-LSTM)
Location: `AI_Fake_News_Detector_LSTM_Project/`

A neural network that analyzes news headlines to detect potential misinformation patterns.

**Architecture:**
- **Input Layer**: Text headlines (max 20 tokens)
- **Embedding Layer**: 5000 vocab → 100 dimensions
- **CNN Layer**: 64 filters, kernel size 3 (pattern detection)
- **MaxPooling**: Dimensionality reduction
- **LSTM Layer**: 64 units (sequential understanding)
- **Dense Layers**: 32 → 16 neurons
- **Output**: Binary classification (Real/Fake)

**Performance:**
- Accuracy: 87%
- Precision: 85%
- Recall: 89%

**Tech Stack:**
- TensorFlow/Keras
- Python 3.10+
- Plotly (3D visualizations)
- Matplotlib (static visuals)

## Installation

```bash
# Clone repository
git clone https://github.com/kira-ml/instagram-ai-demos.git
cd instagram-ai-demos

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install tensorflow keras numpy pandas matplotlib plotly scikit-learn pillow
```

## Usage

### Run Demo UI
```bash
cd AI_Fake_News_Detector_LSTM_Project
python demo_ui.py
```

### Run Core Training
```bash
python fake_news_detector_reel.py
```

### Generate 3D Visualizations
```bash
python architecture_3d_visual.py
```

### Generate PIL-based Visuals
```bash
python pil_architecture_visual.py
```

## Project Structure

```
instagram-ai-demos/
├── AI_Fake_News_Detector_LSTM_Project/
│   ├── fake_news_detector_reel.py    # Core ML model
│   ├── demo_ui.py                     # Interactive demo
│   ├── architecture_3d_visual.py      # 3D network visualization
│   ├── pil_architecture_visual.py     # Clean static visuals
│   ├── voiceover_script.md            # Instagram script
│   └── README.md                      # Project details
└── README.md                          # This file
```

## Features

✅ Production-grade code structure  
✅ CNN-LSTM hybrid architecture  
✅ Interactive 3D network visualization  
✅ Instagram Reels optimized visuals (1080x1080)  
✅ Educational demonstrations  
✅ Real-time headline analysis  

## Educational Purpose

These projects are designed for:
- Teaching ML concepts visually
- Instagram/TikTok educational content
- Understanding neural network architectures
- Demonstrating AI applications in misinformation detection

**Not for production use** - these are educational demonstrations to help people understand how AI works.

## Contributing

This is an educational project for Instagram content. Feel free to fork and create your own demos!

## License

MIT License - Educational use encouraged

## Author

**Ken Ira Talingting**  
Instagram: [@your_handle]  
GitHub: [@kira-ml](https://github.com/kira-ml)

---

**Note:** Models and visualizations are simplified for educational purposes. Always verify information from multiple credible sources.
