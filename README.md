# 🧠 AI/ML Demos for Learners

> **Hey there!** I'm Kira, an ML student on a mission to make machine learning accessible to everyone. This repo is where I share projects I build while learning—because the best way to understand AI is to build it yourself.

Welcome to my learning journey! 🚀 These aren't "perfect production systems"—they're educational demos designed to help you (and me) understand how neural networks actually work under the hood.

## 🎯 Current Project: Fake News Detector

**What it does:** Catches misinformation patterns before you fall for them

Ever wonder why fake news headlines feel... different? I built a CNN-LSTM neural network to figure out exactly what those patterns are. Trained on 25,000+ real and fake headlines, this model learned to spot the warning signs.

### 🏗️ The Architecture (What I Learned)

This was my first time combining CNNs and LSTMs, and honestly? It was mind-blowing seeing how they complement each other:

- **Input Layer**: Feeds in headlines (up to 20 words)
- **Embedding Layer**: Converts 5,000 vocab words → 100-dimensional vectors (this is where meaning happens!)
- **CNN Layer**: 64 filters scan for patterns like "ALL CAPS" or "!!!" (local feature detection)
- **MaxPooling**: Keeps the important stuff, drops the noise
- **LSTM Layer**: 64 units remember word sequences (because "SHOCKING" + "doctors HATE" = red flag)
- **Dense Layers**: 32 → 16 neurons compress everything down
- **Output**: One neuron says "real" or "fake" with confidence score

### 📊 Results (Real Talk)

- **87% Accuracy** - Not perfect, but solid for a learning project
- **85% Precision** - When it says "fake," it's usually right
- **89% Recall** - Catches most fake headlines (this is the important one!)

**Dataset:** [ISOT Fake News Detection](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)
- 12,600+ real articles from Reuters (2016-2017)
- 12,600+ fake articles from Politifact-flagged sources
- Political & world news focus

### 🛠️ Tech Stack I Used

```
TensorFlow/Keras  → Building the neural network
Python 3.10+      → Core language
Plotly            → 3D architecture visualizations (so cool!)
Matplotlib        → Static visuals for Instagram
scikit-learn      → Train/test splits & metrics
PIL/Pillow        → Clean graphics for social media
```

## 🚀 Want to Run It Yourself?

I built this so YOU can learn from it too. Here's how to get started:

### Quick Start (5 minutes)

```bash
# 1. Grab the code
git clone https://github.com/kira-ml/instagram-ai-demos.git
cd instagram-ai-demos

# 2. Set up your environment (keeps things clean)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install what you need
pip install tensorflow keras numpy pandas matplotlib plotly scikit-learn pillow
```

### What You Can Do

**Try the Interactive Demo:**
```bash
cd AI_Fake_News_Detector_LSTM_Project
python demo_ui.py
# Type any headline → get instant analysis!
```

**Train Your Own Model:**
```bash
python fake_news_detector_reel.py
# Watch the training happen in real-time
```

**Generate 3D Visualizations:**
```bash
python architecture_3d_visual.py
# Creates rotating 3D network diagrams (perfect for understanding architecture)
```

**Make Instagram-Ready Graphics:**
```bash
python pil_architecture_visual.py
# Generates clean, readable visuals for social media
```

## 📂 How It's Organized

```
instagram-ai-demos/
├── AI_Fake_News_Detector_LSTM_Project/
│   ├── fake_news_detector_reel.py    # The main model (start here!)
│   ├── demo_ui.py                     # Try it yourself
│   ├── architecture_3d_visual.py      # 3D network viz
│   ├── pil_architecture_visual.py     # Social media graphics
│   ├── voiceover_script.md            # My Instagram script
│   └── instagram_caption.md           # Social media captions
└── README.md                          # You are here
```

## ✨ What Makes This Different

✅ **Real Code, Real Learning** - No shortcuts, proper ML practices  
✅ **Visual Learning** - 3D architecture diagrams help you *see* how it works  
✅ **Instagram-Optimized** - All visuals ready for social media (1080x1080)  
✅ **Interactive Demos** - Try it yourself, break it, learn from it  
✅ **Transparent Results** - I show the actual metrics, not inflated numbers  
✅ **Beginner-Friendly** - Comments explain WHY, not just WHAT  

## 💡 Why I Built This

As a student learning ML, I noticed something: most tutorials either oversimplify (hello, MNIST for the 100th time) or assume you already know everything. 

**I wanted something in between.**

This project tackles a real problem (misinformation), uses real data (ISOT dataset), and achieves real results (87% accuracy). But it's also:
- **Documented** so you understand each decision
- **Visual** so you can *see* what's happening
- **Shareable** so you can teach others

**The goal?** Help you understand ML deeply enough to build your own projects and explain them confidently.

## 🤝 Learning Together

I'm still learning, and that's the point! If you:
- Find bugs → Open an issue (we'll debug together)
- Have questions → Start a discussion (I love explaining ML!)
- Built something cool → Share it with me on Instagram!
- Want to improve this → PRs welcome (let's make it better)

**This is for learners, by a learner.** We're all figuring this out together. 🌱

## 📚 What I Learned Building This

Some real talk about the challenges:
1. **CNNs + LSTMs** are powerful together, but hyperparameter tuning is an art
2. **Data quality > Data quantity** - cleaning the dataset mattered more than size
3. **Overfitting is REAL** - dropout layers saved me
4. **Embeddings are magical** - watching words become meaningful vectors never gets old
5. **Visualization helps** - I understood my model 10x better after making those 3D diagrams

## ⚖️ Important Notes

🎓 **Educational Purpose**: This is a learning project, not production software. Use it to learn, teach, and understand ML—not as a real fact-checker.

✓ **Always Verify**: AI can help spot patterns, but critical thinking and source verification are irreplaceable.

📖 **Open Learning**: All code is open-source (MIT License) because education should be accessible.

## 🌐 Connect & Learn Together

**Ken Ira Talingting (Kira)**  
📸 Instagram: [@keniii_lacson](https://instagram.com/keniii_lacson) - Follow for ML tutorials!  
💻 GitHub: [@kira-ml](https://github.com/kira-ml) - More projects here  
🎥 Content: Educational ML demos for Instagram Reels  

*Building AI projects in public. Learning in public. Teaching in public.*

---

### 🎁 One More Thing

If this helped you understand neural networks better, star this repo ⭐ so others can find it too!

Have questions? Don't hesitate to reach out. Seriously—I'm here to help you learn. That's the whole point. 🚀

**Happy Learning!**  
— Kira
