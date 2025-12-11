"""
Instagram Reels Demo UI - Fake News Detector
Interactive visual interface for showcasing ML model predictions
Perfect for screen recording and social media content
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
import random
from pathlib import Path
from typing import Optional
import sys

# Try to import the model
try:
    from fake_news_detector_reel import FakeNewsModel, TextPreprocessor, ModelConfig
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("⚠️  Model not available. Running in demo mode.")


class FakeNewsDetectorUI:
    """Modern, Instagram-ready UI for fake news detection demo."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Fake News Detector - Instagram Reels Demo")
        self.root.geometry("1080x1920")  # 9:16 ratio for vertical video
        self.root.configure(bg="#0F0F1E")
        
        # Make window scrollable
        self.canvas = tk.Canvas(self.root, bg="#0F0F1E", highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#0F0F1E")
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Enable mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Model components
        self.model = None
        self.preprocessor = None
        self.is_analyzing = False
        
        # Color scheme
        self.colors = {
            'bg_dark': '#0F0F1E',
            'bg_card': '#1A1A2E',
            'accent': '#00D9FF',
            'accent_glow': '#00FFF0',
            'real': '#00FF88',
            'fake': '#FF3366',
            'text': '#E0E0E0',
            'text_dim': '#808080',
            'warning': '#FFB800'
        }
        
        # Sample headlines for demo
        self.sample_headlines = {
            'real': [
                "Scientists discover new species in Amazon rainforest",
                "Stock market shows steady growth in quarterly report",
                "New climate policy announced at international summit",
                "Researchers develop improved battery technology",
                "Local government approves infrastructure funding"
            ],
            'fake': [
                "SHOCKING: This ONE trick doctors don't want you to know!",
                "Celebrity EXPOSED in leaked documents (you won't believe this)",
                "They're HIDING the truth about this common food!",
                "BREAKING: Scandal rocks Washington elite (VIRAL)",
                "This MIRACLE cure will change EVERYTHING (watch now)"
            ]
        }
        
        self._setup_ui()
        self._load_model_async()
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
    def _setup_ui(self):
        """Create the main UI layout."""
        # Main container - use scrollable_frame instead of root
        main_frame = tk.Frame(self.scrollable_frame, bg=self.colors['bg_dark'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header Section
        self._create_header(main_frame)
        
        # Split Screen Section (Real vs Fake comparison)
        self._create_split_screen(main_frame)
        
        # Input Section
        self._create_input_section(main_frame)
        
        # Analysis Section
        self._create_analysis_section(main_frame)
        
        # Quick Demo Buttons
        self._create_quick_demo_buttons(main_frame)
        
        # Footer
        self._create_footer(main_frame)
        
    def _create_header(self, parent):
        """Create animated header with title."""
        header = tk.Frame(parent, bg=self.colors['bg_dark'])
        header.pack(fill=tk.X, pady=(0, 20))
        
        # Main title with glow effect
        title = tk.Label(
            header,
            text="🤖 AI FAKE NEWS DETECTOR",
            font=("Arial", 36, "bold"),
            fg=self.colors['accent_glow'],
            bg=self.colors['bg_dark']
        )
        title.pack()
        
        # Subtitle
        subtitle = tk.Label(
            header,
            text="Powered by Deep Learning LSTM Network",
            font=("Arial", 14),
            fg=self.colors['text_dim'],
            bg=self.colors['bg_dark']
        )
        subtitle.pack(pady=(5, 0))
        
        # Status indicator
        self.status_label = tk.Label(
            header,
            text="⚙️ Loading Model...",
            font=("Arial", 12),
            fg=self.colors['warning'],
            bg=self.colors['bg_dark']
        )
        self.status_label.pack(pady=(10, 0))
        
    def _create_split_screen(self, parent):
        """Create split screen showing real vs fake examples."""
        split_container = tk.Frame(parent, bg=self.colors['bg_dark'])
        split_container.pack(fill=tk.X, pady=(0, 20))
        
        # Title
        title = tk.Label(
            split_container,
            text="📊 REAL vs FAKE PATTERNS",
            font=("Arial", 18, "bold"),
            fg=self.colors['text'],
            bg=self.colors['bg_dark']
        )
        title.pack(pady=(0, 10))
        
        # Split frame
        split_frame = tk.Frame(split_container, bg=self.colors['bg_dark'])
        split_frame.pack(fill=tk.X)
        
        # Left: Real News
        left_frame = tk.Frame(split_frame, bg=self.colors['bg_card'], relief=tk.RAISED, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        tk.Label(
            left_frame,
            text="✅ TRUSTWORTHY",
            font=("Arial", 16, "bold"),
            fg=self.colors['real'],
            bg=self.colors['bg_card']
        ).pack(pady=10)
        
        self.real_example = tk.Label(
            left_frame,
            text=self.sample_headlines['real'][0],
            font=("Arial", 11),
            fg=self.colors['text'],
            bg=self.colors['bg_card'],
            wraplength=450,
            justify=tk.CENTER,
            height=4
        )
        self.real_example.pack(padx=15, pady=(0, 15))
        
        # Right: Fake News
        right_frame = tk.Frame(split_frame, bg=self.colors['bg_card'], relief=tk.RAISED, bd=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        tk.Label(
            right_frame,
            text="⚠️ SUSPECT",
            font=("Arial", 16, "bold"),
            fg=self.colors['fake'],
            bg=self.colors['bg_card']
        ).pack(pady=10)
        
        self.fake_example = tk.Label(
            right_frame,
            text=self.sample_headlines['fake'][0],
            font=("Arial", 11),
            fg=self.colors['text'],
            bg=self.colors['bg_card'],
            wraplength=450,
            justify=tk.CENTER,
            height=4
        )
        self.fake_example.pack(padx=15, pady=(0, 15))
        
    def _create_input_section(self, parent):
        """Create headline input section."""
        input_container = tk.Frame(parent, bg=self.colors['bg_card'], relief=tk.RAISED, bd=2)
        input_container.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(
            input_container,
            text="🔍 TEST A HEADLINE",
            font=("Arial", 18, "bold"),
            fg=self.colors['text'],
            bg=self.colors['bg_card']
        ).pack(pady=15)
        
        # Text input
        input_frame = tk.Frame(input_container, bg=self.colors['bg_card'])
        input_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        self.headline_input = scrolledtext.ScrolledText(
            input_frame,
            height=3,
            font=("Arial", 14),
            bg=self.colors['bg_dark'],
            fg=self.colors['text'],
            insertbackground=self.colors['accent'],
            relief=tk.FLAT,
            wrap=tk.WORD
        )
        self.headline_input.pack(fill=tk.X)
        self.headline_input.insert(1.0, "Enter a news headline to analyze...")
        self.headline_input.bind("<FocusIn>", self._clear_placeholder)
        
        # Analyze button
        self.analyze_btn = tk.Button(
            input_container,
            text="🚀 ANALYZE HEADLINE",
            font=("Arial", 16, "bold"),
            bg=self.colors['accent'],
            fg=self.colors['bg_dark'],
            activebackground=self.colors['accent_glow'],
            relief=tk.FLAT,
            cursor="hand2",
            command=self._analyze_headline,
            height=2
        )
        self.analyze_btn.pack(pady=(0, 20), padx=20, fill=tk.X)
        
    def _create_analysis_section(self, parent):
        """Create results display section."""
        analysis_container = tk.Frame(parent, bg=self.colors['bg_card'], relief=tk.RAISED, bd=2)
        analysis_container.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        tk.Label(
            analysis_container,
            text="📈 ANALYSIS RESULTS",
            font=("Arial", 18, "bold"),
            fg=self.colors['text'],
            bg=self.colors['bg_card']
        ).pack(pady=15)
        
        # Result display
        self.result_frame = tk.Frame(analysis_container, bg=self.colors['bg_dark'])
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Verdict label
        self.verdict_label = tk.Label(
            self.result_frame,
            text="Waiting for analysis...",
            font=("Arial", 28, "bold"),
            fg=self.colors['text_dim'],
            bg=self.colors['bg_dark'],
            height=2
        )
        self.verdict_label.pack(pady=15)
        
        # Confidence meter
        confidence_frame = tk.Frame(self.result_frame, bg=self.colors['bg_dark'])
        confidence_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(
            confidence_frame,
            text="Confidence Level:",
            font=("Arial", 14),
            fg=self.colors['text'],
            bg=self.colors['bg_dark']
        ).pack()
        
        self.confidence_bar = ttk.Progressbar(
            confidence_frame,
            length=800,
            mode='determinate',
            style="Confidence.Horizontal.TProgressbar"
        )
        self.confidence_bar.pack(pady=10)
        
        self.confidence_label = tk.Label(
            confidence_frame,
            text="0%",
            font=("Arial", 20, "bold"),
            fg=self.colors['text'],
            bg=self.colors['bg_dark']
        )
        self.confidence_label.pack(pady=5)
        
        # Indicators
        indicators_frame = tk.Frame(self.result_frame, bg=self.colors['bg_dark'])
        indicators_frame.pack(fill=tk.X, pady=15)
        
        self.indicator_labels = []
        indicators = [
            ("Language Patterns", "pattern"),
            ("Emotional Language", "emotion"),
            ("Credibility Markers", "credibility")
        ]
        
        for indicator, key in indicators:
            ind_frame = tk.Frame(indicators_frame, bg=self.colors['bg_card'], relief=tk.RAISED, bd=1)
            ind_frame.pack(fill=tk.X, pady=3, padx=20)
            
            tk.Label(
                ind_frame,
                text=f"• {indicator}:",
                font=("Arial", 11),
                fg=self.colors['text'],
                bg=self.colors['bg_card'],
                anchor=tk.W
            ).pack(side=tk.LEFT, padx=15, pady=8)
            
            value_label = tk.Label(
                ind_frame,
                text="—",
                font=("Arial", 11, "bold"),
                fg=self.colors['text_dim'],
                bg=self.colors['bg_card']
            )
            value_label.pack(side=tk.RIGHT, padx=15, pady=8)
            self.indicator_labels.append(value_label)
            
    def _create_quick_demo_buttons(self, parent):
        """Create quick demo buttons."""
        demo_frame = tk.Frame(parent, bg=self.colors['bg_dark'])
        demo_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(
            demo_frame,
            text="⚡ QUICK DEMO",
            font=("Arial", 14, "bold"),
            fg=self.colors['text'],
            bg=self.colors['bg_dark']
        ).pack(pady=(0, 10))
        
        btn_frame = tk.Frame(demo_frame, bg=self.colors['bg_dark'])
        btn_frame.pack()
        
        tk.Button(
            btn_frame,
            text="✅ Test Real Headline",
            font=("Arial", 12),
            bg=self.colors['real'],
            fg=self.colors['bg_dark'],
            relief=tk.FLAT,
            cursor="hand2",
            command=lambda: self._load_sample('real'),
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            btn_frame,
            text="⚠️ Test Fake Headline",
            font=("Arial", 12),
            bg=self.colors['fake'],
            fg='white',
            relief=tk.FLAT,
            cursor="hand2",
            command=lambda: self._load_sample('fake'),
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            btn_frame,
            text="🔄 Rotate Examples",
            font=("Arial", 12),
            bg=self.colors['warning'],
            fg=self.colors['bg_dark'],
            relief=tk.FLAT,
            cursor="hand2",
            command=self._rotate_examples,
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
    def _create_footer(self, parent):
        """Create footer with info."""
        footer = tk.Frame(parent, bg=self.colors['bg_dark'])
        footer.pack(fill=tk.X)
        
        tk.Label(
            footer,
            text="💡 Demo for Instagram Reels | Built with TensorFlow & Keras",
            font=("Arial", 10),
            fg=self.colors['text_dim'],
            bg=self.colors['bg_dark']
        ).pack()
        
    def _clear_placeholder(self, event):
        """Clear placeholder text on focus."""
        current = self.headline_input.get(1.0, tk.END).strip()
        if current == "Enter a news headline to analyze...":
            self.headline_input.delete(1.0, tk.END)
            
    def _load_model_async(self):
        """Load model in background thread."""
        def load():
            if MODEL_AVAILABLE:
                try:
                    import numpy as np
                    from tensorflow import keras
                    
                    # Check if model file exists
                    model_path = Path("best_model.keras")
                    if model_path.exists():
                        self.preprocessor = TextPreprocessor(ModelConfig())
                        self.model = FakeNewsModel(ModelConfig())
                        self.model.model = keras.models.load_model(str(model_path))
                        
                        self.root.after(0, lambda: self.status_label.config(
                            text="✅ Model Ready",
                            fg=self.colors['real']
                        ))
                    else:
                        self.root.after(0, lambda: self.status_label.config(
                            text="⚠️ Model file not found - Using demo mode",
                            fg=self.colors['warning']
                        ))
                except Exception as e:
                    self.root.after(0, lambda: self.status_label.config(
                        text=f"⚠️ Demo Mode (Model error)",
                        fg=self.colors['warning']
                    ))
            else:
                self.root.after(0, lambda: self.status_label.config(
                    text="⚠️ Demo Mode (Dependencies not available)",
                    fg=self.colors['warning']
                ))
                
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
        
    def _analyze_headline(self):
        """Analyze the entered headline."""
        if self.is_analyzing:
            return
            
        headline = self.headline_input.get(1.0, tk.END).strip()
        if not headline or headline == "Enter a news headline to analyze...":
            return
            
        self.is_analyzing = True
        self.analyze_btn.config(state=tk.DISABLED, text="⏳ Analyzing...")
        
        # Run analysis in thread
        thread = threading.Thread(target=self._run_analysis, args=(headline,), daemon=True)
        thread.start()
        
    def _run_analysis(self, headline):
        """Run the actual analysis with animation."""
        # Simulate analysis delay for dramatic effect
        for i in range(10):
            progress = (i + 1) * 10
            self.root.after(0, lambda p=progress: self.confidence_bar.config(value=p))
            time.sleep(0.1)
            
        # Get prediction
        if self.model and self.preprocessor:
            try:
                import numpy as np
                # Preprocess
                processed = self.preprocessor.preprocess_texts([headline])
                # Predict
                prediction, confidence = self.model.predict(processed)
                is_fake = prediction[0] == 1
                conf_value = float(confidence[0])
            except Exception as e:
                print(f"Prediction error: {e}")
                is_fake, conf_value = self._demo_prediction(headline)
        else:
            is_fake, conf_value = self._demo_prediction(headline)
            
        # Update UI
        self.root.after(0, lambda: self._display_results(is_fake, conf_value, headline))
        
    def _demo_prediction(self, headline):
        """Simple rule-based prediction for demo mode."""
        headline_lower = headline.lower()
        
        # Keywords that indicate fake news
        fake_indicators = [
            'shocking', 'unbelievable', 'you won\'t believe',
            'this one trick', 'doctors hate', 'they don\'t want',
            'exposed', 'leaked', 'secret', 'miracle', 'breaking',
            'viral', 'scandal', 'won\'t believe', 'must see',
            'click here', 'shocking truth', 'revealed'
        ]
        
        # Count fake indicators
        fake_score = sum(1 for indicator in fake_indicators if indicator in headline_lower)
        
        # All caps words (common in clickbait)
        words = headline.split()
        caps_ratio = sum(1 for word in words if word.isupper() and len(word) > 2) / max(len(words), 1)
        
        # Excessive punctuation
        exclamation_count = headline.count('!')
        
        # Calculate confidence
        total_score = fake_score * 15 + caps_ratio * 30 + exclamation_count * 10
        confidence = min(max(total_score, 20), 95) / 100
        
        is_fake = total_score > 15
        
        return is_fake, confidence if is_fake else 1 - confidence
        
    def _display_results(self, is_fake, confidence, headline):
        """Display analysis results with animation."""
        # Set verdict
        if is_fake:
            verdict = "⚠️ SUSPECT"
            color = self.colors['fake']
            emoji = "🚨"
        else:
            verdict = "✅ TRUSTWORTHY"
            color = self.colors['real']
            emoji = "✓"
            
        self.verdict_label.config(text=verdict, fg=color)
        
        # Update confidence
        conf_percent = int(confidence * 100)
        self.confidence_bar.config(value=conf_percent)
        self.confidence_label.config(text=f"{conf_percent}%", fg=color)
        
        # Update indicators with random but consistent values
        indicators = [
            ("Normal" if not is_fake else "Unusual", color),
            ("Moderate" if not is_fake else "High", color),
            (emoji + " Good" if not is_fake else "⚠ Low", color)
        ]
        
        for label, (text, col) in zip(self.indicator_labels, indicators):
            label.config(text=text, fg=col)
            
        # Re-enable button
        self.analyze_btn.config(state=tk.NORMAL, text="🚀 ANALYZE HEADLINE")
        self.is_analyzing = False
        
    def _load_sample(self, sample_type):
        """Load a sample headline."""
        headline = random.choice(self.sample_headlines[sample_type])
        self.headline_input.delete(1.0, tk.END)
        self.headline_input.insert(1.0, headline)
        
    def _rotate_examples(self):
        """Rotate the split screen examples."""
        real = random.choice(self.sample_headlines['real'])
        fake = random.choice(self.sample_headlines['fake'])
        self.real_example.config(text=real)
        self.fake_example.config(text=fake)
        
    def run(self):
        """Start the UI."""
        # Configure progress bar style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(
            "Confidence.Horizontal.TProgressbar",
            troughcolor=self.colors['bg_dark'],
            bordercolor=self.colors['bg_dark'],
            background=self.colors['accent'],
            lightcolor=self.colors['accent_glow'],
            darkcolor=self.colors['accent']
        )
        
        self.root.mainloop()


def main():
    """Launch the demo UI."""
    print("🚀 Launching Instagram Reels Demo UI...")
    print("📱 Window size: 1080x1920 (9:16 ratio)")
    print("🎬 Perfect for screen recording vertical video!")
    print("\n" + "="*60)
    
    app = FakeNewsDetectorUI()
    app.run()


if __name__ == "__main__":
    main()
