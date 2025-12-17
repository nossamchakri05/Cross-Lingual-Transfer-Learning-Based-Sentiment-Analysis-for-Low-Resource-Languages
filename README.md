# Cross-Lingual Transfer Learning-Based Sentiment Analysis for Low-Resource Languages

## 📋 Overview

This project implements a sophisticated **cross-lingual transfer learning approach** for sentiment analysis that leverages high-resource languages (English, Spanish, French, Hindi, German, Arabic) to perform sentiment analysis on low-resource languages including Bengali, Odia, Afrikaans, Malay, and Urdu.

The system uses **language similarity graphs** to intelligently select the most linguistically similar high-resource language for each low-resource language input, ensuring optimal transfer learning performance.

## ✨ Key Features

- **Multi-Language Support**: Analyzes sentiment for 11+ languages across multiple scripts
- **Intelligent Language Matching**: Uses NetworkX-based language similarity graphs to find optimal high-resource language matches
- **Automatic Language Detection**: FastText-based language identification
- **Multi-Model Architecture**: Specialized BERT-based sentiment models for different languages
- **Web Interface**: Flask-based application with user authentication and sentiment logging
- **Translation Pipeline**: Automatic translation to matched high-resource languages using Google Translate
- **Database Logging**: Tracks all sentiment analysis results with timestamps and user information

## 🗣️ Supported Languages

### High-Resource Languages (Primary Models)
- **English** (en)
- **Spanish** (es)
- **French** (fr)
- **Hindi** (hi)
- **German** (de)
- **Arabic** (ar)

### Low-Resource Languages (Transfer Learning)
- **Bengali** (bn)
- **Odia** (or)
- **Afrikaans** (af)
- **Malay** (ms)
- **Urdu** (ur)

## 🏗️ Architecture

```
Low-Resource Input (e.g., Odia)
          ↓
    [Language Detection] (FastText)
          ↓
    [Language Similarity Graph]
          ↓
    [Find Optimal High-Resource Match]
          ↓
    [Automatic Translation]
          ↓
    [Language-Specific BERT Model]
          ↓
    [Sentiment Classification] (Positive/Negative/Neutral)
          ↓
    [Database Logging & Result Display]
```

## 📊 Sentiment Classification

- **Binary Classification**: Positive/Negative (for most models)
- **Ternary Classification**: Positive/Neutral/Negative (Spanish and some extended models)
- **Confidence Scores**: Softmax probability outputs for each prediction

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration, optional)
- MySQL Server
- 8GB+ RAM recommended

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/nossamchakri05/Cross-Lingual-Transfer-Learning-Based-Sentiment-Analysis-for-Low-Resource-Languages.git
   cd Cross-Lingual-Transfer-Learning-Based-Sentiment-Analysis-for-Low-Resource-Languages
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download pre-trained models**
   Ensure the following directories are present:
   - `sentiment_model/` (English BERT)
   - `french_model/` (French BERT)
   - `hindi_model/` (Hindi BERT)
   - `spanish_sentiment_model/` (Spanish BERT)
   - `afrikaans_finetuned_model/` (Afrikaans BERT)
   - `malay_finetuned_model/` (Malay BERT)

4. **Configure Database**
   - Update `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` in `app.py`
   - Create the required MySQL database and tables (schema provided below)

5. **Configure FastText Model**
   - Ensure `lid.176.bin` is in the project root (language identification model)
   - Download from: https://dl.fbaipublicfiles.com/fasttext/supervised-models/

6. **Load Language Similarity Graph**
   - Ensure `language_similarity_graph.gml` is in the project root
   - Generated using `graphconstruction.ipynb`

### Running the Application

```bash
python app.py
```

The application will start at `http://localhost:5000`

## 📁 Project Structure

```
.
├── app.py                              # Main Flask application
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
│
├── Notebooks/
│   ├── nlpproj.ipynb                  # Main project notebook with language detection
│   ├── model_codes.ipynb              # Model training and evaluation code
│   └── graphconstruction.ipynb        # Language similarity graph construction
│
├── Models/
│   ├── sentiment_model/               # English sentiment model (BERT)
│   ├── french_model/                  # French sentiment model
│   ├── hindi_model/                   # Hindi sentiment model
│   ├── spanish_sentiment_model/       # Spanish sentiment model (3-class)
│   ├── afrikaans_finetuned_model/    # Afrikaans finetuned model
│   └── malay_finetuned_model/        # Malay finetuned model
│
├── Datasets/
│   ├── afrikaans_sentiment_75.csv     # Afrikaans labeled data
│   ├── malay_sentiment_75.csv         # Malay labeled data
│   ├── odia_sentiment_100.csv         # Odia labeled data
│   ├── urdu_sentiment_100.csv         # Urdu labeled data
│   ├── IMDB Dataset.csv               # IMDB sentiment data
│   └── news_sentiment_annotated_data.tsv  # News domain sentiment data
│
├── Corpora/
│   ├── afrikaans.txt                  # Afrikaans language corpus
│   ├── malay.txt                      # Malay language corpus
│   ├── odia.txt                       # Odia language corpus
│   ├── english.txt                    # English language corpus
│   ├── hindi.txt                      # Hindi language corpus
│   └── [other_language].txt           # Additional language corpora
│
├── Language Resources/
│   ├── language_similarity_graph.gml  # NetworkX language similarity graph
│   ├── lid.176.bin                    # FastText language identification model
│   └── test sample.txt                # Sample test inputs
│
└── Templates/
    ├── reg.html                       # Registration page
    ├── registration.html              # Registration form
    ├── index.html                     # Language selection
    ├── input_page.html                # Text input interface
    ├── result_page.html               # Results display
    └── admin_portal.html              # Admin dashboard
```

## 🔄 Workflow

### User Registration & Login
1. User registers with email and password
2. Credentials stored securely with bcrypt hashing
3. User logs in to access sentiment analysis features

### Sentiment Analysis Pipeline
1. **Language Input**: User selects target low-resource language
2. **Text Input**: User provides text via direct input or file upload
3. **Language Detection**: FastText identifies detected language
4. **Validation**: Confirms input matches selected language
5. **Transfer Learning**: 
   - Finds optimal high-resource language using similarity graph
   - Translates text to high-resource language
   - Applies corresponding BERT model
6. **Classification**: Performs sentiment analysis and generates score
7. **Logging**: Stores results in database for admin review

### Admin Portal
- View all sentiment analysis history
- Filter results by user, timestamp, or language
- Monitor model performance and user activity

## 🧠 Model Details

### Base Models
All models are based on **BERT** (Bidirectional Encoder Representations from Transformers):
- `bert-base-multilingual-cased`: Initial backbone
- Fine-tuned on language-specific and domain-specific datasets

### Training Details
- **Batch Size**: 16
- **Learning Rate**: 2e-5 (AdamW optimizer)
- **Max Sequence Length**: 128 tokens
- **Epochs**: Variable based on dataset size
- **Train-Test Split**: 80-20

### Performance Metrics
- Accuracy
- Precision, Recall, F1-Score (macro-averaged for multiclass)
- Weighted metrics for imbalanced datasets

## 📊 Supported Sentiment Labels

| Language | Labels | Type |
|----------|--------|------|
| English | Positive, Negative | Binary |
| French | Positive, Negative | Binary |
| Hindi | Positive, Negative | Binary |
| Spanish | Positive, Neutral, Negative | Ternary |
| Afrikaans | Positive, Negative | Binary |
| Malay | Positive, Negative | Binary |
| Urdu | Positive, Negative | Binary |

## 🗄️ Database Schema

### Users Table
```sql
CREATE TABLE users (
    user_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    phone_number VARCHAR(20),
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Sentiment Logs Table
```sql
CREATE TABLE sentiment_logs (
    log_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    input_sentence TEXT NOT NULL,
    translated_sentence TEXT,
    sentiment VARCHAR(50),
    sentiment_score FLOAT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
```

## 🔧 API Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Home/Login page |
| `/registration` | GET, POST | User registration |
| `/login` | POST | User login |
| `/language` | GET, POST | Language selection |
| `/input` | GET, POST | Text input & processing |
| `/english` | GET | English model prediction |
| `/french` | GET | French model prediction |
| `/spanish` | GET | Spanish model prediction |
| `/hindi` | GET | Hindi model prediction |
| `/admin` | GET | Admin sentiment logs dashboard |

## 🎯 Key Technologies

- **Deep Learning**: PyTorch, Hugging Face Transformers
- **Language Processing**: FastText, NetworkX, Multilingual BERT
- **Translation**: Google Translate API (deep-translator)
- **Web Framework**: Flask, Jinja2
- **Database**: MySQL, PyMySQL
- **Data Science**: Pandas, NumPy, Scikit-learn
- **Utilities**: Werkzeug (security), TorchVision

## 💻 Hardware Requirements

- **Minimum**: 4GB RAM, CPU-only inference
- **Recommended**: 8GB+ RAM, NVIDIA GPU with CUDA support
- **Optimal**: 16GB+ RAM, RTX 3060 or better for faster inference

## 📈 Performance Notes

- Average inference time: 0.5-2 seconds per prediction
- Batch processing supported for multiple texts
- GPU acceleration: 3-5x faster inference on CUDA-enabled devices

## 🔐 Security Considerations

- Passwords hashed using Werkzeug's `generate_password_hash()`
- SQL injection prevention via parameterized queries
- Session-based authentication
- Environment variables for sensitive credentials (recommended)

## 📝 Usage Examples

### Direct Text Input
1. Log in to the application
2. Select target language (e.g., Odia)
3. Enter text in Odia script
4. System automatically detects language, translates, and analyzes
5. View sentiment result (Positive/Negative with confidence score)

### File Upload
1. Prepare a `.txt` file with one sentence per line
2. Upload through the interface
3. System processes each line sequentially
4. Results displayed for each sentence

### Admin Dashboard
- View all sentiment analysis history
- Monitor user activity and trends
- Export results for further analysis

## 🚧 Future Enhancements

- [ ] Real-time model performance metrics dashboard
- [ ] Support for more low-resource languages
- [ ] Fine-tuning UI for custom models
- [ ] Multi-GPU support for faster inference
- [ ] REST API endpoints
- [ ] Export results to CSV/JSON
- [ ] Sentiment trend analysis visualizations
- [ ] Aspect-based sentiment analysis
- [ ] Emotion classification (anger, joy, sadness, etc.)

## 📚 Research Background

This project implements concepts from:
- Cross-lingual transfer learning in NLP
- Language similarity metrics and graph-based language relationships
- Multilingual BERT and its applications
- Domain adaptation for sentiment analysis
- Low-resource language processing

**Key Publications**:
- Devlin et al., 2018 - BERT: Pre-training of Deep Bidirectional Transformers
- Pires et al., 2019 - How Multilingual is Multilingual BERT?
- Ruder et al., 2017 - Cross-lingual Transfer Learning

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Authors

- **Nossamchakri** - Initial project concept and implementation

## 🙏 Acknowledgments

- Hugging Face for Transformers library and pre-trained models
- Facebook for FastText language identification
- The NLP research community for multilingual models
- Contributors and maintainers of PyTorch and related libraries

## 📧 Contact & Support

For questions, issues, or suggestions:
- GitHub Issues: [Project Issues](https://github.com/nossamchakri05/Cross-Lingual-Transfer-Learning-Based-Sentiment-Analysis-for-Low-Resource-Languages/issues)
- Email: nossamchakri05@gmail.com

## 📊 Citation

If you use this project in your research, please cite:

```bibtex
@software{cross_lingual_sentiment_2024,
  title={Cross-Lingual Transfer Learning-Based Sentiment Analysis for Low-Resource Languages},
  author={Nossamchakri},
  year={2024},
  url={https://github.com/nossamchakri05/Cross-Lingual-Transfer-Learning-Based-Sentiment-Analysis-for-Low-Resource-Languages}
}
```

## 📋 Changelog

### Version 1.0.0 (Initial Release)
- Multi-language sentiment analysis support
- Language similarity graph-based model selection
- Web interface with user authentication
- Database logging of all predictions
- Admin dashboard for result tracking
- Support for 11+ languages

---

**Last Updated**: December 2024
**Repository**: https://github.com/nossamchakri05/Cross-Lingual-Transfer-Learning-Based-Sentiment-Analysis-for-Low-Resource-Languages
