# Cyber-Security-AI-file-scanner
Certainly! Here's the corrected structure for the README file content:

---

# DariusAI - Cybersecurity File Scanner

DariusAI is a Cybersecurity File Scanner implemented in Python, designed to assist users in scanning various file formats for potential malware threats. It offers a user-friendly interface allowing users to interact via text input, voice commands, or file scanning.

## Features

- **File Type Support**: DariusAI supports scanning of multiple file types including PDF, TXT, DOC, and PPT.
- **Malware Detection**: Utilizes machine learning techniques to detect and classify potential malware threats within scanned files.
- **Multiple Input Modes**: Users can interact with DariusAI via text input, voice commands, or by scanning files directly.
- **Real-time Feedback**: Provides real-time feedback on the scanning process and the results of the malware detection.
- **Training Mode**: Allows users to train the underlying machine learning model with custom datasets.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/DariusAI.git
   ```

2. Navigate to the project directory:
   ```
   cd DariusAI
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python darius_ai.py
   ```

2. Interaction Modes:
   - **Text Input**: Type your query or command in the provided text entry box and press Enter.
   - **Voice Input**: Click the "Speak" button to activate voice input mode. Speak your query and wait for the response.
   - **File Scanning**: Click the "Scan File" button to browse and select a file for scanning.

3. Commands:
   - **clear**: Clears the conversation history.
   - **exit**: Exits the application.

## Training a Custom Model

If you wish to train a custom machine learning model:

1. Modify the `train_model` method in `darius_ai.py` to load your dataset and train the model.
2. Run the application and wait for the new model to be trained and saved.

## Dependencies

- `customtkinter==0.1.0`: Customized version of the tkinter library for GUI.
- `python-magic==0.4.24`: Python wrapper for the libmagic file type identification library.
- `scikit-learn==0.24.2`: Machine learning library for training and using classification models.
- `SpeechRecognition==3.8.1`: Library for performing speech recognition.
- `pdfplumber==0.5.28`: Library for extracting text from PDF files.
- `python-pptx==0.6.19`: Library for working with PowerPoint files.

## Contributing

Contributions to DariusAI are welcome! If you'd like to contribute, please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries or feedback, please contact [Your Name] at [your.email@example.com].

---

This structure provides a clear and concise overview of the DariusAI project, its features, installation instructions, usage guidelines, and other relevant details. Feel free to customize it further based on your preferences and project requirements.
