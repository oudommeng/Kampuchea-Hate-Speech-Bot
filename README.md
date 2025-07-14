# Kampuchea Hate Speech Bot 🇰🇭🤖

This project is a Telegram bot designed to detect and filter Khmer hate speech using Natural Language Processing (NLP). Developed as a research project by students at the **Cambodia Academy of Digital Technology (CADT)**, this bot helps maintain a healthier online community.

This is an open-source project, and we welcome collaboration and contributions.

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org/)
[![Telegram](https://img.shields.io/badge/Telegram-Bot-blue?logo=telegram)](https://telegram.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

* **Real-time Hate Speech Detection**: Monitors messages in Telegram groups and identifies harmful content in the Khmer language.
* **NLP-Powered Filtering**: Uses a machine learning model (`scikit-learn`) to accurately classify text.
* **Automatic Message Deletion**: Instantly removes messages that are classified as hate speech to keep chats clean.
* **Easy to Deploy**: Can be run locally or deployed to a server.

## 🛠️ Technology Stack

* **Backend**: Python
* **Telegram API Wrapper**: `python-telegram-bot`
* **Machine Learning**: `scikit-learn`, `pandas`
* **Natural Language Processing**: Custom model using TF-IDF Vectorization.

## 🚀 Getting Started

Follow these instructions to get a local copy of the bot up and running for development and testing.

### Prerequisites

* [Python 3.9+](https://www.python.org/downloads/)
* A Telegram Bot Token. You can get one from [BotFather](https://t.me/botfather).

### Installation & Local Development

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/oudommeng/Kampuchea-Hate-Speech-Bot.git](https://github.com/oudommeng/Kampuchea-Hate-Speech-Bot.git)
    cd Kampuchea-Hate-Speech-Bot
    ```

2.  **Create and activate a virtual environment:**
    ```sh
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies from `requirements.txt`:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    * Create a file named `.env` in the root directory.
    * Add your Telegram bot token to this file:
        ```env
        TOKEN="YOUR_TELEGRAM_BOT_TOKEN_HERE"
        ```

5.  **Run the application:**
    ```sh
    python3 app.py
    ```

6.  **Add the bot to your Telegram group:**
    * Find your bot on Telegram and add it to a group.
    * Make sure to grant it administrator permissions so it can delete messages.

## ☁️ Deployment

For continuous operation, you can deploy this bot to a cloud service like Heroku, AWS, or Google Cloud.

### GitHub Actions Deployment
The repository includes a placeholder for GitHub Actions deployment (`deploy.yml`). The Telegram bot token should be stored as a repository secret named `TOKEN` and will be automatically configured in the production environment when the workflow is fully set up.

## 🤝 Contributing

Contributions are what make the open-source community an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

This project is distributed under the MIT License. See the `LICENSE` file for more information.

## 🙏 Acknowledgements

* Cambodia Academy of Digital Technology (CADT)
* All Supervisors and Mentors
* The `python-telegram-bot` community
### Project Team

<a href="https://github.com/oudommeng">
  <img src="https://github.com/oudommeng.png?size=100" width="100">
</a>
<a href="https://github.com/HimRonald">
  <img src="https://github.com/HimRonald.png?size=100" width="100">
</a>
<a href="https://github.com/hengtramit">
  <img src="https://github.com/hengtramit.png" width="100">
</a>
<a href="https://github.com/witKen">
  <img src="https://github.com/witKen.png?size=100" width="100">
</a>