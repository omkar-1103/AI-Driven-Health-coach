# 🩺 AI Health Coach

AI Health Coach is a full-stack, AI-powered health monitoring and predictive analytics dashboard. It seamlessly combines live wearable sensor simulation, traditional Machine Learning for disease prediction, and an advanced Retrieval-Augmented Generation (RAG) chatbot for personalized medical guidance.

## 🚀 Key Features

*   **⌚ Live Wearable Simulation:** Simulates real-time physiological data (Heart Rate, Blood Oxygen, Blood Pressure, Temperature, Step Count) using a realistic random-walk algorithm.
*   **⚠️ Health Risk Prediction:** Uses a Random Forest ML model to analyze live wearable data and predict general health risks on the fly.
*   **🩸 Diabetes Prediction:** Evaluates user-inputted clinical metrics (Glucose, BMI, Insulin, Diabetes Pedigree Function) through a Logistic Regression model to assess diabetes risk.
*   **🤖 AI Health Assistant (RAG):** Features a specialized AI chatbot powered by Google Gemini and LangChain. 
    *   **Vector Search:** Answers are grounded in actual medical guidelines using a local FAISS vector database.
    *   **Context Injection:** The chatbot automatically reads the user's real-time wearable vitals and recent ML predictions to provide highly personalized health assessments.
    *   **Guardrails:** Strict safety prompts prevent the AI from answering non-medical or dangerous questions.
*   **🔐 Secure Authentication:** Full user registration, login, and JWT-style token management powered by SQLite and `bcrypt`.
*   **✨ Glassmorphic UI:** A stunning, responsive, dark-mode frontend built with pure CSS and Vanilla JavaScript.

## 🛠️ Technology Stack

*   **Backend:** Python, Flask
*   **Database:** SQLite (`health.db`)
*   **Machine Learning:** Scikit-Learn, Pandas, Numpy, Joblib
*   **AI / RAG:** LangChain, FAISS, HuggingFace (`all-MiniLM-L6-v2`), Google Generative AI (Gemini 2.5 Flash)
*   **Frontend:** HTML5, Vanilla CSS3 (Glassmorphism), Vanilla JavaScript
