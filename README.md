# StockSense - AI-Powered Stock Analysis Platform

StockSense is a production-ready web application that provides layperson-friendly stock insights using AI-powered forecasting, sentiment analysis, and smart money tracking. The system integrates LangChain and LangGraph for orchestrated multi-agent workflows.

## 🚀 Features

### Core Capabilities
- **Probabilistic Forecasting**: AI-powered directional predictions with 95% confidence horizons
- **Sentiment Analysis**: Real-time news and social media sentiment using FinBERT
- **Smart Money Tracking**: Institutional flows, insider trading, and congressional disclosures
- **Layperson Explanations**: Plain English analysis with tier-based detail levels
- **Multi-Source Data**: Swappable adapters for Alpha Vantage, Tiingo, Finnhub, and NewsAPI

### User Tiers
- **Basic**: Core forecasting and sentiment analysis
- **Premium**: Advanced features including smart money insights, alerts, and historical analysis

## 🏗️ Architecture

### System Components
- **Frontend**: Next.js/React with TypeScript and Radix UI
- **Backend**: FastAPI with LangGraph workflow orchestration
- **AI Agents**: Specialized agents for prediction, sentiment, explanation, and smart money
- **Data Layer**: Swappable adapters following OOP principles
- **ML Pipeline**: Probabilistic forecasting with confidence calibration

### Data Flow
```
User Input → API Gateway → LangGraph Workflow → AI Agents → Response Assembly → Frontend
```

## 📋 Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL (for production)
- API Keys for data sources

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd capstone
```

### 2. Backend Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys
```

### 3. Frontend Setup
```bash
cd frontend
npm install
```

### 4. Environment Configuration
**IMPORTANT:** The `.env` file has already been created with your API keys. Never commit this file to git!

If you need to create a new `.env` file:
```bash
cp .env.example .env
# Then edit .env and add your actual API keys
```

**Get your free API keys from:**
- Alpha Vantage: https://www.alphavantage.co/support/#api-key (500 calls/day)
- Tiingo: https://www.tiingo.com/ (500 calls/day)
- Finnhub: https://finnhub.io/register (60 calls/minute)
- NewsAPI: https://newsapi.org/register (100 requests/day)
- OpenAI: https://platform.openai.com/api-keys (required for AI agents)
- LangChain: https://smith.langchain.com/ (optional, for monitoring)

## 🚀 Running the Application

### Development Mode

#### Backend (FastAPI)
```bash
# From root directory
python -m pipelines.realtime.api
# Or with uvicorn directly
uvicorn pipelines.realtime.api:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend (Next.js)
```bash
cd frontend
npm run dev
```

### Production Mode
```bash
# Backend
uvicorn pipelines.realtime.api:app --host 0.0.0.0 --port 8000

# Frontend
cd frontend
npm run build
npm start
```

## 📊 API Endpoints

### Core Analysis
- `POST /analyze` - Main stock analysis endpoint
- `GET /prices` - Historical price data
- `GET /news` - News articles and sentiment
- `GET /sentiment` - Sentiment analysis results
- `GET /smart-money` - Institutional and insider data

### Health and Info
- `GET /` - API information
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation

## 🧪 Testing

### Backend Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_agents.py

# Run with coverage
pytest --cov=pipelines
```

### Frontend Tests
```bash
cd frontend
npm test
```

## 📈 Usage Examples

### Basic Stock Analysis
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "user_tier": "basic"}'
```

### Premium Analysis
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "user_tier": "premium"}'
```

## 🔧 Configuration

### Data Adapters
The system uses swappable data adapters following the Strategy pattern:

```python
from pipelines.realtime.data_adapters import DataAdapterFactory

# Create adapter
adapter = DataAdapterFactory.create_adapter("alpha_vantage", api_key)

# Use in service
from pipelines.realtime.data_adapters import DataService
service = DataService({"alpha_vantage": adapter})
```

### LangGraph Workflow
Customize the workflow by modifying agent prompts:

```python
from pipelines.realtime.agents import PredictionAgent
from pipelines.realtime.langgraph_workflow import create_stocksense_workflow

# Create custom workflow
workflow = create_stocksense_workflow(
    prediction_agent=prediction_agent,
    sentiment_agent=sentiment_agent,
    explanation_agent=explanation_agent,
    smart_money_agent=smart_money_agent,
    confidence_threshold=0.95
)
```

## 📚 Documentation

- [System Architecture](docs/architecture.md)
- [API Documentation](http://localhost:8000/docs)
- [Data Sources](docs/data/)
- [Deployment Guide](docs/deployment.md)

## 🚨 Important Disclaimers

- **Informational Only**: StockSense provides educational information, not investment advice
- **No Guarantees**: Past performance does not guarantee future results
- **Risk Warning**: All investments carry risk of loss
- **Data Accuracy**: While we strive for accuracy, data may contain errors

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the [documentation](docs/)
- Review the [API documentation](http://localhost:8000/docs)

## 🔮 Roadmap

### Phase 1 (Current)
- ✅ Core forecasting and sentiment analysis
- ✅ Basic and premium user tiers
- ✅ Multi-source data integration

### Phase 2 (Next)
- 🔄 Advanced ML models and backtesting
- 🔄 Real-time streaming data
- 🔄 Portfolio analysis features

### Phase 3 (Future)
- 📋 Options data integration
- 📋 Alternative data sources
- 📋 Mobile application
- 📋 Advanced risk metrics

---

**StockSense** - Making stock analysis accessible to everyone through AI-powered insights.














Byeol Ahn - 301288852
Joshua Oliver - 301299194
Pamuditha Gamage - 301231718
Sua Cha - 301271420
Taehyun Oh - 301286174


Software Engineering Technology – AI
COMP385 SEC.001


September 14, 2025




















________________


Introduction
Stock market forecasting is a challenging yet in-demand problem in applied artificial intelligence. Financial markets are dynamic and influenced by a wide range of factors, making prediction complex. Traditional machine learning methods rely mainly on numerical inputs such as historical prices, trading volumes, and technical indicators. While these provide valuable insights, they overlook external elements like financial news, investor sentiment, and global events, which often drive sudden market shifts. In this project, we aim to integrate both numerical indicators and sentiment factors from financial news to develop a model that forecasts and analyzes market behavior for research and educational purposes.


Problem Statement
Many retail investors lack the financial literacy needed to make informed investment choices. A World Economic Forum study found that 40% of non-investors avoid markets because they “don’t know how” or find it confusing, and nearly 70% would invest more with better education. Studies also confirm that financial literacy strongly predicts participation in markets. At the same time, nearly 47% of U.S. adults report using AI chatbots like ChatGPT for financial advice, showing a growing reliance on large language models. Our project addresses this gap by creating a financial stock playground that presents critical financial data from major indices (Nasdaq, Dow Jones, and S&P 500) in a human-readable form and forecasts short-term (≤2-week) loss risk for educational purposes.


Methodology
First we will search for research on using artificial intelligence to conduct stock analysis as well as research other financial investment applications (with and without Agentic AI). From our research we will determine an approach to predicting the short term valuation of stocks as well as the data needed for our model. The research will determine the structure of our multi-agent model and how we divide the prediction subtasks between them. Once we have defined an overall structure for the model, we will identify our data sources and conduct an exploratory data analysis to identify any data-cleaning or pre-processing steps we need to add. Using the approach we identified in the research phase, we will build an AI model that will predict the short-term valuation of a stock in the S&P 500 index. We will then test this model to ensure that its predictions are accurate, explainable, and generalizes well. We will test our model with both statistical measures like the RMSE and profit-based measures, for instance by comparing the rate of return from using the model’s predictions versus passively investing in the S&P 500 index. Additionally, we will integrate our model into a web application where users can search for a stock and see whether the price is likely to go up or down in the short term and how volatile the stock is. Initially we will go through this process to create a minimum viable product (MVP) with only the most critical functional requirements implemented. Once we have tested the MVP with non-technical users we will move to adding improvements and making changes based on the feedback we have received. The second iteration will yield an improved version of our application. Finally, we will test the application again and correct any remaining issues or bugs. This will produce the final, polished version of our product.


Hypothesis
By combining numerical indicators with financial news sentiment, our model will predict short-term (≤2-week) stock movements with higher than baseline accuracy. Presenting results in clear, human-readable text will make the output easier to understand, and at least 70% of novice investors will feel confident about investing choices when given better education.


Work Cited
Khan, M. S. R., Rabbani, N., & Kadoya, Y. (2020). Is financial literacy associated with investment in financial markets in the United States? Sustainability, 12(18), 7370. https://doi.org/10.3390/su12187370
World Economic Forum. (2022, August 31). New study finds financial education gaps are primary barrier to retail investing in capital markets. https://www.weforum.org/press/2022/08/new-study-finds-financial-education-gaps-are-primary-barrier-to-retail-investing-in-capital-markets/
Manning-Schaffel, V. (2024, April 4). Don’t ask ChatGPT for financial advice, study warns. Entrepreneur. https://www.entrepreneur.com/business-news/dont-ask-chatgpt-for-financial-advice-study/489792






________________


Attachments
1. AI Capability Description
1.1. Overview
Our project’s AI capability is built on LangGraph and CrewAI. LangGraph provides workflow orchestration for the entire prediction pipeline, while CrewAI enables role-based collaboration among specialized agents (Prediction, Sentiment, and Explanation). Together, these frameworks support the creation of a compound AI system that integrates stock market data and financial news sentiment to generate short-term forecasts (≤2 weeks) with explainability and educational value.


1.2. AI Capability Patterns
We combined three patterns out of seven patterns of AI.
1.2.1. Autonomous Systems Pattern
* LangGraph automatically routes requests and coordinates the execution of Prediction and Sentiment Agents.
* CrewAI ensures agents collaborate with minimal human intervention, producing reliable forecasts.
1.2.2. Conversational Pattern
* Users interact through natural language queries.
* LLM integration enables clear, human-readable responses, tailored for both Basic and Premium users.
1.2.3. Predictive Analytics & Decision Support
* The Prediction Agent estimates stock movement probabilities (Up/Down/Neutral) within a ≤2-week horizon.
* The Sentiment Agent aggregates financial news and social media to provide a positive/neutral/negative score.
* The Explanation Agent delivers plain-language reasoning, with Premium users receiving up to three contributing factors.
* Predictions use a 95% confidence threshold: if reached, a clear rating is given; otherwise, “trend-only” guidance is displayed.


1.3. Compound AI System Design
1.3.1. System Components
* LLM (e.g., GPT-4o or fine-tuned FinGPT): Natural language understanding and explanation.
* LangGraph: Orchestrates agent workflows.
* CrewAI: Manages agent collaboration across roles.
* Agents:
   * Prediction Agent: Runs statistical/ML models (evaluated with RMSE and rate-of-return vs passive S&P 500).
   * Sentiment Agent: Processes financial news and filings for sentiment classification.
   * Explanation Agent: Produces human-readable insights, with constraints for Premium user output.
* External Data Sources: OHLCV stock data, fundamentals, earnings events, news APIs, social data, institutional filings.
* Inference Module: Integrates outputs and delivers user-facing dashboards.
1.3.2. Workflow
1. User enters a stock ticker or watchlist.
2. LangGraph retrieves data and assigns subtasks to CrewAI agents.
3. Prediction Agent computes probabilities (Up/Down/Neutral).
4. Sentiment Agent produces aggregated sentiment score.
5. Confidence Gate applies 95% threshold.
6. Explanation Agent translates results into user-friendly output.
7. Results displayed with differences for Basic vs Premium users.


1.4. Validation & Tracking Practices
1.4.1. Validation
* Evaluate predictions with RMSE for accuracy.
* Compare portfolio returns against a passive S&P 500 baseline.
* Measure user trust: at least 70% of novice investors should report confidence when guided by explanations.
1.4.2. Tracking:
* Maintain model history and prediction logs for Premium users.
* Version control for workflows and agents via LangGraph and CrewAI logs.
1.4.3. Deployment Plan:
* Cloud-based deployment (AWS/GCP) with modular orchestration.
* Continuous monitoring for data freshness (≤24h for Free users, ≤1 min for Premium).


2. Stakeholder Register
Stakeholder Name
	Stakeholder
Position (Role)
	External/Internal
	Stakeholder Contact Details
	Operational/ Executive
	Interest (high, medium, low)
	Ethan Anderson
	Customer
	External
	ethan.anderson.ai@gmail.com
	Operational
	High
	Mason Taylor
	App Admin
	External
	mason.taylor.research@outlook.com
	Operational
	High
	Mason White
	AI Developer
	Internal
	mason.white.investor@gmail.com
	Operational
	High
	Liam Brown
	Programmer
	Internal
	liam.brown.edu@outlook.com
	Operational
	High
	Emma Thompson
	Financial Data Provider
	Internal
	emma.thompson.data@gmail.com
	Operational
	Medium
	Olivia Taylor
	Investor
	External
	olivia.taylor.academic@outlook.com
	Executive
	High
	Lucas Kim
	IT Security Officer
	Internal
	lucas.kim.public@gmail.com
	Executive
	Medium
	



3. Functional Requirements
Our system defines four user types: Anonymous User, Registered User, Premium User, and Administrator. Each higher-level role inherits all the functionality of lower-level roles, with additional capabilities as described below.
3.1. Anonymous User
* The system shall allow an anonymous user to search for any stock ticker and view its prediction page without restrictions on the number of searches.
* The system shall not allow an anonymous user to save search history, compare multiple stocks simultaneously, or manage favorites.
3.2. Registered User
* The system shall allow a registered user to perform all functions of an anonymous user.
* The system shall allow a registered user to view their personal search history of stock predictions.
* The system shall allow a registered user to add or remove favorite stock tickers and view them on a personalized dashboard.
* The system shall allow all users to view at least one highlighted evidence sentence and a confidence score (0–100%) for each prediction.
3.3. Premium User 
* The system shall allow a premium user to perform all functions of a registered user.
* The system shall allow a premium user to interact with an AI assistant for conversational and personalized stock analysis.
* The system shall allow a premium user to evaluate an entire watchlist or portfolio and view combined prediction results.
* The system shall allow a premium user to receive alerts when prediction outcomes change or when a probability threshold is met (e.g., probability of decline ≥70%).
* The system shall allow a premium user to view historical predictions and accuracy metrics compared to actual market performance.
* The system shall allow a premium user to export results to a CSV file (up to 5,000 rows within 5 seconds).
* The system shall allow premium users to view plain-language summaries of up to three key contributing factors for each prediction.
3.4. Administrator 
* The system shall allow an administrator to manage user accounts and subscription plans.
* The system shall allow an administrator to update AI models and data sources.
* The system shall allow an administrator to monitor system performance and usage statistics.
Access Control Matrix


Function
	Anonymous
	Registered
	Premium
	Admin
	Search stock ticker (unlimited)
	Yes
	Yes
	Yes
	Yes
	View limited news sentiment
	Yes
	Yes
	Yes
	Yes
	View search history
	No
	Yes
	Yes
	No
	Manage favorites
	No
	Yes
	Yes
	No
	Compare multiple stocks
	No
	Yes
	Yes
	No
	Conversational AI analysis
	No
	No
	Yes
	No
	Portfolio/watchlist analysis
	No
	No
	Yes
	No
	Alerts on prediction changes
	No
	No
	Yes
	No
	View prediction history/accuracy
	No
	No
	Yes
	No
	Export results (CSV)
	No
	No
	Yes
	No
	View explainability features (evidence sentences, confidence scores, contributing factors)
	

No
	

No
	

Yes
	

Yes
	Manage users/subscriptions
	No
	No
	No
	Yes
	Update models/data sources
	No
	No
	No
	Yes
	Monitor system performance
	No
	No
	No
	Yes
	

4. Non-Functional Requirements 
4.1. Performance 
* Response time : For Free users, 95% of requests must complete within 2 seconds. For Subscription users, 95% of requests must complete within 1 second.
* Dashboard Load Time :  The initial dashboard with charts must load within 3 seconds in 95% of cases


4.2. Reliability & Availability
* During trading hours (Mon–Fri, 9:30–16:00 ET), the system must be operational at least 95% of the time.
* Data Freshness : Time from ingestion to display must be within 24 hours for Free users and within 1 minute for Subscription users.
4.3. Accessibility
* Color Contrast: Text and background must have a contrast ratio of at least 4.5:1 for normal text and 3:1 for large text (18pt+ or 14pt bold).
* Alt Text: 100% of images and icons must include meaningful alternative text for screen readers.
* Form Labels: All form fields (e.g., login, stock search) must have explicit labels so that assistive technologies like screen readers can announce their purpose. Using only placeholder text is not sufficient, because placeholders disappear once users start typing and they are not treated as proper labels by accessibility tools.
Incorrect Example (placeholder only)
<input type="text" placeholder="Enter stock symbol">
Correct Example (with <label> tag)
<label for="stockSymbol">Stock Symbol</label>
<input type="text" id="stockSymbol" name="stockSymbol">
* Error Messages: Input errors must trigger a visible message and a screen-reader friendly alert within 1 second.
4.4. Usability
* Learnability : First-time users must be able to complete the basic task — search a stock symbol and view sentiment results — in no more than 2 clicks.
* Task Success Rate : In a usability test with at least 5 participants, at least 90% of them must successfully complete the stock sentiment search task without assistance.
* Report Export : Exporting sentiment results as a CSV file with up to 5,000 rows must complete within 5 seconds.
4.5. Explainability
* Explanations shall be provided consistently for 100% of prediction results.
* Contributing factor summaries for premium users shall not exceed three items and must be expressed in plain language.


5. End User Use Cases
5.1. Basic Use Cases
5.1.1. Use Case ID: UC-A01
Actor: Anonymous User
Use Case Name: Sign up for an account
Trigger: User clicks "Sign Up" button
Preconditions: User is not registered
Basic Flow:
1. Open sign-up page
2. Enter name, email, password and other user information
3. Update user data to the Account database
Exceptions: If user inputs invalid character popup will inform the user 
Outcome: User account is created and user becomes a Basic User.
5.1.2. Use Case ID: UC-B01
Actor: Basic User (Registered User)
Use Case Name: Upgrade to Premium Subscription
Trigger: User clicks “Upgrade to Premium”
Preconditions: User is logged in as a Basic User
Basic Flow:
1. Open subscription page
2. Select premium plan
3. Enter payment details
4. Confirm purchase and receive confirmation email
5. User data updated in the Account database
Exceptions: If user inputs invalid information, popup will inform the user 
Outcome: User account is upgraded to Premium and premium features are unlocked
5.1.3. Use Case ID: UC-P01
Actor: Premium User
Use Case Name: Analyze entire watchlist
Trigger: Open Watchlist
Preconditions: Premium service active for the user
Basic Flow:
1. Load watchlist
2. Batch predict
3. Display table/cards
Exceptions: 
1. System timeout or API error
2. If user did not have no item in watchlist, display message “Your Watchlist is empty”
Outcome: User sees predictions for all stocks in their watchlist. 
5.1.4. Use Case ID: UC-P02
Actor: Premium User
Use Case Name: Receive prediction alerts
Trigger: Significant prediction update or user preference change detected for watched stocks  
Preconditions: Premium service active for the user and alerts function is turned on
Basic Flow:
1. Monitor threshold
2. Generate alert
3. Send email/app push
Exceptions: 
1. Alert delivery fails due to email server or push notification outage
2. Multiple rapid prediction changes occur within a short period
Outcome: User notified promptly
5.1.5. Use Case ID: UC-P03
Actor: Premium User
Use Case Name: View plain-language explanations
Trigger: Click “Explain”
Preconditions: Premium service active for the user
Basic Flow:
1. Retrieve model’s explanation data
2. Generate a human-readable summary of the main factors influencing the prediction
3. Display the explanation clearly on the UI
Exceptions: 
1. Explanation data unavailable
2. Model returns incomplete or low-confidence reasoning
3. System timeout or API error
Outcome: User understands “why”
5.1.6. Use Case ID: UC-S01 
Actor: Anonymous, Basic, Premium User 
Use Case Name: Search for a stock ticker and view stock info & prediction 
Trigger: Enter ticker symbol in search bar 
Preconditions: None 
Basic Flow: 
1. Validate ticker input 
2. Retrieve latest market data and prediction 
3. Display stock information and next-day forecast, with tier-based detail 
Exceptions: 
1. Invalid ticker symbol 
2. Market data API unavailable 
3. Prediction service timeout 
Outcome: User sees relevant stock information and predictions appropriate for their tier
5.1.7. Use Case ID: UC-S02 
Actor: Basic, Premium User 
Use Case Name: Save favorite tickers (watchlist) 
Trigger: Click “Save” on ticker detail page 
Preconditions: User logged in 
Basic Flow: 
1. Add ticker to watchlist 
2. Confirm visually that it was saved 
3. Add or Update ‘watchlist’ data of the user in the Account Database
4. Sync watchlist across devices 
Exceptions: 
1. Watchlist storage unavailable 
2. User reached max saved tickers (Basic tier limit) 
3. Database write failure 
Outcome: Ticker is successfully saved and available for future tracking
5.1.8. Use Case ID: UC-S03 
Actor: Basic, Premium User 
Use Case Name: View model explanation (technical) 
Trigger: Click “View Explanation” on prediction 
Preconditions: Logged in as Basic or Premium user 
Basic Flow: 
1. Retrieve model explanation data (feature importances, contributions) 
2. Display numeric/graphical output (bar chart, feature contribution) 
3. Provide partial confidence indicator 
Exceptions: 
1. Explanation data unavailable 
2. Incomplete or corrupted results 
3. Visualization fails → display text fallback 
Outcome: User sees a technical breakdown of factors influencing prediction
5.2. Admin Use Cases
5.2.1. Use Case ID: UC-AD01
Actor: Admin
Use Case Name: View usage statistics
Trigger: Click “Analytics”
Preconditions: Admin logged in with analytics permission
Basic Flow:
1. Retrieve aggregated usage metrics
2. Generate charts (top searched tickers, active users, feature usage)
3. Display dashboard with filters
Exceptions:
1. Metrics service unavailable
2. Data latency or incomplete logs
3. Visualization rendering error
Outcome: Admin sees clear usage trends and engagement insights
5.2.2. Use Case ID: UC-AD02 
Actor: Admin 
Name: Update AI models and data sources 
Trigger: Click “Models & Data” → “Update/Deploy” 
Preconditions: Admin logged in with deployment permission 
Basic Flow: 
1. Upload/select model or edit data-source config 
2. Validate version and run health checks 
3. Deploy update and confirm status 
Exceptions: 
1. Validation fails or checksum mismatch 
2. Data connector credentials invalid 
3. Deployment timeout or rollback triggered 
Outcome: New model/data config is live and healthy
5.2.3. Use Case ID: UC-AD03 
Actor: Admin 
Use Case Name: Manage user accounts 
Trigger: Click “Users” 
Preconditions: Admin logged in with user-management permission 
Basic Flow: 
1. Search and select user account 
2. Edit profile/role or suspend/reactivate 
3. Save changes and notify user if needed 
Exceptions: 
1. Insufficient permissions for role change 
2. Conflicting state (e.g., billing lock) 
3. Write failure to user directory 
Outcome: User account state updated correctly
5.3. Detailed AI Use Case - Stock Prediction & Sentiment Analysis
Title: AI-Powered Stock Prediction with Sentiment Context
Primary Actor: Basic User (extends to Premium User with advanced features)
Stakeholders & Interests:
* Basic User wants a simple, trustworthy next-day prediction and a quick sentiment snapshot.
* Premium User wants portfolio-wide evaluations, alerts, explanations, prediction history, and insights into big investor activity.
* Admin wants to ensure the AI models and data pipelines are accurate, scalable, and properly managed.
Preconditions
* The AI system is trained on historical stock data, financial metrics, and sentiment data.
* External data sources (news APIs, financial filings, institutional trading disclosures) are up-to-date.
* User is connected to the app (logged in for premium features).
Main Flow
* User Inputs Ticker
* Basic user enters a stock ticker (e.g., AAPL).
* Premium user uploads or selects from their watchlist.
AI Model Evaluation
* The system retrieves recent stock performance, fundamentals, and sentiment data.
* The AI runs predictive models to forecast short-term performance.
* If confidence ≥95%, the system assigns a clear rating (Good / Neutral / Bad).
* If confidence <95%, the system flags as “trend only, not reliable.”
Sentiment Analysis
* The AI scans financial news, social posts, and filings.
* Produces a summary sentiment score (positive / neutral / negative).
* Basic users see a limited summary (e.g., “News sentiment positive: strong earnings report”).
* Premium users get detailed plain-language reasoning (e.g., “Positive sentiment driven by Apple’s strong quarterly EPS and increased institutional purchases”).
Result Display
* Basic User: sees next-day prediction + limited sentiment snapshot.
* Premium User: sees full watchlist predictions, alerts (if enabled), reasoning, accuracy history, and major investor activity.
Postconditions
* Users receive reliable AI predictions and context for decision-making.
* Premium users gain deeper insights and monitoring tools.
* Admin has visibility into usage and performance to ensure trustworthiness.
  

5.4. Use Case: Evaluate a stock for next-day performance 
Description:
A retail investor enters a stock ticker and taps Predict. The system gathers market/fundamental data and recent news/social signals, then uses AI to forecast next-day movement. If any class (Up/Down/Neutral) reaches ≥95% confidence, the app issues a clear rating; otherwise it presents trend-only guidance. A concise sentiment summary explains the current mood; premium users also get plain-language reasons, history, alerts, and big-player activity.
Primary actor:
Retail investor (Basic user) (Premium user is an extension of this actor)
Goal in context:
Know if a ticker is high-confidence (≥95%) Up/Down/Neutral for tomorrow, or otherwise see trend-only guidance, with a short sentiment snapshot.
Trigger:
User taps Predict after entering/selecting a ticker (or watchlist item for Premium).
Step
	Actor
	Sys
	1
	Enters ticker (e.g., AAPL) and taps Predict.
	2. Validates ticker/market; checks freshness; shows “Processing…”
	

	

	3. Retrieves data: recent OHLCV, fundamentals, earnings/events; news/social; institutional/insider filings. Normalizes & deduplicates.
	

	

	4. Runs predictive model → class probabilities for Up / Down / Neutral. Applies 95% confidence gate.
	

	

	5. Runs sentiment pipeline → aggregates sources → positive/neutral/negative + top drivers.
	

	

	6. Persists prediction (time, inputs, outputs). For Premium: append to history, evaluate alerts, fetch big-player activity.
	7
	Views result card.
	8. If ≥95%: shows rating (Good/Neutral/Bad), probability, and concise sentiment. If <95%: shows trend-only with caveat. Premium also sees plain-language explanation, history accuracy, alerts toggle, big-player moves.
	9
	(Optional) Adds to watchlist, enables alerts, or checks another ticker.
	10. Confirms watchlist/alerts; schedules background checks (Premium).
	

Exceptions
* 2.1 Invalid/unsupported ticker → Return “Ticker not found/unsupported market.”

* 3.1 Data source timeout/rate-limit → Fallback to cached data; if stale, inform: “Live data unavailable; showing latest cached snapshot.”

* 4.1 Model returns <95% for all classes → No rating; display trend-only with disclaimer.

* 4.2 Model unavailable/drift detected → “Predictions temporarily unavailable; we’re recalibrating the model.” (Admin alerted.)

* 5.1 Sentiment sources unavailable → Show prediction (if available) with “Sentiment temporarily unavailable.”

* 6.1 Alert scheduling failure (Premium) → “Couldn’t set alerts; please retry.”

Priority: High
Channel: Web or mobile (or both)




   6. Research References
Research on Artificial Intelligence for Stock Market Analysis
 
History of Using AI to Predict Stock Market
 
The problem of using artificial intelligence techniques to predict future stock performance is not a new problem and there is a plethora of existing literature on the subject. To start, we looked at a literature review to see how the field has evolved and the current trends in the research. According to Chopra et al. (2024) there are three distinct phases:
 
Early Neural Networks and Forecasting (1990 – 2008)
   * focused primarily on artificial neural networks (ANNs)
   * studies examined whether AI techniques could outperform traditional models
 
Expansion into Hybrids, Feature Selection, and Investor Sentiment (2009 – 2015)
   * research broadened to include hybrid models: ANNs with genetic algorithms (GAs), particle swarm optimization (PSO), fuzzy systems
   * models also integrated feature selection techniques
   * investor sentiment became arose as a meaningful variable in models, included through behavioural and textual signals
 
Deep Learning, Sentiment Analysis, and Hybrid Dominance (2016 – 2022)
   * research is dominated by deep learning models, more specifically CNNs, RNNs, and LSTM models
   * models are often paired with attention mechanisms
   * models generally use a blend of multiple AI techniques
   * sentiment analysis is more prominent, with researchers mining news articles, social media and disclosures to capture investor sentiment
 
From this we see that the most recent methods for stock forecasting involve deep learning, use attention mechanisms, and integrate sentiment analysis (Chopra et al., 2024). While neural networks have long dominated the field of predictive market analysis, emerging techniques like fuzzy logic and attentional mechanisms can be blended with these models to enhance performance (Chopra & Sharma, 2021). Fuzzy logic is a method that converts precise inputs into fuzzy categories, allowing machine learning models to work with values where the input is imprecise and imperfect (Geeks for Geeks, 2025). Integrating fuzzy logic into neural networks allows the model to work better with vague and incomplete real-world data, improving its performance at predicting a stock’s performance.
 
Current Approaches in Market Forecasting with AI
 
We also looked into how others in the field have approached the problem. What we found is that neural network models are the most popular approach to stock market forecasting.
 
Inputs
The inputs for these models commonly include: stock market data, technical indicators (e.g. RSI, MACD), economic variables (e.g. interest rates, exchange rates), company financial indicators (e.g. price-to-earnings ratio), and textual sentiment data (e.g. news, social media posts) (Chopra & Sharma, 2021).
 
Pre-Processing
Common methods for pre-processing include normalization, principle component analysis (PCA), and Z-score transformation (Chopra & Sharma, 2021).
 
Model Selection
The most popular methods in recent years are ANNs, RNNs, LSTMs, and CNNs. Some models also integrate fuzzy logic and deep learning (Chopra & Sharma, 2021).
 
Training and Optimization
Most commonly models are trained and optimized using backpropagation and gradient descent (Chopra & Sharma, 2021). However, there are other options such as: genetic algorithms, Levenburg-Marquadt, stochastic gradient conjugate algorithms (Chopra & Sharma, 2021).
Optimizers used for training include Adam, LightGBM, and XGBoost (Chopra & Sharma, 2021).
 
Measuring Performance
Research uses a mix of both statistical metrics like MAE or RMSE and profit-based measures, with the latter being increasingly emphasized as statistical measures often miss financial relevance (Chopra & Sharma, 2021).
 
Areas to Improve On
One crucial area we looked at when reviewing others’ approaches is what gaps exist in the research and where there is room for improvement. To this end, Chopra and Sharma (2021) have laid out some areas where our model can improve on existing approaches:
   * we can use a systematic approach to feature selection by integrating feature engineering techniques we have learned—many studies use an ad-hoc approach to feature selection
   * integration of sentiment data is currently an underexplored
   * we can use explainability methods like SHAP or LIME to improve model interpretability—this is another crucial missing piece in existing approaches
   * using our model in emerging global markets, as most existing models focus on US or Chinese markets
 
Specific Studies on Investing with AI
 
To get a better idea of the techniques used in practice and draw inspiration for our own project, we will also investigate some papers on specific models used to predict stock prices. We will also look closely at using agentic AI as that is one type of model we can use to handle complex user inquiries and make decisions based on multiple different types of data.
 
Model A: Dynamic Deep Neural Network with “Small Data”
This model uses a small-data approach as it only considers standard market data (S&P 500 price and volume) and the VIX volatility index. The model will independently decide whether to buy, sell, or not trade a certain stock based on its inputs and what it’s learned through reinforcement learning. It achieved an annualized return of 14.92% compared to a passive S&P 500 return of 3.09% while minimizing volatility/risk (Finnegan et al., 2024).
 
FinVision: Multi-Agent Framework for Stock Market Prediction
FinVision is a multi-modal, multi-agent framework for financial trading that uses LLM-based agents (Fatemi & Hu, 2024). Each agent specializes in processing a type of data like news, candlestick charts, trading signals (Fatemi & Hu, 2024). A unique reflection module analyzes past signals to improve future decisions—ablation studies showed this was critical as it greatly boosted performance (Fatemi & Hu, 2024).
 
FinVision is built as a team of cooperating agents, each handling a part of the trading workflow. The Summary Module condenses financial news into trading-relevant insights (Fatemi & Hu, 2024). The Technical Analyst Module uses vision-enabled LLMs to interpret candlestick charts and indicators (Fatemi & Hu, 2024). The Reflection Module analyzes past trades and trading signals to refine strategies, offering both textual and visual insights (Fatemi & Hu, 2024). The Final Decision Module integrates outputs from all other modules to recommend BUY/SELL/HOLD actions with position sizing (Fatemi & Hu, 2024).
 
Implementation uses the LangGraph library, with GPT-4o-mini powering most agents and o1-mini for advanced reasoning in final predictions (Fatemi & Hu, 2024). Because all agents within FinVision are LLMs, it is also able to generate human-readable explanations of its recommendations (Fatemi & Hu, 2024).
 
ElliottAgents: Multi-Agent Approach Combining EWP, LLMs, RAG and DRL
This is another multi-agent model that combines the Elliott Wave Principle (EWP) with LLMs, RAG, and deep reinforcement learning (DRL) to predict stock prices (Wawer et al., 2024). The Elliott Wave Principle is a traditional technical analysis method that posits that markets move in repeating 5-wave impulse and 3-wave corrective cycles (Wawer et al., 2024). They also use multiple agents to split tasks between the specialized agents. Deep reinforcement learning is also used to adapt and learn from previous feedback (Wawer et al., 2024). This model combines traditional financial analysis with advanced AI techniques.
 
The system is built in Python, using LangGraph within LangChain for multi-agent orchestration (Wawer et al., 2024). Agents are divided into specialized roles, such as Data Engineer, Elliott Wave Analyst, Backtester (using DRL), Technical Analysis Expert (with RAG knowledge base), Investment Advisor, and Report Writer (Wawer et al., 2024). A centralized Coordinator Agent manages workflows (Wawer et al., 2024). Together, these agents ingest data (via yfinance), identify Elliott wave structures, backtest strategies with DRL, and produce interpretable analyses with annotated charts and written reports (Wawer et al., 2024). Each agent has access to specific tools based on its task, e.g. chart generators, databases, etc.) (Wawer et al., 2024).
 
It integrates traditional financial analysis methods like EWP by using RAG to pull in EWP related documentation and other related financial knowledge (Wawer et al., 2024). This allows theoretical knowledge to be consistently applied during pattern analysis and reduces model hallucination (Wawer et al., 2024).
 


 
Key Takeaways
   * neural nets with fuzzy logic seems like a promising model type (maybe also with an attention mechanism)
◦          as seen in Model A paper, we don’t always need big data to predict short-term stock value accurately
◦          when creating this model select features systematically, not on an ad hoc basis
   * should evaluate model based on both statistical and profit-based measures
   * include risk information as well as profit: even if a stock seems likely to go up it’s also good to know how risky the investment will be
◦          VIX volatility index
◦          Sharpe/Sortino ratios
◦          drawdowns
   * agentic model structure: multiple agents with each agent focusing on a specific sub-task with output being put together and turned into final decision/output by a decision-making agent
   * add a reflection module/agent to analyze past trades and signals to refine its strategies/recommendations
   * integrate traditional financial analysis methods (like Elliott Wave Principle) into our agentic AI system
◦          can use RAG to pull in domain knowledge
   * model explainability is key: this is natural when using LLM-based agents but harder if using neural networks (need to add explainability modules and convert results to human-readable explanations)

References
Chopra, R., & Sharma, G. D. (2021). Application of artificial intelligence in stock market forecasting: A critique, review, and research agenda. Journal of Risk and Financial Management, 14(11), 1–34. https://doi.org/10.3390/jrfm14110526
Chopra, R., Sharma, G. D., & Pereira, V. (2024). Identifying Bulls and bears? A bibliometric review of applying artificial intelligence innovations for stock market prediction. Technovation, 135, Article 103067. https://doi.org/10.1016/j.technovation.2024.103067
Fatemi, S., & Hu, Y. (2024). FinVision: A Multi-Agent Framework for Stock Market Prediction. Proceedings of the 5th ACM International Conference on AI in Finance, 582–590. https://doi.org/10.1145/3677052.3698688
Finnegan, C. J., McCann, J. F., & Moutari, S. (2024). Less is more: AI decision-making using dynamic deep neural networks for short-term stock index prediction. Ithaca: Retrieved from http://ra.ocls.ca/ra/login.aspx?inst=centennial&url=https://www.proquest.com/working-papers/less-is-more-ai-decision-making-using-dynamic/docview/3095811055/se-2
Geeks for Geeks. (22 Aug 2025). Fuzzy Logic | Introduction. https://www.geeksforgeeks.org/artificial-intelligence/fuzzy-logic-introduction/
Wawer, M., Chudziak, J. A., & Niewiadomska-Szynkiewicz, E. (2024). Large language models and the Elliott wave principle: A multi-agent deep learning approach to big data analysis in financial markets. Applied Sciences, 14(24), 11897. doi:https://doi.org/10.3390/app142411897
