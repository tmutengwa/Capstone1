# Food Demand Forecasting

A machine learning project to predict meal demand across fulfillment centers for a meal delivery company.

## Implementation Approach

✅ **sklearn Pipeline with DictVectorizer** - Production-ready ML
✅ **Automatic Model Selection** - Compares 7 models, uses best
✅ **Single Artifact Deployment** - Complete pipeline in one pickle file
✅ **FastAPI Service** - RESTful API for predictions

## Problem Statement

This project helps meal delivery fulfillment centers predict demand for the upcoming weeks to optimize:
- **Procurement Planning**: Stock raw materials efficiently (mostly perishable items replenished weekly)
- **Staffing Optimization**: Plan workforce based on predicted demand

### Objective
Predict the number of orders (`num_orders`) for the next 10 weeks (weeks 146-155) for each center-meal combination.

### Evaluation Metric
**100 × RMSLE** (Root Mean Squared Logarithmic Error)

```
RMSLE = sqrt(mean((log(actual + 1) - log(predicted + 1))^2))
```

## Dataset

The project uses historical data from weeks 1-145 to predict demand for weeks 146-155.

### Data Files

1. **train.csv** - Historical demand data
   - `id`: Unique identifier
   - `week`: Week number
   - `center_id`: Fulfillment center ID
   - `meal_id`: Meal ID
   - `checkout_price`: Final price (with discounts, taxes, delivery)
   - `base_price`: Original meal price
   - `emailer_for_promotion`: Email promotion sent (0/1)
   - `homepage_featured`: Featured on homepage (0/1)
   - `num_orders`: **Target variable** - Number of orders

2. **test_QoiMO9B.csv** - Test data (same features except `num_orders`)

3. **fulfilment_center_info.csv** - Center information
   - `center_id`: Fulfillment center ID
   - `city_code`: City identifier
   - `region_code`: Region identifier
   - `center_type`: Type of center (anonymized)
   - `op_area`: Operating area in km²

4. **meal_info.csv** - Meal information
   - `meal_id`: Meal identifier
   - `category`: Meal type (beverages/snacks/soups)
   - `cuisine`: Cuisine type (Indian/Italian/Thai, etc.)

## Project Structure

```
food_demand/
├── data/                          # Data directory
│   ├── train.csv
│   ├── test_QoiMO9B.csv
│   ├── fulfilment_center_info.csv
│   └── meal_info.csv
├── food_demand.ipynb              # Jupyter notebook (EDA & modeling)
├── train.py                       # Model training script
├── predict.py                     # FastAPI prediction service
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
├── .dockerignore                  # Docker ignore file
├── .gitignore                     # Git ignore file
├── README.md                      # This file
├── final_model.pkl               # Trained model (generated)
├── feature_cols.pkl              # Feature columns (generated)
├── label_encoders.pkl            # Label encoders (generated)
└── submission.csv                # Predictions (generated)
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) Docker for containerization

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd food_demand
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Exploratory Data Analysis (Jupyter Notebook)

Open and run the notebook to explore data and understand the modeling process:

```bash
jupyter notebook food_demand.ipynb
```

The notebook includes:
- Data loading and merging
- Data cleaning and preparation
- Exploratory data analysis (EDA)
- Feature engineering
- Multiple model comparison (Linear Regression, Lasso, Ridge, Decision Tree, Random Forest, Gradient Boosting, XGBoost)
- Hyperparameter tuning
- Feature importance analysis
- Model evaluation

### 2. Train the Model

Train the final model and save artifacts:

```bash
python train.py
```

This will:
- Load and preprocess data
- Apply feature engineering
- Train the XGBoost model
- Evaluate performance
- Save model artifacts:
  - `final_model.pkl` - Trained model
  - `feature_cols.pkl` - Feature column names
  - `label_encoders.pkl` - Categorical encoders

### 3. Run the Prediction Service

#### Option A: Run Locally

Start the FastAPI server:

```bash
uvicorn predict:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

#### Option B: Run with Docker

1. **Build the Docker image**
   ```bash
   docker build -t food-demand-api .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 food-demand-api
   ```

The API will be available at `http://localhost:8000`

### 4. Access API Documentation

Once the service is running, visit:
- **Interactive API docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/
```

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "week": 146,
    "center_id": 55,
    "meal_id": 1885,
    "checkout_price": 136.83,
    "base_price": 152.0,
    "emailer_for_promotion": 0,
    "homepage_featured": 0,
    "city_code": 647,
    "region_code": 56,
    "center_type": "TYPE_A",
    "op_area": 3.5,
    "category": "Beverages",
    "cuisine": "Thai"
  }'
```

**Response:**
```json
{
  "predicted_orders": 152.34,
  "week": 146,
  "center_id": 55,
  "meal_id": 1885
}
```

### Batch Predictions
```bash
curl -X POST "http://localhost:8000/predict_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "predictions": [
      {
        "week": 146,
        "center_id": 55,
        "meal_id": 1885,
        ...
      },
      {
        "week": 147,
        "center_id": 55,
        "meal_id": 1885,
        ...
      }
    ]
  }'
```

## Feature Engineering

The model uses the following engineered features:

1. **Price Features**
   - `discount`: Difference between base and checkout price
   - `discount_percentage`: Discount as percentage of base price

2. **Promotional Features**
   - `total_promotion`: Sum of emailer and homepage features

3. **Time-based Features**
   - `week_mod_4`: Weekly pattern (monthly cycle)
   - `week_mod_13`: Weekly pattern (quarterly cycle)
   - `week_mod_52`: Weekly pattern (yearly cycle)

4. **Encoded Categorical Features**
   - Label-encoded: center_id, meal_id, city_code, region_code, center_type, category, cuisine

## Model Performance

The final model is an XGBoost Regressor with the following hyperparameters:
- `n_estimators`: 200
- `max_depth`: 7
- `learning_rate`: 0.05
- `subsample`: 0.9

Performance metrics are logged during training and evaluation.

## Deployment

### Local Deployment
The FastAPI service can be deployed locally as shown in the usage section.

### Cloud Deployment Options

1. **Docker-based platforms**: Deploy the Docker container to:
   - AWS ECS/EKS
   - Google Cloud Run
   - Azure Container Instances
   - Heroku
   - DigitalOcean App Platform

2. **Platform-as-a-Service**:
   - Deploy directly to platforms supporting Python/FastAPI:
   - Railway
   - Render
   - Fly.io

### Example: Deploy to Google Cloud Run

```bash
# Build and tag image
docker build -t gcr.io/[PROJECT-ID]/food-demand-api .

# Push to Google Container Registry
docker push gcr.io/[PROJECT-ID]/food-demand-api

# Deploy to Cloud Run
gcloud run deploy food-demand-api \
  --image gcr.io/[PROJECT-ID]/food-demand-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Example: Deploy to AWS ECS

```bash
# Create ECR repository
aws ecr create-repository --repository-name food-demand-api

# Build and tag
docker build -t food-demand-api .
docker tag food-demand-api:latest [ACCOUNT-ID].dkr.ecr.us-east-1.amazonaws.com/food-demand-api:latest

# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin [ACCOUNT-ID].dkr.ecr.us-east-1.amazonaws.com
docker push [ACCOUNT-ID].dkr.ecr.us-east-1.amazonaws.com/food-demand-api:latest

# Deploy using ECS CLI or AWS Console
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov httpx

# Run tests
pytest tests/ -v --cov=.
```

### Code Quality

```bash
# Install linting tools
pip install black flake8 isort

# Format code
black .
isort .

# Lint code
flake8 .
```

## Troubleshooting

### Model files not found
Ensure you've run `python train.py` before starting the prediction service.

### Port already in use
Change the port number:
```bash
uvicorn predict:app --host 0.0.0.0 --port 8080
```

### Docker build fails
- Ensure Docker is running
- Check that all required files exist
- Verify requirements.txt is complete

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## License

This project is licensed under the MIT License.

## Authors

- Tongai Mutengwa

## Acknowledgments

- Dataset provided by Analytics Vidya
- Built with scikit-learn, Random Forest, and FastAPI
- Inspired by real-world demand forecasting challenges

# Deploying Food Demand Forecasting API to AWS Lambda

This guide outlines the steps to deploy the FastAPI application to AWS Lambda using Docker.

## Prerequisites
- AWS Account with appropriate permissions
- AWS CLI installed and configured (`aws configure`)
- Docker installed
- SAM CLI (optional but recommended for local testing)

## Step 1: Prepare the Dockerfile for Lambda

AWS Lambda requires a specific runtime interface client. You need to modify your `Dockerfile` to be compatible with Lambda or use the AWS Lambda Adapter.

**Option A: Using AWS Lambda Adapter (Recommended for FastAPI)**
This allows you to run standard web apps on Lambda without changing code.

Update your `Dockerfile` to add the adapter:

```dockerfile
# Use official Python runtime as base image
FROM python:3.12-slim

# Install AWS Lambda Web Adapter
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.7.0 /lambda-adapter /opt/extensions/lambda-adapter

# Set working directory in container
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application files
COPY predict.py .
COPY final_model.pkl .
COPY feature_cols.pkl .

# Run the application
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Step 2: Create an ECR Repository

Create a repository in Amazon Elastic Container Registry (ECR) to store your Docker image.

```bash
aws ecr create-repository --repository-name food-demand-api --region us-east-1
```

*Replace `us-east-1` with your preferred region.*

## Step 3: Build and Push Docker Image

1.  **Login to ECR:**
    ```bash
    aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <YOUR_ACCOUNT_ID=955423509456>.dkr.ecr.us-east-1.amazonaws.com
    ```

2.  **Build the Image:**
    ```bash
    docker build -t food-demand-api .
    ```

3.  **Tag the Image:**
    ```bash
    docker tag food-demand-api:latest <YOUR_ACCOUNT_ID=955423509456>.dkr.ecr.us-east-1.amazonaws.com/food-demand-api:latest
    ```

4.  **Push to ECR:**
    ```bash
    docker push <YOUR_ACCOUNT_ID=955423509456>.dkr.ecr.us-east-1.amazonaws.com/food-demand-api:latest
    ```

## Step 4: Create Lambda Function

1.  Go to the AWS Lambda Console.
2.  Click **Create function**.
3.  Select **Container image**.
4.  Name: `food-demand-api`.
5.  Container image URI: Select the image you pushed to ECR.
6.  Architecture: Select `x86_64` (or `arm64` if you built on M1/M2 Mac, ensuring your Docker build matches).

## Step 5: Configure Lambda

1.  **Configuration** tab -> **General configuration** -> **Edit**.
    *   **Memory:** Increase to at least 1024 MB (ML models need memory).
    *   **Timeout:** Increase to 30 seconds or more.
2.  **Configuration** tab -> **Environment variables**.
    *   Add any necessary env vars.

## Step 6: Expose via Function URL or API Gateway

**Option A: Function URL (Simplest)**
1.  Go to **Configuration** -> **Function URL**.
2.  Click **Create function URL**.
3.  Auth type: `NONE` (for public access) or `AWS_IAM`.
4.  Copy the Function URL.

**Option B: API Gateway**
1.  Go to **Add trigger**.
2.  Select **API Gateway**.
3.  Create a new HTTP API.
4.  Security: Open (or JWT).

## Step 7: Test the Deployment

Use the Function URL (or API Gateway URL) to test the health endpoint.

```bash
curl -i https://jeraevxv6d4mjcsb22ovqlzg3y0lrmtm.lambda-url.us-east-1.on.aws/health
```

Test a prediction:

```bash
curl -X 'POST' \
  'https://jeraevxv6d4mjcsb22ovqlzg3y0lrmtm.lambda-url.us-east-1.on.aws/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "week": 146,
  "center_id": 55,
  "meal_id": 1885,
  "checkout_price": 136.83,
  "base_price": 152.0,
  "emailer_for_promotion": 0,
  "homepage_featured": 0,
  "city_code": 647,
  "region_code": 56,
  "center_type": "TYPE_A",
  "op_area": 3.5,
  "category": "Beverages",
  "cuisine": "Thai"
}'
```
